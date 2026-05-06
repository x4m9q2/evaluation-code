from pathlib import Path

DEBUG_DIR = Path(__file__).resolve().parent
BUNDLE_ROOT = DEBUG_DIR.parents[1]
import gc
import os

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, logging as hf_logging

from src.dataset import make_dpo_data_module
from src.params import DataArguments, DPOArguments
from monkey_patch_forward import replace_gemma3_forward, _get_gated_projected_image_features
from train.train_dpo import configure_llm, configure_vision_tower
from train.train_sft import configure_dual_input_gate, maybe_restore_dual_input_gate_from_checkpoint
from trl.trainer.utils import flush_left, pad_to_length


hf_logging.set_verbosity_error()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
replace_gemma3_forward(use_liger=False)

MODEL_ID = os.environ.get(
    "MODEL_ID",
    str(BUNDLE_ROOT / "checkpoints/gemma3_4b_stage2_gate_l1_mask_sdpa"),
)
DATA_PATH = os.environ.get(
    "DATA_PATH",
    str(DEBUG_DIR / "test_raw_with_shortcut_answer_16.json"),
)
IMAGE_FOLDER = os.environ.get("IMAGE_FOLDER", str(BUNDLE_ROOT / "data/playground_data/coco/train2014"))
GATE_TEXT_MODEL_ID = os.environ.get("GATE_TEXT_MODEL_ID", str(BUNDLE_ROOT / "models/siglip-so400m-patch14-384"))
ATTN = os.environ.get("ATTN", "sdpa")
FORCE_VISION_ATTN = os.environ.get("FORCE_VISION_ATTN", "")
DTYPE = torch.bfloat16
INCLUDE_TTI = os.environ.get("TTI", "1") == "1"
USE_GATE = os.environ.get("USE_GATE", "1") == "1"


def stat(name, tensor, active_rows=None):
    x = tensor.detach()
    if active_rows is not None and x.ndim >= 3:
        flat = x.reshape(-1, x.shape[-1])
        x = flat.index_select(0, active_rows)
    finite = torch.isfinite(x)
    xf = torch.nan_to_num(x.float())
    print(
        f"[stat] {name} shape={tuple(x.shape)} dtype={x.dtype} finite={bool(finite.all().item())} "
        f"nan={int(torch.isnan(x).sum().item())} inf={int(torch.isinf(x).sum().item())} "
        f"min={float(xf.amin().item())} max={float(xf.amax().item())} mean={float(xf.mean().item())}",
        flush=True,
    )


def mem(tag):
    torch.cuda.synchronize()
    print(
        f"[mem] {tag} alloc={torch.cuda.memory_allocated() / 2**30:.2f}GiB "
        f"reserved={torch.cuda.memory_reserved() / 2**30:.2f}GiB",
        flush=True,
    )


def make_args():
    return DPOArguments(
        output_dir=str(DEBUG_DIR / "probe_out"),
        use_dual_input_gate=USE_GATE,
        gate_text_model_id=GATE_TEXT_MODEL_ID,
        freeze_gate_text_encoder=True,
        freeze_llm=False,
        freeze_vision_tower=True,
        freeze_projector=False,
        disable_flash_attn2=True,
        attn_implementation=ATTN,
        bf16=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        remove_unused_columns=False,
        gradient_checkpointing=False,
        report_to=[],
    )


def load_model(tag):
    args = make_args()
    print(f"[load] {tag} model={MODEL_ID} attn={ATTN} gate={USE_GATE}", flush=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        attn_implementation=ATTN,
    ).cuda()
    configure_llm(model, args)
    configure_vision_tower(model, args, DTYPE, torch.device("cuda"))
    if FORCE_VISION_ATTN:
        model.config.vision_config._attn_implementation = FORCE_VISION_ATTN
        model.vision_tower.config._attn_implementation = FORCE_VISION_ATTN
        print(f"[attn] forced vision_tower attn to {FORCE_VISION_ATTN}", flush=True)
    gate_tok = None
    if USE_GATE:
        gate_tok = configure_dual_input_gate(model, args, DTYPE)
        maybe_restore_dual_input_gate_from_checkpoint(model, MODEL_ID)
        model.siglip_text_model.to(device="cuda", dtype=DTYPE)
        model.gate.to(device="cuda", dtype=DTYPE)
    else:
        model.config.use_dual_input_gate = False
    model.config.use_cache = False
    model.eval()
    mem(f"after_load_{tag}")
    return model, gate_tok


def build_inputs(gate_tok):
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    module = make_dpo_data_module(
        processor,
        DataArguments(data_path=DATA_PATH, image_folder=IMAGE_FOLDER, lazy_preprocess=True),
        gate_text_tokenizer=gate_tok,
        gate_text_max_length=64,
    )
    batch = module["data_collator"]([module["train_dataset"][0]])
    for k, v in list(batch.items()):
        if torch.is_tensor(v):
            batch[k] = v.cuda()

    prompt_input_ids = torch.cat([batch["prompt_input_ids"], batch["prompt_input_ids"]], dim=0)
    prompt_attention_mask = torch.cat([batch["prompt_attention_mask"], batch["prompt_attention_mask"]], dim=0)
    pixel_values = torch.cat([batch["pixel_values"], batch["pixel_values"]], dim=0)
    max_completion_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])
    completion_input_ids = torch.cat(
        [
            pad_to_length(batch["chosen_input_ids"], max_completion_length, pad_value=processor.tokenizer.pad_token_id),
            pad_to_length(batch["rejected_input_ids"], max_completion_length, pad_value=processor.tokenizer.pad_token_id),
        ],
        dim=0,
    )
    completion_attention_mask = torch.cat(
        [
            pad_to_length(batch["chosen_attention_mask"], max_completion_length, pad_value=0),
            pad_to_length(batch["rejected_attention_mask"], max_completion_length, pad_value=0),
        ],
        dim=0,
    )
    input_ids = torch.cat([prompt_input_ids, completion_input_ids], dim=1)
    attention_mask = torch.cat([prompt_attention_mask, completion_attention_mask], dim=1)
    loss_mask = torch.cat([torch.zeros_like(prompt_attention_mask), completion_attention_mask], dim=1)
    items = [attention_mask, input_ids, loss_mask]
    token_type_ids = None
    if INCLUDE_TTI:
        prompt_token_type_ids = torch.cat([batch["token_type_ids"], batch["token_type_ids"]], dim=0)
        completion_token_type_ids = torch.zeros_like(completion_input_ids)
        token_type_ids = torch.cat([prompt_token_type_ids, completion_token_type_ids], dim=1)
        items.append(token_type_ids)
    flushed = flush_left(*items)
    attention_mask, input_ids, loss_mask = flushed[:3]
    if INCLUDE_TTI:
        token_type_ids = flushed[3]
    active_rows = torch.roll(loss_mask, shifts=-1, dims=1).bool().reshape(-1).nonzero(as_tuple=False).squeeze(1)
    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "token_type_ids": token_type_ids,
        "pixel_values": pixel_values,
        "active_rows": active_rows,
    }
    if USE_GATE:
        out["gate_input_ids"] = torch.cat([batch["gate_input_ids"], batch["gate_input_ids"]], dim=0)
        out["gate_attention_mask"] = torch.cat([batch["gate_attention_mask"], batch["gate_attention_mask"]], dim=0)
    return out


def inspect_forward(model, inputs, tag):
    print(f"[inspect] {tag} training={model.training}", flush=True)
    stat(f"{tag}/pixel_values", inputs["pixel_values"])
    with torch.no_grad(), torch.autocast("cuda", dtype=DTYPE):
        vision = model.vision_tower(pixel_values=inputs["pixel_values"]).last_hidden_state
        stat(f"{tag}/vision_last_hidden", vision)
        if USE_GATE:
            text_outputs = model.siglip_text_model(
                input_ids=inputs["gate_input_ids"],
                attention_mask=inputs["gate_attention_mask"],
                return_dict=True,
            )
            text_feat = text_outputs.pooler_output if text_outputs.pooler_output is not None else text_outputs.last_hidden_state.mean(dim=1)
            stat(f"{tag}/gate_text_feat", text_feat)
        projected, patch_values = _get_gated_projected_image_features(
            model,
            inputs["pixel_values"],
            gate_input_ids=inputs.get("gate_input_ids"),
            gate_attention_mask=inputs.get("gate_attention_mask"),
        )
        stat(f"{tag}/patch_values", patch_values)
        stat(f"{tag}/projected", projected)

        llm_ids = inputs["input_ids"].clone()
        llm_ids[llm_ids == model.config.image_token_index] = 0
        embeds = model.get_input_embeddings()(llm_ids)
        mask = (inputs["input_ids"] == model.config.image_token_index).unsqueeze(-1).expand_as(embeds)
        merged = embeds.masked_scatter(mask, projected.to(embeds.device, embeds.dtype))
        stat(f"{tag}/inputs_embeds_all", merged)
        stat(f"{tag}/inputs_embeds_active", merged, inputs["active_rows"])

        cache_position = torch.arange(0, merged.shape[1], device=merged.device)
        causal_mask = model._update_causal_mask(
            inputs["attention_mask"],
            inputs["token_type_ids"],
            None,
            cache_position,
            merged,
            False,
        )
        stat(f"{tag}/causal_mask", causal_mask)
        outputs = model.language_model(
            attention_mask=causal_mask,
            inputs_embeds=merged,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
            cache_position=cache_position,
        )
        for idx, hidden in enumerate(outputs.hidden_states):
            stat(f"{tag}/hidden_{idx}_active", hidden, inputs["active_rows"])
            active = hidden.reshape(-1, hidden.shape[-1]).index_select(0, inputs["active_rows"])
            if not bool(torch.isfinite(active).all().item()):
                break
        stat(f"{tag}/logits_active", outputs.logits, inputs["active_rows"])


def main():
    print(
        f"[start] device={torch.cuda.get_device_name(0)} attn={ATTN} "
        f"force_vision_attn={FORCE_VISION_ATTN or 'none'} gate={USE_GATE} tti={INCLUDE_TTI}",
        flush=True,
    )
    policy, policy_gate_tok = load_model("policy")
    ref, ref_gate_tok = load_model("ref")
    inputs = build_inputs(ref_gate_tok or policy_gate_tok)
    inspect_forward(ref, inputs, "ref_with_policy_resident")

    if os.environ.get("DELETE_POLICY", "1") == "1":
        del policy
        gc.collect()
        torch.cuda.empty_cache()
        mem("after_delete_policy")
        inspect_forward(ref, inputs, "ref_after_delete_policy")


if __name__ == "__main__":
    main()
