from pathlib import Path

DEBUG_DIR = Path(__file__).resolve().parent
BUNDLE_ROOT = DEBUG_DIR.parents[1]
import json
import os

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, logging as hf_logging
from trl.trainer.utils import flush_left, pad_to_length, selective_log_softmax

from src.dataset import make_dpo_data_module
from src.params import DataArguments, DPOArguments
from train.monkey_patch_forward import replace_gemma3_forward, _get_gated_projected_image_features
from train.train_dpo import configure_llm, configure_vision_tower
from train.train_sft import configure_dual_input_gate, maybe_restore_dual_input_gate_from_checkpoint


hf_logging.set_verbosity_error()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")

REPO_ROOT = str(BUNDLE_ROOT)
MODEL_ID = os.environ.get(
    "MODEL_ID",
    str(BUNDLE_ROOT / "checkpoints/gemma3_4b_stage2_gate_l1_mask_sdpa/checkpoint-10293"),
)
DATA_PATH = os.environ.get("DATA_PATH", str(BUNDLE_ROOT / "data/eval/test_raw_with_shortcut_answer.json"))
IMAGE_FOLDER = os.environ.get("IMAGE_FOLDER", str(BUNDLE_ROOT / "data/playground_data/coco/train2014"))
GATE_TEXT_MODEL_ID = os.environ.get("GATE_TEXT_MODEL_ID", str(BUNDLE_ROOT / "models/siglip-so400m-patch14-384"))
ATTN = os.environ.get("ATTN", "eager")
VISION_ATTN = os.environ.get("VISION_ATTN", "sdpa")
DTYPE_NAME = os.environ.get("DTYPE", "bf16").lower()
INDEX = int(os.environ.get("INDEX", "0"))
BRANCH = os.environ.get("BRANCH", "both").lower()
TRAIN_MODE = os.environ.get("TRAIN_MODE", "0") == "1"
USE_AUTOCAST = os.environ.get("USE_AUTOCAST", "0") == "1"
INCLUDE_TTI = os.environ.get("TTI", "0") == "1"
USE_DUAL_INPUT_GATE = os.environ.get("USE_DUAL_INPUT_GATE", "0").lower() in {"1", "true", "yes"}


DTYPES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
}
DTYPE = DTYPES[DTYPE_NAME]


def stat(name, tensor, active_rows=None):
    x = tensor.detach()
    if active_rows is not None and x.ndim >= 3:
        flat = x.reshape(-1, x.shape[-1])
        x = flat.index_select(0, active_rows)
    finite = torch.isfinite(x)
    xf = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    print(
        f"[stat] {name} shape={tuple(x.shape)} dtype={x.dtype} "
        f"finite={bool(finite.all().item())} nan={int(torch.isnan(x).sum().item())} "
        f"inf={int(torch.isinf(x).sum().item())} min={float(xf.amin().item())} "
        f"max={float(xf.amax().item())} mean={float(xf.mean().item())}",
        flush=True,
    )


def row_bad_counts(name, tensor, active_rows, labels):
    flat = tensor.detach().reshape(-1, tensor.shape[-1])
    active = flat.index_select(0, active_rows)
    bad_counts = (~torch.isfinite(active)).sum(dim=-1).cpu().tolist()
    for i, count in enumerate(bad_counts):
        print(f"[row] {name} active_idx={i} label={labels[i]!r} bad_dims={int(count)}", flush=True)


def decode_active_labels(processor, input_ids, loss_mask):
    shifted_labels = torch.roll(input_ids, shifts=-1, dims=1)
    shifted_loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()
    active = shifted_labels[shifted_loss_mask].detach().cpu().tolist()
    return [processor.tokenizer.decode([tok]) for tok in active]


def load_model():
    replace_gemma3_forward(use_liger=False)
    args = DPOArguments(
        output_dir=str(DEBUG_DIR / "probe_single_out"),
        use_dual_input_gate=USE_DUAL_INPUT_GATE,
        gate_text_model_id=GATE_TEXT_MODEL_ID,
        freeze_gate_text_encoder=True,
        freeze_llm=False,
        freeze_vision_tower=True,
        freeze_projector=False,
        disable_flash_attn2=True,
        attn_implementation=ATTN,
        vision_attn_implementation=VISION_ATTN,
        bf16=DTYPE is torch.bfloat16,
        fp16=DTYPE is torch.float16,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        remove_unused_columns=False,
        gradient_checkpointing=False,
        report_to=[],
    )
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        attn_implementation=ATTN,
    ).cuda()
    configure_llm(model, args)
    configure_vision_tower(model, args, DTYPE, torch.device("cuda"))
    gate_tok = configure_dual_input_gate(model, args, DTYPE)
    if USE_DUAL_INPUT_GATE:
        maybe_restore_dual_input_gate_from_checkpoint(model, MODEL_ID)
    if getattr(model, "siglip_text_model", None) is not None:
        model.siglip_text_model.to(device="cuda", dtype=DTYPE)
    if getattr(model, "gate", None) is not None:
        model.gate.to(device="cuda", dtype=DTYPE)
    model.config.use_cache = False
    model.train(TRAIN_MODE)
    return model, gate_tok


def build_inputs(processor, gate_tok):
    data_module = make_dpo_data_module(
        processor=processor,
        data_args=DataArguments(data_path=DATA_PATH, image_folder=IMAGE_FOLDER, lazy_preprocess=True),
        gate_text_tokenizer=gate_tok,
        gate_text_max_length=64,
    )
    sample = data_module["train_dataset"][INDEX]
    batch = data_module["data_collator"]([sample])
    batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}

    branches = ["chosen", "rejected"] if BRANCH == "both" else [BRANCH]
    prompt_input_ids = torch.cat([batch["prompt_input_ids"] for _ in branches], dim=0)
    prompt_attention_mask = torch.cat([batch["prompt_attention_mask"] for _ in branches], dim=0)
    completion_ids = []
    completion_masks = []
    for branch in branches:
        completion_ids.append(batch[f"{branch}_input_ids"])
        completion_masks.append(batch[f"{branch}_attention_mask"])
    max_completion_length = max(x.shape[1] for x in completion_ids)
    completion_input_ids = torch.cat(
        [pad_to_length(x, max_completion_length, pad_value=processor.tokenizer.pad_token_id) for x in completion_ids],
        dim=0,
    )
    completion_attention_mask = torch.cat(
        [pad_to_length(x, max_completion_length, pad_value=0) for x in completion_masks],
        dim=0,
    )
    input_ids = torch.cat([prompt_input_ids, completion_input_ids], dim=1)
    attention_mask = torch.cat([prompt_attention_mask, completion_attention_mask], dim=1)
    loss_mask = torch.cat([torch.zeros_like(prompt_attention_mask), completion_attention_mask], dim=1)

    items = [attention_mask, input_ids, loss_mask]
    token_type_ids = None
    if INCLUDE_TTI:
        prompt_token_type_ids = torch.cat([batch["token_type_ids"] for _ in branches], dim=0)
        completion_token_type_ids = torch.zeros_like(completion_input_ids)
        token_type_ids = torch.cat([prompt_token_type_ids, completion_token_type_ids], dim=1)
        items.append(token_type_ids)
    flushed = flush_left(*items)
    attention_mask, input_ids, loss_mask = flushed[:3]
    if INCLUDE_TTI:
        token_type_ids = flushed[3]

    pixel_values = torch.cat([batch["pixel_values"] for _ in branches], dim=0)
    active_rows = torch.roll(loss_mask, shifts=-1, dims=1).bool().reshape(-1).nonzero(as_tuple=False).squeeze(1)
    inputs = {
        "branches": branches,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "token_type_ids": token_type_ids,
        "pixel_values": pixel_values,
        "active_rows": active_rows,
        "rows": data_module["train_dataset"].list_data_dict,
    }
    if USE_DUAL_INPUT_GATE:
        inputs["gate_input_ids"] = torch.cat([batch["gate_input_ids"] for _ in branches], dim=0)
        inputs["gate_attention_mask"] = torch.cat([batch["gate_attention_mask"] for _ in branches], dim=0)
    return inputs


def inspect(model, processor, inputs):
    row = json.load(open(DATA_PATH, "r"))[INDEX]
    print(
        f"[sample] idx={INDEX} qid={row.get('question_id')} image_id={row.get('image_id')} "
        f"answer={row.get('answer')!r} shortcut={row.get('shortcut_answer')!r} "
        f"question={row.get('question')!r}",
        flush=True,
    )
    print(
        f"[config] dtype={DTYPE} attn={ATTN} vision_attn={VISION_ATTN} "
        f"train_mode={TRAIN_MODE} autocast={USE_AUTOCAST} branch={inputs['branches']} "
        f"tti={INCLUDE_TTI} gate={USE_DUAL_INPUT_GATE}",
        flush=True,
    )
    active_labels = decode_active_labels(processor, inputs["input_ids"], inputs["loss_mask"])
    print(f"[active_labels] {active_labels}", flush=True)

    context = torch.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else torch.cuda.amp.autocast(enabled=False)
    with torch.no_grad(), context:
        stat("pixel_values", inputs["pixel_values"])
        vision = model.vision_tower(pixel_values=inputs["pixel_values"]).last_hidden_state
        stat("vision_last_hidden", vision)
        if USE_DUAL_INPUT_GATE:
            text_outputs = model.siglip_text_model(
                input_ids=inputs["gate_input_ids"],
                attention_mask=inputs["gate_attention_mask"],
                return_dict=True,
            )
            text_feat = (
                text_outputs.pooler_output
                if getattr(text_outputs, "pooler_output", None) is not None
                else text_outputs.last_hidden_state.mean(dim=1)
            )
            stat("gate_text_feat", text_feat)
        projected, patch_values = _get_gated_projected_image_features(
            model,
            inputs["pixel_values"],
            gate_input_ids=inputs.get("gate_input_ids"),
            gate_attention_mask=inputs.get("gate_attention_mask"),
        )
        stat("patch_values", patch_values)
        stat("projected_image_features", projected)

        llm_ids = inputs["input_ids"].clone()
        llm_ids[llm_ids == model.config.image_token_index] = 0
        embeds = model.get_input_embeddings()(llm_ids)
        image_mask = (inputs["input_ids"] == model.config.image_token_index).unsqueeze(-1).expand_as(embeds)
        merged = embeds.masked_scatter(image_mask, projected.to(embeds.device, embeds.dtype))
        stat("inputs_embeds_all", merged)
        stat("inputs_embeds_active", merged, inputs["active_rows"])

        cache_position = torch.arange(0, merged.shape[1], device=merged.device)
        causal_mask = model._update_causal_mask(
            inputs["attention_mask"],
            inputs["token_type_ids"],
            None,
            cache_position,
            merged,
            INCLUDE_TTI,
        )
        stat("causal_mask", causal_mask)
        outputs = model.language_model(
            attention_mask=causal_mask,
            inputs_embeds=merged,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
            cache_position=cache_position,
        )
        first_bad_layer = None
        for layer_idx, hidden in enumerate(outputs.hidden_states):
            stat(f"hidden_{layer_idx}_active", hidden, inputs["active_rows"])
            row_bad_counts(f"hidden_{layer_idx}", hidden, inputs["active_rows"], active_labels)
            active = hidden.reshape(-1, hidden.shape[-1]).index_select(0, inputs["active_rows"])
            if first_bad_layer is None and not bool(torch.isfinite(active).all().item()):
                first_bad_layer = layer_idx
                break
        stat("logits_active", outputs.logits, inputs["active_rows"])
        row_bad_counts("logits", outputs.logits, inputs["active_rows"], active_labels)

        labels = torch.roll(inputs["input_ids"], shifts=-1, dims=1)
        loss_mask = torch.roll(inputs["loss_mask"], shifts=-1, dims=1).bool()
        labels[~loss_mask] = 0
        per_token_logps = selective_log_softmax(outputs.logits, labels)
        per_token_logps[~loss_mask] = 0
        per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)
        stat("per_token_logps_active", per_token_logps.reshape(-1, 1), inputs["active_rows"])
        print(f"[result] first_bad_layer={first_bad_layer}", flush=True)


def main():
    model, gate_tok = load_model()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    inputs = build_inputs(processor, gate_tok)
    inspect(model, processor, inputs)


if __name__ == "__main__":
    main()
