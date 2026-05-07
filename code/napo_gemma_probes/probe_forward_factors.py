from pathlib import Path

DEBUG_DIR = Path(__file__).resolve().parent
BUNDLE_ROOT = DEBUG_DIR.parents[1]
import os
import gc
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, logging as hf_logging

from src.dataset import make_dpo_data_module
from src.params import DataArguments, DPOArguments
from monkey_patch_forward import replace_gemma3_forward
from train.train_dpo import configure_llm, configure_vision_tower
from train.train_sft import configure_dual_input_gate, maybe_restore_dual_input_gate_from_checkpoint
from trl.trainer.utils import flush_left, selective_log_softmax, pad_to_length


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
ATTN = os.environ.get("ATTN", "eager")
DTYPE = torch.bfloat16


def args(use_gate=True):
    return DPOArguments(
        output_dir=str(DEBUG_DIR / "probe_out"),
        use_dual_input_gate=use_gate,
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


def load_model(tag, use_gate=True):
    training_args = args(use_gate=use_gate)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        attn_implementation=ATTN,
    ).cuda()
    configure_llm(model, training_args)
    configure_vision_tower(model, training_args, DTYPE, torch.device("cuda"))
    gate_tok = None
    if use_gate:
        gate_tok = configure_dual_input_gate(model, training_args, DTYPE)
        maybe_restore_dual_input_gate_from_checkpoint(model, MODEL_ID)
        model.siglip_text_model.to(device="cuda", dtype=DTYPE)
        model.gate.to(device="cuda", dtype=DTYPE)
    else:
        model.config.use_dual_input_gate = False
    model.config.use_cache = False
    model.eval()
    print(f"[load] {tag} gate={use_gate} attn={ATTN} mem={torch.cuda.memory_allocated() / 2**30:.2f}GiB", flush=True)
    return model, gate_tok


def build_batch(gate_tok):
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    data_args = DataArguments(data_path=DATA_PATH, image_folder=IMAGE_FOLDER, lazy_preprocess=True)
    module = make_dpo_data_module(
        processor,
        data_args,
        gate_text_tokenizer=gate_tok,
        gate_text_max_length=64,
    )
    batch = module["data_collator"]([module["train_dataset"][0]])
    for k, v in list(batch.items()):
        if torch.is_tensor(v):
            batch[k] = v.cuda()
    return processor, batch


def make_concat(batch, padding_value, include_token_type=True, include_gate=True):
    prompt_input_ids = torch.cat([batch["prompt_input_ids"], batch["prompt_input_ids"]], dim=0)
    prompt_attention_mask = torch.cat([batch["prompt_attention_mask"], batch["prompt_attention_mask"]], dim=0)
    pixel_values = torch.cat([batch["pixel_values"], batch["pixel_values"]], dim=0)
    max_completion_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])
    completion_input_ids = torch.cat(
        [
            pad_to_length(batch["chosen_input_ids"], max_completion_length, pad_value=padding_value),
            pad_to_length(batch["rejected_input_ids"], max_completion_length, pad_value=padding_value),
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
    if include_token_type:
        prompt_token_type_ids = torch.cat([batch["token_type_ids"], batch["token_type_ids"]], dim=0)
        completion_token_type_ids = torch.zeros_like(completion_input_ids)
        token_type_ids = torch.cat([prompt_token_type_ids, completion_token_type_ids], dim=1)
        items.append(token_type_ids)
    flushed = flush_left(*items)
    attention_mask, input_ids, loss_mask = flushed[:3]
    if include_token_type:
        token_type_ids = flushed[3]
    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "pixel_values": pixel_values,
    }
    if include_token_type:
        out["token_type_ids"] = token_type_ids
    if include_gate and "gate_input_ids" in batch:
        out["gate_input_ids"] = torch.cat([batch["gate_input_ids"], batch["gate_input_ids"]], dim=0)
        out["gate_attention_mask"] = torch.cat([batch["gate_attention_mask"], batch["gate_attention_mask"]], dim=0)
    return out


def active_logps(model, name, inputs):
    kwargs = {
        "pixel_values": inputs["pixel_values"],
        "attention_mask": inputs["attention_mask"],
    }
    for key in ("token_type_ids", "gate_input_ids", "gate_attention_mask"):
        if key in inputs:
            kwargs[key] = inputs[key]
    with torch.no_grad(), torch.autocast("cuda", dtype=DTYPE):
        outputs = model(inputs["input_ids"], **kwargs)
    logits = outputs.logits
    labels = torch.roll(inputs["input_ids"], shifts=-1, dims=1).clone()
    loss_mask = torch.roll(inputs["loss_mask"], shifts=-1, dims=1).bool()
    labels[~loss_mask] = 0
    active_flat = loss_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
    active_logits = logits.reshape(-1, logits.shape[-1]).index_select(0, active_flat)
    active_labels = labels.reshape(-1).index_select(0, active_flat)
    selected = active_logits.gather(-1, active_labels[:, None]).squeeze(-1).float()
    lse = active_logits.float().logsumexp(dim=-1)
    logps = selective_log_softmax(logits, labels)
    logps[~loss_mask] = 0
    summed = torch.roll(logps, shifts=1, dims=1).sum(-1).float()
    print(
        f"[case] {name} full_nan={bool(torch.isnan(logits).any().item())} "
        f"active_nan={int(torch.isnan(active_logits).sum().item())} "
        f"selected={selected.tolist()} lse={lse.tolist()} summed={summed.tolist()}",
        flush=True,
    )


def unload(*models):
    for model in models:
        del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[unload] mem={torch.cuda.memory_allocated() / 2**30:.2f}GiB", flush=True)


def run_single(use_gate, include_token_type):
    model, gate_tok = load_model(f"single_gate={use_gate}", use_gate=use_gate)
    processor, batch = build_batch(gate_tok)
    inputs = make_concat(
        batch,
        processor.tokenizer.pad_token_id,
        include_token_type=include_token_type,
        include_gate=use_gate,
    )
    active_logps(model, f"single gate={use_gate} tti={include_token_type}", inputs)
    unload(model)


def run_double(use_gate, include_token_type):
    model_a, gate_tok = load_model("double/a", use_gate=use_gate)
    model_b, _ = load_model("double/b", use_gate=use_gate)
    processor, batch = build_batch(gate_tok)
    inputs = make_concat(
        batch,
        processor.tokenizer.pad_token_id,
        include_token_type=include_token_type,
        include_gate=use_gate,
    )
    active_logps(model_a, f"double/a gate={use_gate} tti={include_token_type}", inputs)
    active_logps(model_b, f"double/b gate={use_gate} tti={include_token_type}", inputs)
    unload(model_a, model_b)


def main():
    print(f"[start] ATTN={ATTN}", flush=True)
    for use_gate in (False, True):
        for include_token_type in (False, True):
            run_single(use_gate, include_token_type)
    for use_gate in (False, True):
        for include_token_type in (False, True):
            run_double(use_gate, include_token_type)


if __name__ == "__main__":
    main()
