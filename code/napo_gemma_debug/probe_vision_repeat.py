from pathlib import Path

DEBUG_DIR = Path(__file__).resolve().parent
BUNDLE_ROOT = DEBUG_DIR.parents[1]
import os
import gc

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, logging as hf_logging

from src.dataset import make_dpo_data_module
from src.params import DataArguments, DPOArguments
from train.train_dpo import configure_llm, configure_vision_tower
from train.train_sft import configure_dual_input_gate, maybe_restore_dual_input_gate_from_checkpoint


hf_logging.set_verbosity_error()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
MODEL_DTYPE = getattr(torch, os.environ.get("MODEL_DTYPE", "bfloat16"))


def stat(name, tensor):
    x = tensor.detach()
    finite = torch.isfinite(x)
    xf = torch.nan_to_num(x.float())
    print(
        f"[stat] {name} shape={tuple(x.shape)} dtype={x.dtype} finite={bool(finite.all().item())} "
        f"nan={int(torch.isnan(x).sum().item())} inf={int(torch.isinf(x).sum().item())} "
        f"min={float(xf.amin().item())} max={float(xf.amax().item())} mean={float(xf.mean().item())}",
        flush=True,
    )


def load_model():
    args = DPOArguments(
        output_dir=str(DEBUG_DIR / "probe_out"),
        use_dual_input_gate=True,
        gate_text_model_id=GATE_TEXT_MODEL_ID,
        freeze_gate_text_encoder=True,
        freeze_llm=False,
        freeze_vision_tower=True,
        freeze_projector=False,
        disable_flash_attn2=True,
        attn_implementation=ATTN,
        bf16=MODEL_DTYPE == torch.bfloat16,
        fp16=MODEL_DTYPE == torch.float16,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        remove_unused_columns=False,
        gradient_checkpointing=False,
        report_to=[],
    )
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=MODEL_DTYPE,
        attn_implementation=ATTN,
    ).cuda()
    configure_llm(model, args)
    configure_vision_tower(model, args, MODEL_DTYPE, torch.device("cuda"))
    gate_tok = configure_dual_input_gate(model, args, MODEL_DTYPE)
    maybe_restore_dual_input_gate_from_checkpoint(model, MODEL_ID)
    model.siglip_text_model.to(device="cuda", dtype=MODEL_DTYPE)
    model.gate.to(device="cuda", dtype=MODEL_DTYPE)
    model.config.use_cache = False
    model.eval()
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
    return batch


def build_n_inputs(gate_tok, n):
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    module = make_dpo_data_module(
        processor,
        DataArguments(data_path=DATA_PATH, image_folder=IMAGE_FOLDER, lazy_preprocess=True),
        gate_text_tokenizer=gate_tok,
        gate_text_max_length=64,
    )
    examples = [module["train_dataset"][idx] for idx in range(min(n, len(module["train_dataset"])))]
    batch = module["data_collator"](examples)
    for k, v in list(batch.items()):
        if torch.is_tensor(v):
            batch[k] = v.cuda()
    return batch


def vision_once(model, pixel_values, name, autocast_dtype=None, force_pixel_dtype=None):
    pv = pixel_values if force_pixel_dtype is None else pixel_values.to(force_pixel_dtype)
    try:
        if autocast_dtype is None:
            out = model.vision_tower(pixel_values=pv).last_hidden_state
        else:
            with torch.autocast("cuda", dtype=autocast_dtype):
                out = model.vision_tower(pixel_values=pv).last_hidden_state
        stat(name, out)
        return out
    except Exception as exc:
        print(f"[error] {name}: {type(exc).__name__}: {exc}", flush=True)
        return None


def text_once(model, batch, name, autocast_dtype=None):
    try:
        if autocast_dtype is None:
            out = model.siglip_text_model(
                input_ids=batch["gate_input_ids"],
                attention_mask=batch["gate_attention_mask"],
                return_dict=True,
            )
        else:
            with torch.autocast("cuda", dtype=autocast_dtype):
                out = model.siglip_text_model(
                    input_ids=batch["gate_input_ids"],
                    attention_mask=batch["gate_attention_mask"],
                    return_dict=True,
                )
        feat = out.pooler_output if out.pooler_output is not None else out.last_hidden_state.mean(dim=1)
        stat(name, feat)
    except Exception as exc:
        print(f"[error] {name}: {type(exc).__name__}: {exc}", flush=True)


def print_module_dtypes(model):
    for name, param in [
        ("vision first", next(model.vision_tower.parameters())),
        ("projector first", next(model.multi_modal_projector.parameters())),
        ("gate fc1", model.gate.fc1.weight),
        ("text first", next(model.siglip_text_model.parameters())),
    ]:
        print(f"[dtype] {name}: {param.dtype}", flush=True)
    print(
        f"[config] model_attn={getattr(model.config, '_attn_implementation', None)} "
        f"vision_attn={getattr(model.config.vision_config, '_attn_implementation', None)} "
        f"vision_class={model.vision_tower.__class__.__name__}",
        flush=True,
    )


def main():
    print(f"[start] ATTN={ATTN} MODEL_DTYPE={MODEL_DTYPE}", flush=True)
    model, gate_tok = load_model()
    print_module_dtypes(model)
    batch = build_inputs(gate_tok)
    batch_two_distinct = build_n_inputs(gate_tok, 2)
    stat("pixel_values", batch["pixel_values"])
    pixel_values_dup = torch.cat([batch["pixel_values"], batch["pixel_values"]], dim=0)
    pixel_values_two_distinct = batch_two_distinct["pixel_values"]
    batch_dup = {
        "gate_input_ids": torch.cat([batch["gate_input_ids"], batch["gate_input_ids"]], dim=0),
        "gate_attention_mask": torch.cat([batch["gate_attention_mask"], batch["gate_attention_mask"]], dim=0),
    }
    stat("pixel_values_dup", pixel_values_dup)
    stat("pixel_values_two_distinct", pixel_values_two_distinct)

    with torch.no_grad():
        print("[case] repeated vision under bf16 autocast", flush=True)
        for i in range(5):
            vision_once(model, batch["pixel_values"], f"vision_autocast_bf16_{i}", autocast_dtype=torch.bfloat16)

        print("[case] text then vision under bf16 autocast", flush=True)
        text_once(model, batch, "text_autocast_bf16_before_vision", autocast_dtype=torch.bfloat16)
        for i in range(3):
            vision_once(model, batch["pixel_values"], f"vision_after_text_autocast_bf16_{i}", autocast_dtype=torch.bfloat16)

        print("[case] explicit bf16 pixels under bf16 autocast", flush=True)
        for i in range(3):
            vision_once(
                model,
                batch["pixel_values"],
                f"vision_pixel_bf16_autocast_bf16_{i}",
                autocast_dtype=torch.bfloat16,
                force_pixel_dtype=torch.bfloat16,
            )

        print("[case] duplicated batch vision under bf16 autocast", flush=True)
        for i in range(5):
            vision_once(model, pixel_values_dup, f"vision_dup_autocast_bf16_{i}", autocast_dtype=torch.bfloat16)

        print("[case] distinct batch size 2 vision under bf16 autocast", flush=True)
        for i in range(3):
            vision_once(model, pixel_values_two_distinct, f"vision_two_distinct_autocast_bf16_{i}", autocast_dtype=torch.bfloat16)

        print("[case] duplicated batch text then vision under bf16 autocast", flush=True)
        text_once(model, batch_dup, "text_dup_autocast_bf16_before_vision", autocast_dtype=torch.bfloat16)
        for i in range(5):
            vision_once(model, pixel_values_dup, f"vision_dup_after_text_autocast_bf16_{i}", autocast_dtype=torch.bfloat16)

        print("[case] no autocast with current module dtype", flush=True)
        for i in range(2):
            vision_once(model, batch["pixel_values"], f"vision_no_autocast_{i}")
        for i in range(2):
            vision_once(model, pixel_values_dup, f"vision_dup_no_autocast_{i}")
        for i in range(2):
            vision_once(model, pixel_values_two_distinct, f"vision_two_distinct_no_autocast_{i}")

        print("[case] vision tower float32, no autocast", flush=True)
        model.vision_tower.float()
        gc.collect()
        torch.cuda.empty_cache()
        for i in range(3):
            vision_once(model, batch["pixel_values"], f"vision_float32_no_autocast_{i}")
        for i in range(3):
            vision_once(model, pixel_values_dup, f"vision_dup_float32_no_autocast_{i}")
        for i in range(3):
            vision_once(model, pixel_values_two_distinct, f"vision_two_distinct_float32_no_autocast_{i}")

        print("[case] vision tower float32 under bf16 autocast", flush=True)
        for i in range(2):
            vision_once(model, pixel_values_dup, f"vision_dup_float32_autocast_bf16_{i}", autocast_dtype=torch.bfloat16)


if __name__ == "__main__":
    main()
