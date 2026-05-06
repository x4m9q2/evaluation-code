from pathlib import Path

DEBUG_DIR = Path(__file__).resolve().parent
BUNDLE_ROOT = DEBUG_DIR.parents[1]
import json
import os
from contextlib import nullcontext

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, logging as hf_logging

from src.dataset import make_dpo_data_module
from src.params import DataArguments, DPOArguments
from src.trainer import GemmaDPOTrainer
from train.monkey_patch_forward import replace_gemma3_forward
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
DTYPE = torch.bfloat16
LIMIT = int(os.environ.get("LIMIT", "64"))
START = int(os.environ.get("START", "0"))
USE_REF = os.environ.get("USE_REF", "0") == "1"
TRAIN_MODE = os.environ.get("TRAIN_MODE", "1") == "1"


def finite_summary(output):
    keys = ("chosen_logps", "rejected_logps", "chosen_avg_logps", "rejected_avg_logps", "mean_chosen_logits", "mean_rejected_logits")
    parts = {}
    for key in keys:
        value = output[key].detach()
        parts[key] = {
            "finite": bool(torch.isfinite(value).all().item()),
            "nan": int(torch.isnan(value).sum().item()),
            "inf": int(torch.isinf(value).sum().item()),
            "value": float(torch.nan_to_num(value.float(), nan=0.0, posinf=0.0, neginf=0.0).mean().item()),
        }
    return parts


def has_bad(summary):
    return any(not item["finite"] for item in summary.values())


def print_row(tag, idx, row, summary):
    bad_keys = [key for key, value in summary.items() if not value["finite"]]
    print(
        f"[bad][{tag}] idx={idx} qid={row.get('question_id')} image_id={row.get('image_id')} "
        f"answer_type={row.get('answer_type')} bad_keys={bad_keys} "
        f"answer={row.get('answer')!r} shortcut={row.get('shortcut_answer')!r} "
        f"question={row.get('question')!r}",
        flush=True,
    )
    print(f"[bad-summary][{tag}] {summary}", flush=True)


def load_model(args):
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        attn_implementation=ATTN,
    ).cuda()
    configure_llm(model, args)
    configure_vision_tower(model, args, DTYPE, torch.device("cuda"))
    gate_tok = configure_dual_input_gate(model, args, DTYPE)
    maybe_restore_dual_input_gate_from_checkpoint(model, MODEL_ID)
    if getattr(model, "siglip_text_model", None) is not None:
        model.siglip_text_model.to(device="cuda", dtype=DTYPE)
    if getattr(model, "gate", None) is not None:
        model.gate.to(device="cuda", dtype=DTYPE)
    model.config.use_cache = False
    if TRAIN_MODE:
        model.train()
    else:
        model.eval()
    return model, gate_tok


def main():
    print(
        f"[config] model={MODEL_ID} data={DATA_PATH} start={START} limit={LIMIT} "
        f"attn={ATTN} vision_attn={VISION_ATTN} use_ref={USE_REF} train_mode={TRAIN_MODE}",
        flush=True,
    )
    replace_gemma3_forward(use_liger=False)
    args = DPOArguments(
        output_dir=f"{REPO_ROOT}/debug/napo_smoke_20260501/probe_scan_out",
        use_dual_input_gate=True,
        gate_text_model_id=GATE_TEXT_MODEL_ID,
        freeze_gate_text_encoder=True,
        freeze_llm=False,
        freeze_vision_tower=True,
        freeze_projector=False,
        disable_flash_attn2=True,
        attn_implementation=ATTN,
        vision_attn_implementation=VISION_ATTN,
        bf16=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        remove_unused_columns=False,
        napo_loss_type="dyn_lq",
        napo_alpha=0.5,
        napo_dyn_q_use_average=True,
        disable_token_type_ids=True,
        disable_ref_model=True,
        gradient_checkpointing=False,
        report_to=[],
        logging_steps=1,
        max_steps=1,
        save_strategy="no",
    )

    model, gate_tok = load_model(args)
    ref_model = None
    if USE_REF:
        ref_model, _ = load_model(args)
        ref_model.eval()

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    data_module = make_dpo_data_module(
        processor=processor,
        data_args=DataArguments(data_path=DATA_PATH, image_folder=IMAGE_FOLDER, lazy_preprocess=True),
        gate_text_tokenizer=gate_tok,
        gate_text_max_length=args.gate_text_max_length,
    )
    trainer = GemmaDPOTrainer(
        model=model,
        ref_model=ref_model,
        train_dataset=data_module["train_dataset"],
        eval_dataset=None,
        data_collator=data_module["data_collator"],
        processing_class=processor,
        args=args,
    )

    rows = json.load(open(DATA_PATH, "r"))
    bad_policy = []
    bad_ref = []
    end = min(len(data_module["train_dataset"]), START + LIMIT)
    ctx = nullcontext()
    with torch.no_grad(), torch.autocast("cuda", dtype=DTYPE), ctx:
        for idx in range(START, end):
            batch = data_module["data_collator"]([data_module["train_dataset"][idx]])
            batch = {key: (value.cuda() if torch.is_tensor(value) else value) for key, value in batch.items()}

            policy_output = trainer.concatenated_forward(trainer.model, batch)
            policy_summary = finite_summary(policy_output)
            if has_bad(policy_summary):
                bad_policy.append(idx)
                print_row("policy", idx, rows[idx], policy_summary)

            if ref_model is not None:
                ref_output = trainer.concatenated_forward(trainer.ref_model, batch)
                ref_summary = finite_summary(ref_output)
                if has_bad(ref_summary):
                    bad_ref.append(idx)
                    print_row("ref", idx, rows[idx], ref_summary)

            if (idx + 1 - START) % 10 == 0:
                print(f"[progress] checked={idx + 1 - START} bad_policy={len(bad_policy)} bad_ref={len(bad_ref)}", flush=True)

    print(f"[done] checked={end - START} bad_policy={bad_policy} bad_ref={bad_ref}", flush=True)


if __name__ == "__main__":
    main()
