from pathlib import Path

DEBUG_DIR = Path(__file__).resolve().parent
BUNDLE_ROOT = DEBUG_DIR.parents[1]
import os

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


def tensor_finite_line(name, tensor):
    if tensor is None:
        return f"{name}=None"
    x = tensor.detach()
    return (
        f"{name}: finite={bool(torch.isfinite(x).all().item())} "
        f"nan={int(torch.isnan(x).sum().item())} inf={int(torch.isinf(x).sum().item())} "
        f"shape={tuple(x.shape)} dtype={x.dtype}"
    )


def grad_report(model, limit=20):
    total = 0
    with_grad = 0
    bad = []
    none = []
    finite_norm_sq = torch.zeros((), device="cuda", dtype=torch.float32)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        total += 1
        if param.grad is None:
            none.append(name)
            continue
        with_grad += 1
        grad = param.grad.detach()
        finite = torch.isfinite(grad)
        if not bool(finite.all().item()):
            bad.append(
                (
                    name,
                    tuple(grad.shape),
                    str(grad.dtype),
                    int(torch.isnan(grad).sum().item()),
                    int(torch.isinf(grad).sum().item()),
                    float(torch.nan_to_num(grad.float(), nan=0.0, posinf=0.0, neginf=0.0).abs().max().item()),
                )
            )
        else:
            finite_norm_sq += grad.float().pow(2).sum()

    print(f"[grads] trainable_params={total} with_grad={with_grad} none={len(none)} bad={len(bad)}", flush=True)
    print(f"[grads] finite_only_l2={float(finite_norm_sq.sqrt().item())}", flush=True)
    for item in bad[:limit]:
        name, shape, dtype, nan_count, inf_count, finite_max_abs = item
        print(
            f"[bad_grad] {name} shape={shape} dtype={dtype} "
            f"nan={nan_count} inf={inf_count} finite_max_abs={finite_max_abs}",
            flush=True,
        )
    if none:
        print(f"[grads] first_none={none[:limit]}", flush=True)


def load_model_and_args(gradient_checkpointing: bool):
    args = DPOArguments(
        output_dir=str(DEBUG_DIR / "probe_backward_out"),
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
        gradient_checkpointing=gradient_checkpointing,
        report_to=[],
        logging_steps=1,
        max_steps=1,
        save_strategy="no",
    )

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
    return model, args, gate_tok


def run_case(gradient_checkpointing: bool):
    print(f"\n===== gradient_checkpointing={gradient_checkpointing} =====", flush=True)
    torch.cuda.empty_cache()
    replace_gemma3_forward(use_liger=False)
    model, args, gate_tok = load_model_and_args(gradient_checkpointing)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    data_module = make_dpo_data_module(
        processor=processor,
        data_args=DataArguments(data_path=DATA_PATH, image_folder=IMAGE_FOLDER, lazy_preprocess=True),
        gate_text_tokenizer=gate_tok,
        gate_text_max_length=args.gate_text_max_length,
    )
    trainer = GemmaDPOTrainer(
        model=model,
        ref_model=None,
        train_dataset=data_module["train_dataset"],
        eval_dataset=None,
        data_collator=data_module["data_collator"],
        processing_class=processor,
        args=args,
    )

    batch = data_module["data_collator"]([data_module["train_dataset"][0]])
    batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
    trainer.model.train()
    loss, metrics = trainer.get_batch_loss_metrics(trainer.model, batch, train_eval="train")
    print(f"[loss] {tensor_finite_line('loss', loss)} value={float(loss.detach().float().item())}", flush=True)
    print(f"[metrics] {metrics}", flush=True)
    trainer.model.zero_grad(set_to_none=True)
    loss.backward()
    grad_report(trainer.model)
    del trainer, model
    torch.cuda.empty_cache()


def main():
    run_case(False)
    run_case(True)


if __name__ == "__main__":
    main()
