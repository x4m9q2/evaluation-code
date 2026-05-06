from pathlib import Path

DEBUG_DIR = Path(__file__).resolve().parent
BUNDLE_ROOT = DEBUG_DIR.parents[1]
import os
import json
import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from src.dataset import make_dpo_data_module
from src.params import DataArguments, DPOArguments
from src.trainer import GemmaDPOTrainer
from monkey_patch_forward import replace_gemma3_forward
from train.train_dpo import configure_llm, configure_vision_tower
from train.train_sft import configure_dual_input_gate, maybe_restore_dual_input_gate_from_checkpoint
from trl.trainer.utils import flush_left, selective_log_softmax, pad_to_length


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
DTYPE = torch.bfloat16


def safe_stats(name, tensor):
    if tensor is None:
        print(f"{name}: None", flush=True)
        return
    with torch.no_grad():
        x = tensor.detach()
        finite = torch.isfinite(x)
        all_finite = bool(finite.all().item())
        nan_count = int(torch.isnan(x).sum().item())
        inf_count = int(torch.isinf(x).sum().item())
        xf = x.float()
        print(
            f"{name}: shape={tuple(x.shape)} dtype={x.dtype} finite={all_finite} "
            f"nan={nan_count} inf={inf_count} min={float(torch.nan_to_num(xf).amin().item())} "
            f"max={float(torch.nan_to_num(xf).amax().item())} mean={float(torch.nan_to_num(xf).mean().item())}",
            flush=True,
        )


def describe_model(tag, model):
    emb = model.get_input_embeddings()
    hooks = getattr(emb, "_forward_hooks", {})
    print(
        f"[model] {tag} training={model.training} use_cache={model.config.use_cache} "
        f"gradient_checkpointing={getattr(model, 'is_gradient_checkpointing', None)} "
        f"emb_hooks={len(hooks)} image_token_index={model.config.image_token_index} "
        f"vocab_size={model.vocab_size} text_vocab={model.config.text_config.vocab_size}",
        flush=True,
    )
    for name in ("language_model.model.embed_tokens.weight", "multi_modal_projector.mm_input_projection_weight"):
        pass


def load_model(tag, gradient_checkpointing=False):
    print(f"\n[load] {tag} model_id={MODEL_ID} attn={ATTN}", flush=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        attn_implementation=ATTN,
    ).cuda()
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
        bf16=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        remove_unused_columns=False,
        napo_loss_type="dyn_lq",
        napo_dyn_q_use_average=True,
        gradient_checkpointing=gradient_checkpointing,
        report_to=[],
        disable_dropout=True,
    )
    configure_llm(model, args)
    configure_vision_tower(model, args, DTYPE, torch.device("cuda"))
    gate_tok = configure_dual_input_gate(model, args, DTYPE)
    maybe_restore_dual_input_gate_from_checkpoint(model, MODEL_ID)
    if getattr(model, "siglip_text_model", None) is not None:
        model.siglip_text_model.to(device="cuda", dtype=DTYPE)
    if getattr(model, "gate", None) is not None:
        model.gate.to(device="cuda", dtype=DTYPE)
    model.config.use_cache = False
    describe_model(tag + "/loaded", model)
    return model, gate_tok, args


def build_batch(processor, gate_tok):
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
            print(f"[batch] {k} {tuple(v.shape)} {v.dtype} min={int(v.min().item()) if v.numel() and v.dtype in (torch.long, torch.int64, torch.int32, torch.bool) else 'na'} max={int(v.max().item()) if v.numel() and v.dtype in (torch.long, torch.int64, torch.int32, torch.bool) else 'na'}", flush=True)
    return module, batch


def make_concat(batch, padding_value):
    prompt_input_ids = torch.cat([batch["prompt_input_ids"], batch["prompt_input_ids"]], dim=0)
    prompt_attention_mask = torch.cat([batch["prompt_attention_mask"], batch["prompt_attention_mask"]], dim=0)
    pixel_values = torch.cat([batch["pixel_values"], batch["pixel_values"]], dim=0)
    gate_input_ids = torch.cat([batch["gate_input_ids"], batch["gate_input_ids"]], dim=0)
    gate_attention_mask = torch.cat([batch["gate_attention_mask"], batch["gate_attention_mask"]], dim=0)
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
    prompt_token_type_ids = torch.cat([batch["token_type_ids"], batch["token_type_ids"]], dim=0)
    completion_token_type_ids = torch.zeros_like(completion_input_ids)
    token_type_ids = torch.cat([prompt_token_type_ids, completion_token_type_ids], dim=1)
    attention_mask, input_ids, loss_mask, token_type_ids = flush_left(
        attention_mask, input_ids, loss_mask, token_type_ids
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "token_type_ids": token_type_ids,
        "pixel_values": pixel_values,
        "gate_input_ids": gate_input_ids,
        "gate_attention_mask": gate_attention_mask,
    }


def forward_logps(model, name, inputs, grad=False):
    describe_model(name, model)
    kwargs = {
        "pixel_values": inputs["pixel_values"],
        "attention_mask": inputs["attention_mask"],
        "token_type_ids": inputs["token_type_ids"],
        "gate_input_ids": inputs["gate_input_ids"],
        "gate_attention_mask": inputs["gate_attention_mask"],
    }
    print(
        f"[input] {name} ids_min={int(inputs['input_ids'].min().item())} ids_max={int(inputs['input_ids'].max().item())} "
        f"image_tokens={int((inputs['input_ids'] == model.config.image_token_index).sum().item())} "
        f"attn_sum={inputs['attention_mask'].sum(-1).tolist()}",
        flush=True,
    )
    with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=DTYPE):
        out = model(inputs["input_ids"], **kwargs)
    logits = out.logits
    safe_stats(name + "/logits", logits)
    labels = torch.roll(inputs["input_ids"], shifts=-1, dims=1).clone()
    loss_mask = torch.roll(inputs["loss_mask"], shifts=-1, dims=1).bool()
    labels[~loss_mask] = 0
    print(
        f"[labels] {name} label_min={int(labels.min().item())} label_max={int(labels.max().item())} "
        f"active={int(loss_mask.sum().item())} vocab={logits.shape[-1]} oob_active={int(((labels >= logits.shape[-1]) & loss_mask).sum().item())}",
        flush=True,
    )
    logps = selective_log_softmax(logits, labels)
    logps[~loss_mask] = 0
    summed = torch.roll(logps, shifts=1, dims=1).sum(-1)
    safe_stats(name + "/sum_logps", summed)
    return summed


def eval_style_case(model, processor, gate_tok, name):
    row = json.load(open(DATA_PATH, "r"))[0]
    image_path = os.path.join(IMAGE_FOLDER, f"COCO_train2014_{int(row['image_id']):012d}.jpg")
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": row["question"]}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[prompt], images=[[image]], return_tensors="pt", padding=True)
    gate = gate_tok(
        [row["question"]],
        add_special_tokens=True,
        truncation=True,
        max_length=64,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    inputs["gate_input_ids"] = gate["input_ids"]
    inputs["gate_attention_mask"] = gate["attention_mask"]
    inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in inputs.items()}
    with torch.no_grad(), torch.autocast("cuda", dtype=DTYPE):
        out = model(**inputs)
    safe_stats(name + "/eval_logits", out.logits)


def run_case(gradient_checkpointing):
    print(f"\n===== gradient_checkpointing={gradient_checkpointing} =====", flush=True)
    policy, gate_tok, args = load_model("policy", gradient_checkpointing=gradient_checkpointing)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    module, batch = build_batch(processor, gate_tok)
    concat = make_concat(batch, processor.tokenizer.pad_token_id)

    policy.train()
    forward_logps(policy, "before_trainer/policy", concat, grad=False)
    eval_style_case(policy, processor, gate_tok, "before_trainer/policy")

    ref, _, _ = load_model("ref", gradient_checkpointing=gradient_checkpointing)
    trainer = GemmaDPOTrainer(
        model=policy,
        ref_model=ref,
        train_dataset=module["train_dataset"],
        eval_dataset=None,
        data_collator=module["data_collator"],
        processing_class=processor,
        args=args,
    )
    print("[trainer] initialized", flush=True)
    describe_model("after_trainer/policy_object", policy)
    describe_model("after_trainer/trainer_model", trainer.model)
    describe_model("after_trainer/ref_model", trainer.ref_model)
    forward_logps(policy, "after_trainer/policy_object", concat, grad=False)
    forward_logps(trainer.model, "after_trainer/trainer_model", concat, grad=False)
    forward_logps(trainer.ref_model, "after_trainer/ref_model", concat, grad=False)
    with torch.no_grad():
        metrics_loss, metrics = trainer.get_batch_loss_metrics(trainer.model, batch, train_eval="train")
    safe_stats("trainer/loss", metrics_loss)
    print(f"[trainer_metrics] {metrics}", flush=True)


def main():
    run_case(gradient_checkpointing=False)
    torch.cuda.empty_cache()
    run_case(gradient_checkpointing=True)


if __name__ == "__main__":
    main()
