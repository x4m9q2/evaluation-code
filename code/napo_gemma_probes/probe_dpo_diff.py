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


def stats(name, tensor):
    if tensor is None:
        print(f"{name}: None", flush=True)
        return
    x = tensor.detach().float()
    finite = torch.isfinite(x)
    finite_vals = x[finite]
    print(
        f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"finite={bool(finite.all().item())} nan={int(torch.isnan(x).sum().item())} "
        f"inf={int(torch.isinf(x).sum().item())} "
        f"min={float(finite_vals.min().item()) if finite_vals.numel() else None} "
        f"max={float(finite_vals.max().item()) if finite_vals.numel() else None} "
        f"mean={float(finite_vals.mean().item()) if finite_vals.numel() else None}",
        flush=True,
    )


def load_model(tag):
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
        report_to=[],
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
    print(f"[load] {tag} training={model.training} dtype={next(model.parameters()).dtype}", flush=True)
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
            print(f"[batch] {k} {tuple(v.shape)} {v.dtype}", flush=True)
    return module, batch


def make_concat(batch, padding_value, do_flush=True, include_token_type=True):
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
    token_type_ids = None
    if include_token_type:
        prompt_token_type_ids = torch.cat([batch["token_type_ids"], batch["token_type_ids"]], dim=0)
        completion_token_type_ids = torch.zeros_like(completion_input_ids)
        token_type_ids = torch.cat([prompt_token_type_ids, completion_token_type_ids], dim=1)
    if do_flush:
        if token_type_ids is None:
            attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
        else:
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


def forward_case(model, name, inputs, use_autocast=True, grad=False):
    kwargs = {
        "pixel_values": inputs["pixel_values"],
        "attention_mask": inputs["attention_mask"],
        "gate_input_ids": inputs["gate_input_ids"],
        "gate_attention_mask": inputs["gate_attention_mask"],
    }
    if inputs["token_type_ids"] is not None:
        kwargs["token_type_ids"] = inputs["token_type_ids"]
    ctx = torch.autocast("cuda", dtype=DTYPE) if use_autocast else torch.cuda.amp.autocast(enabled=False)
    with torch.set_grad_enabled(grad), ctx:
        out = model(inputs["input_ids"], **kwargs)
    logits = out.logits
    stats(name + "/logits", logits)
    labels = torch.roll(inputs["input_ids"], shifts=-1, dims=1)
    loss_mask = torch.roll(inputs["loss_mask"], shifts=-1, dims=1).bool()
    labels = labels.clone()
    labels[~loss_mask] = 0
    try:
        logps = selective_log_softmax(logits, labels)
        logps[~loss_mask] = 0
        summed = torch.roll(logps, shifts=1, dims=1).sum(-1)
        stats(name + "/sum_logps", summed)
    except Exception as exc:
        print(f"{name}/logps_exception: {type(exc).__name__}: {exc}", flush=True)


def eval_style_case(model, processor, gate_tok, name, use_autocast=True):
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
    ctx = torch.autocast("cuda", dtype=DTYPE) if use_autocast else torch.cuda.amp.autocast(enabled=False)
    with torch.no_grad(), ctx:
        out = model(**inputs)
    stats(name + "/logits", out.logits)


def main():
    policy, gate_tok, args = load_model("policy")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    module, batch = build_batch(processor, gate_tok)
    padding_value = processor.tokenizer.pad_token_id
    print(f"[padding] {padding_value}", flush=True)

    policy.train()
    for do_flush in (False, True):
        for include_tti in (False, True):
            concat = make_concat(batch, padding_value, do_flush=do_flush, include_token_type=include_tti)
            print(
                f"\n[case-input] before_trainer flush={do_flush} token_type={include_tti} "
                f"shape={tuple(concat['input_ids'].shape)} image_tokens={int((concat['input_ids'] == policy.config.image_token_index).sum().item())} "
                f"attn_sum={concat['attention_mask'].sum(-1).tolist()}",
                flush=True,
            )
            forward_case(policy, f"before_trainer/flush={do_flush}/tti={include_tti}/autocast", concat, use_autocast=True)
            forward_case(policy, f"before_trainer/flush={do_flush}/tti={include_tti}/no_autocast", concat, use_autocast=False)
    eval_style_case(policy, processor, gate_tok, "before_trainer/eval_style/autocast", use_autocast=True)

    ref, _, _ = load_model("ref")
    trainer = GemmaDPOTrainer(
        model=policy,
        ref_model=ref,
        train_dataset=module["train_dataset"],
        eval_dataset=None,
        data_collator=module["data_collator"],
        processing_class=processor,
        args=args,
    )
    print(f"\n[trainer] policy.training={policy.training} trainer.model.training={trainer.model.training}", flush=True)
    print(f"[trainer] ref type={type(trainer.ref_model)} training={getattr(trainer.ref_model, 'training', None)}", flush=True)

    concat = make_concat(batch, padding_value, do_flush=True, include_token_type=True)
    forward_case(policy, "after_trainer/policy_object", concat, use_autocast=True)
    forward_case(trainer.model, "after_trainer/trainer_model", concat, use_autocast=True)
    forward_case(trainer.ref_model, "after_trainer/ref_model", concat, use_autocast=True)
    eval_style_case(policy, processor, gate_tok, "after_trainer/eval_style/autocast", use_autocast=True)


if __name__ == "__main__":
    main()
