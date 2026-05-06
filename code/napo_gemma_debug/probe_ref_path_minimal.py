from pathlib import Path

DEBUG_DIR = Path(__file__).resolve().parent
BUNDLE_ROOT = DEBUG_DIR.parents[1]
import os
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, logging as hf_logging

from src.dataset import make_dpo_data_module
from src.params import DataArguments, DPOArguments
from src.trainer import GemmaDPOTrainer
from monkey_patch_forward import replace_gemma3_forward, _get_gated_projected_image_features
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
ATTN = os.environ.get("ATTN", "sdpa")
VISION_ATTN = os.environ.get("VISION_ATTN", os.environ.get("FORCE_VISION_ATTN", ""))
DTYPE = torch.bfloat16
GC = os.environ.get("GC", "1") == "1"


def make_args():
    return DPOArguments(
        output_dir=str(DEBUG_DIR / "probe_out"),
        use_dual_input_gate=True,
        gate_text_model_id=GATE_TEXT_MODEL_ID,
        freeze_gate_text_encoder=True,
        freeze_llm=False,
        freeze_vision_tower=True,
        freeze_projector=False,
        disable_flash_attn2=True,
        attn_implementation=ATTN,
        vision_attn_implementation=VISION_ATTN or None,
        bf16=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        remove_unused_columns=False,
        napo_loss_type="dyn_lq",
        napo_dyn_q_use_average=True,
        gradient_checkpointing=GC,
        report_to=[],
        disable_dropout=True,
    )


def load_model(tag, args):
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
    describe(tag + "/loaded", model)
    return model, gate_tok


def describe(tag, model):
    emb = model.get_input_embeddings()
    hooks = getattr(emb, "_forward_hooks", {})
    first = next(model.parameters())
    print(
        f"[model] {tag} type={type(model).__name__} training={model.training} "
        f"param_dtype={first.dtype} param_device={first.device} "
        f"gc={getattr(model, 'is_gradient_checkpointing', None)} emb_hooks={len(hooks)} "
        f"use_cache={model.config.use_cache}",
        flush=True,
    )


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


def active_logps(model, name, inputs, autocast_enabled, use_cache=None):
    ctx = torch.autocast("cuda", dtype=DTYPE) if autocast_enabled else torch.amp.autocast("cuda", enabled=False)
    with torch.no_grad(), ctx:
        kwargs = {}
        if use_cache is not None:
            kwargs["use_cache"] = use_cache
        outputs = model(
            inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            gate_input_ids=inputs["gate_input_ids"],
            gate_attention_mask=inputs["gate_attention_mask"],
            **kwargs,
        )
    logits = outputs.logits
    labels = torch.roll(inputs["input_ids"], shifts=-1, dims=1).clone()
    loss_mask = torch.roll(inputs["loss_mask"], shifts=-1, dims=1).bool()
    labels[~loss_mask] = 0
    flat_logits = logits.reshape(-1, logits.shape[-1])
    active_flat = loss_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
    active_logits = flat_logits.index_select(0, active_flat)
    active_labels = labels.reshape(-1).index_select(0, active_flat)
    selected = active_logits.gather(-1, active_labels[:, None]).squeeze(-1)
    logsumexp = active_logits.float().logsumexp(dim=-1)
    logps_manual = selected.float() - logsumexp
    logps_selective = selective_log_softmax(logits, labels)
    logps_selective[~loss_mask] = 0
    summed = torch.roll(logps_selective, shifts=1, dims=1).sum(-1)
    print(
        f"[forward] {name} autocast={autocast_enabled} use_cache={use_cache} logits_dtype={logits.dtype} "
        f"full_any_nan={bool(torch.isnan(logits).any().item())} full_any_inf={bool(torch.isinf(logits).any().item())} "
        f"active_all_finite={bool(torch.isfinite(active_logits).all().item())} "
        f"active_nan={int(torch.isnan(active_logits).sum().item())} active_inf={int(torch.isinf(active_logits).sum().item())} "
        f"selected={selected.float().tolist()} logsumexp={logsumexp.tolist()} "
        f"manual_logps={logps_manual.tolist()} summed={summed.float().tolist()}",
        flush=True,
    )
    return summed


def manual_forward_logps(model, name, inputs, output_hidden_states=False, return_dict=True, use_cache=False):
    with torch.no_grad(), torch.autocast("cuda", dtype=DTYPE):
        llm_ids = inputs["input_ids"].clone()
        llm_ids[llm_ids == model.config.image_token_index] = 0
        inputs_embeds = model.get_input_embeddings()(llm_ids)
        image_features, _ = _get_gated_projected_image_features(
            model,
            inputs["pixel_values"],
            gate_input_ids=inputs["gate_input_ids"],
            gate_attention_mask=inputs["gate_attention_mask"],
        )
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        special_image_mask = (inputs["input_ids"] == model.config.image_token_index).unsqueeze(-1)
        special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
        cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
        causal_mask = model._update_causal_mask(
            inputs["attention_mask"],
            inputs["token_type_ids"],
            None,
            cache_position,
            inputs_embeds,
            False,
        )
        outputs = model.language_model(
            attention_mask=causal_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=False,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=0,
        )
    logits = outputs.logits if return_dict else outputs[0]
    labels = torch.roll(inputs["input_ids"], shifts=-1, dims=1).clone()
    loss_mask = torch.roll(inputs["loss_mask"], shifts=-1, dims=1).bool()
    labels[~loss_mask] = 0
    active_flat = loss_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
    active_logits = logits.reshape(-1, logits.shape[-1]).index_select(0, active_flat)
    active_labels = labels.reshape(-1).index_select(0, active_flat)
    selected = active_logits.gather(-1, active_labels[:, None]).squeeze(-1).float()
    logsumexp = active_logits.float().logsumexp(dim=-1)
    print(
        f"[manual] {name} output_hidden_states={output_hidden_states} return_dict={return_dict} "
        f"use_cache={use_cache} logits_dtype={logits.dtype} full_any_nan={bool(torch.isnan(logits).any().item())} "
        f"active_all_finite={bool(torch.isfinite(active_logits).all().item())} "
        f"active_nan={int(torch.isnan(active_logits).sum().item())} selected={selected.tolist()} "
        f"logsumexp={logsumexp.tolist()}",
        flush=True,
    )


def main():
    print(f"[case] GC={GC} ATTN={ATTN}", flush=True)
    args = make_args()
    policy, gate_tok = load_model("policy", args)
    ref, _ = load_model("ref", args)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    module, batch = build_batch(processor, gate_tok)
    concat = make_concat(batch, processor.tokenizer.pad_token_id)

    policy.train()
    ref.eval()
    describe("ref/before_trainer", ref)
    manual_forward_logps(ref, "ref/before_trainer", concat, output_hidden_states=True, return_dict=True, use_cache=False)
    manual_forward_logps(ref, "ref/before_trainer", concat, output_hidden_states=False, return_dict=True, use_cache=False)
    manual_forward_logps(ref, "ref/before_trainer", concat, output_hidden_states=False, return_dict=False, use_cache=False)
    active_logps(ref, "ref/before_trainer", concat, autocast_enabled=True)
    active_logps(ref, "ref/before_trainer", concat, autocast_enabled=True, use_cache=False)
    active_logps(ref, "ref/before_trainer", concat, autocast_enabled=True, use_cache=True)
    active_logps(ref, "ref/before_trainer", concat, autocast_enabled=False, use_cache=False)

    trainer = GemmaDPOTrainer(
        model=policy,
        ref_model=ref,
        train_dataset=module["train_dataset"],
        eval_dataset=None,
        data_collator=module["data_collator"],
        processing_class=processor,
        args=args,
    )
    print(f"[trainer] accelerator_mp={trainer.accelerator.mixed_precision} ref_is_same={trainer.ref_model is ref}", flush=True)
    describe("ref/after_trainer_original", ref)
    describe("ref/after_trainer_trainer_ref", trainer.ref_model)
    active_logps(ref, "ref/original_after_trainer", concat, autocast_enabled=True)
    active_logps(ref, "ref/original_after_trainer", concat, autocast_enabled=False)
    active_logps(trainer.ref_model, "ref/trainer_ref_after_trainer", concat, autocast_enabled=True)
    active_logps(trainer.ref_model, "ref/trainer_ref_after_trainer", concat, autocast_enabled=False)

    with torch.no_grad():
        loss, metrics = trainer.get_batch_loss_metrics(trainer.model, batch, train_eval="train")
    print(f"[trainer_loss] finite={bool(torch.isfinite(loss).item())} loss={float(loss.detach().float().item())} metrics={metrics}", flush=True)


if __name__ == "__main__":
    main()
