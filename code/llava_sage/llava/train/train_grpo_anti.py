import pathlib
import copy
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from transformers import CLIPTokenizer

from llava import conversation as conversation_lib
from llava.model import LlavaLlamaForCausalLM, LlavaMptForCausalLM
from llava.train.grpo_dataset_anti import make_grpo_anti_data_module
from llava.train.grpo_trainer_anti import LLaVAAntiShortcutGRPOTrainer
from llava.train.train import (
    ModelArguments,
    find_all_linear_names,
    rank0_print,
    safe_save_model_for_hf_trainer,
    smart_tokenizer_and_embedding_resize,
)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to pair-level anti-shortcut GRPO data."})
    eval_data_path: Optional[str] = field(default=None, metadata={"help": "Optional path to pair-level anti-shortcut GRPO validation data."})
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = field(default="square")
    clip_tokenizer_path: str = field(default="./clip-vit-large-patch14-336")
    lazy_preprocess: bool = field(default=True)
    is_multimodal: bool = field(default=False)


@dataclass
class GRPOTrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(default=512)
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    bits: int = field(default=16)
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_weight_path: str = field(default="")
    lora_bias: str = field(default="none")
    mm_projector_lr: Optional[float] = field(default=None)
    group_by_length: bool = field(default=False)
    tune_language_model: bool = field(default=False)
    tune_vision_tower: bool = field(default=False)
    tune_mm_projector: bool = field(default=True)
    tune_gate: bool = field(default=True)
    tune_lm_head: bool = field(default=False)
    grpo_max_new_tokens: int = field(default=16)
    grpo_min_new_tokens: int = field(default=1)
    grpo_do_sample: bool = field(default=True)
    grpo_temperature: float = field(default=0.7)
    grpo_top_p: float = field(default=0.9)
    grpo_reward_eps: float = field(default=1e-6)
    grpo_reward_match_as: float = field(default=1.0)
    grpo_reward_other: float = field(default=0.2)
    grpo_reward_shortcut: float = field(default=-1.0)

    grpo_empty_penalty: float = field(default=1.0)
    grpo_length_penalty: float = field(default=0.5)
    grpo_repeat_penalty: float = field(default=0.5)
    grpo_max_answer_words: int = field(default=3)

    grpo_group_size: int = field(default=4)
    grpo_update_epochs: int = field(default=1)
    grpo_adaptive_group_reuse: bool = field(default=False)
    grpo_low_var_threshold: float = field(default=0.0)
    grpo_high_var_threshold: float = field(default=0.0)
    grpo_clip_epsilon: float = field(default=0.2)
    grpo_kl_coef: float = field(default=0.01)
    grpo_eval_max_groups: int = field(default=0)


def _set_trainable(module, enabled: bool):
    for param in module.parameters():
        param.requires_grad = enabled


def configure_trainable_components(model, training_args: GRPOTrainingArguments):
    model.requires_grad_(False)
    base_model = model.get_model() if hasattr(model, "get_model") else model

    if training_args.tune_language_model:
        for name, param in base_model.named_parameters():
            if any(keyword in name for keyword in ["vision_tower", "mm_projector", "gate"]):
                continue
            param.requires_grad = True

    if training_args.tune_vision_tower and hasattr(base_model, "vision_tower") and base_model.vision_tower is not None:
        _set_trainable(base_model.vision_tower, True)

    if training_args.tune_mm_projector and hasattr(base_model, "mm_projector") and base_model.mm_projector is not None:
        _set_trainable(base_model.mm_projector, True)

    if training_args.tune_gate and hasattr(base_model, "gate") and base_model.gate is not None:
        _set_trainable(base_model.gate, True)

    if training_args.tune_lm_head and hasattr(model, "lm_head") and model.lm_head is not None:
        _set_trainable(model.lm_head, True)


def log_trainable_parameters(model):
    trainable_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        numel = param.numel()
        total_params += numel
        if param.requires_grad:
            trainable_params += numel
            rank0_print(f"[GRPO trainable] {name} | shape={list(param.shape)} | dtype={param.dtype}")
    trainable_ratio = 100.0 * trainable_params / max(total_params, 1)
    rank0_print(f"[GRPO trainable] params={trainable_params} / {total_params} ({trainable_ratio:.4f}%)")


def build_model_and_tokenizer(model_args, data_args, training_args, attn_implementation=None):
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["mm_projector"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,
                ),
            )
        )

    if model_args.vision_tower is not None:
        if "mpt" in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config["attn_impl"] = training_args.mpt_attn_impl
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args,
            )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args,
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args,
        )

    model.config.use_cache = False

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training

        model.config.torch_dtype = compute_dtype
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if "mpt" in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True
        data_args.mm_use_im_start_end = model_args.mm_use_im_start_end

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        model.config.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    configure_trainable_components(model, training_args)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters for GRPO...")
        model = get_peft_model(model, lora_config)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer) and training_args.bf16:
                module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if ("lm_head" in name or "embed_tokens" in name) and hasattr(module, "weight"):
                if training_args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    log_trainable_parameters(model)
    return model, tokenizer


def train(attn_implementation=None):
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, GRPOTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model, tokenizer = build_model_and_tokenizer(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        attn_implementation=attn_implementation,
    )

    ref_model = copy.deepcopy(model)

    ref_dtype = None
    if training_args.bf16:
        ref_dtype = torch.bfloat16
    elif training_args.fp16:
        ref_dtype = torch.float16

    if ref_dtype is not None:
        ref_model.to(device=training_args.device, dtype=ref_dtype)
    else:
        ref_model.to(device=training_args.device)

    ref_model.requires_grad_(False)
    ref_model.eval()

    clip_tokenizer = CLIPTokenizer.from_pretrained(data_args.clip_tokenizer_path)
    data_module = make_grpo_anti_data_module(
        clip_tokenizer=clip_tokenizer,
        tokenizer=tokenizer,
        data_args=data_args,
    )

    trainer = LLaVAAntiShortcutGRPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
