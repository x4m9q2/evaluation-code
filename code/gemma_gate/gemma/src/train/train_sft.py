import os
import torch
import transformers
from peft import LoraConfig, get_peft_model
import ast
from transformers import AutoModel, AutoProcessor, AutoTokenizer, BitsAndBytesConfig, Gemma3ForConditionalGeneration
from transformers.modeling_utils import load_sharded_checkpoint
from safetensors import safe_open
from src.trainer import GemmaSFTTrainer
from src.dataset import make_supervised_data_module
from src.params import DataArguments, ModelArguments, TrainingArguments
from train.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer
import pathlib
from monkey_patch_forward import replace_gemma3_forward
from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_gemma3_text
try:
    from src.gate_model.build_gate_model import DualInputGate
except ImportError:
    # Backward compatibility when running inside the original LLaVA workspace.
    from llava.model.gate_model.build_gate_model import DualInputGate

local_rank = None
VALID_ATTN_IMPLEMENTATIONS = {"auto", "flash_attention_2", "sdpa", "eager"}

def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)


def resolve_attn_implementation(training_args):
    attn_implementation = training_args.attn_implementation
    if attn_implementation not in VALID_ATTN_IMPLEMENTATIONS:
        raise ValueError(
            f"Unsupported attn_implementation={attn_implementation!r}. "
            f"Expected one of {sorted(VALID_ATTN_IMPLEMENTATIONS)}."
        )
    if attn_implementation == "auto":
        return "eager" if training_args.disable_flash_attn2 else "flash_attention_2"
    return attn_implementation

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names

def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad

def configure_vision_tower(model, training_args, compute_dtype, device):
    vision_tower = model.vision_tower
    vision_tower.to(dtype=compute_dtype, device=device)
    vision_attn_implementation = getattr(training_args, "vision_attn_implementation", None)
    if vision_attn_implementation:
        model.config.vision_config._attn_implementation = vision_attn_implementation
        vision_tower.config._attn_implementation = vision_attn_implementation
        rank0_print(f"Using vision_attn_implementation={vision_attn_implementation}")

    img_projection_params = model.multi_modal_projector.parameters()
    set_requires_grad(img_projection_params, not training_args.freeze_projector)

    vision_model_params = vision_tower.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)

    if training_args.bits in [4, 8]:
        model.model.vision_embed_tokens.img_processor.to(dtype=compute_dtype, device=device)

def configure_llm(model, training_args):
    llm_params = model.language_model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)


def configure_dual_input_gate(model, training_args, compute_dtype):
    if not training_args.use_dual_input_gate:
        model.config.use_dual_input_gate = False
        return None

    siglip_model = AutoModel.from_pretrained(
        training_args.gate_text_model_id,
        torch_dtype=compute_dtype,
        trust_remote_code=False,
    )
    text_model = siglip_model.text_model
    text_hidden_size = text_model.config.hidden_size
    gate = DualInputGate(model.config.vision_config.hidden_size, text_hidden_size)

    model.siglip_text_model = text_model
    model.gate = gate
    model.config.use_dual_input_gate = True

    set_requires_grad(model.gate.parameters(), True)
    set_requires_grad(model.siglip_text_model.parameters(), not training_args.freeze_gate_text_encoder)
    del siglip_model

    return AutoTokenizer.from_pretrained(training_args.gate_text_model_id)


def iter_checkpoint_safetensors(model_path):
    seen = set()
    model_path = pathlib.Path(model_path)
    for pattern in ("model-*.safetensors", "model.safetensors", "*.safetensors"):
        for path in sorted(model_path.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            yield path


def checkpoint_has_dual_input_gate_weights(model_path):
    model_path = pathlib.Path(model_path)
    if not model_path.exists() or not model_path.is_dir():
        return False

    for shard_path in iter_checkpoint_safetensors(model_path):
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("gate.") or key.startswith("siglip_text_model."):
                    return True
    return False


def get_state_tensor_device(model, name):
    obj = model
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


def resolve_checkpoint_tensor(model_path, tensor_name):
    for shard_path in iter_checkpoint_safetensors(model_path):
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            if tensor_name in f.keys():
                return f.get_tensor(tensor_name)
    raise KeyError(f"Tensor not found in checkpoint shards: {tensor_name}")


def validate_dual_input_gate_checkpoint_loaded(model, model_path):
    check_names = [
        "gate.fc1.weight",
        "gate.fc2.bias",
        "siglip_text_model.embeddings.token_embedding.weight",
        "siglip_text_model.final_layer_norm.weight",
    ]
    max_diff = 0.0
    for name in check_names:
        model_tensor = get_state_tensor_device(model, name).detach()
        ckpt_tensor = resolve_checkpoint_tensor(model_path, name).to(
            device=model_tensor.device,
            dtype=model_tensor.dtype,
        )
        diff = float((model_tensor - ckpt_tensor).abs().max().item())
        max_diff = max(max_diff, diff)
        rank0_print(f"[gate-load-check] {name} max_abs_diff={diff:.6g}")
    rank0_print(f"[gate-load-check] overall_max_abs_diff={max_diff:.6g}")
    if max_diff != 0.0:
        raise RuntimeError(f"Gate/SigLIP checkpoint validation failed: max_abs_diff={max_diff}")


def maybe_restore_dual_input_gate_from_checkpoint(model, model_id):
    model_path = pathlib.Path(model_id)
    if not checkpoint_has_dual_input_gate_weights(model_path):
        rank0_print(f"[gate-load] no gate weights found in {model_id}; using freshly attached gate/SigLIP text encoder")
        return

    rank0_print(f"[gate-load] restoring gate/SigLIP text weights from {model_id}")
    load_sharded_checkpoint(model, str(model_path), strict=False, prefer_safe=True)
    validate_dual_input_gate_checkpoint_loaded(model, model_path)

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.use_liger:
        apply_liger_kernel_to_gemma3_text(
            rope=True, cross_entropy=False, fused_linear_cross_entropy=False, rms_norm=True, geglu=True
        )
    
    replace_gemma3_forward(use_liger=training_args.use_liger)

    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("If `lora_enable` is True, `freeze_llm` must also be True.")
    
    if training_args.vision_lora and not training_args.freeze_vision_tower:
        raise ValueError("If `vision_lora` is True, `freeze_vision_tower` must also be True.")

    if not training_args.lora_enable:
        assert not training_args.vision_lora, \
            "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."

    if training_args.lora_namespan_exclude is not None:
        training_args.lora_namespan_exclude = ast.literal_eval(training_args.lora_namespan_exclude)
    else:
        training_args.lora_namespan_exclude = ["multi_modal_projector"]

    if not training_args.vision_lora:
        training_args.lora_namespan_exclude += ["vision_tower", "multi_modal_projector"]

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4,8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"":training_args.device},
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=training_args.bits==4,
                load_in_8bit=training_args.bits==8,
                llm_int8_skip_modules=["vision_tower", "multi_modal_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            )
        ))

    attn_implementation = resolve_attn_implementation(training_args)
    rank0_print(f"Using attn_implementation={attn_implementation}")
    
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_args.model_id,
        torch_dtype=compute_dtype,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        **bnb_model_from_pretrained_args
    )
    
    model_to_configure = model
    configure_llm(model_to_configure, training_args)
    configure_vision_tower(model_to_configure, training_args, compute_dtype, training_args.device)
    gate_text_tokenizer = configure_dual_input_gate(model_to_configure, training_args, compute_dtype)
    if training_args.use_dual_input_gate:
        maybe_restore_dual_input_gate_from_checkpoint(model_to_configure, model_args.model_id)

    model.config.use_cache = False
    model.current_mask_patch_suppression_loss = None
    model.last_mask_patch_suppression_loss = None

    if training_args.bits in [4,8]:
        model.config.torch_dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        from peft import prepare_model_for_kbit_training
        # This is a workaround for a bug in the current implementation of gradient checkpointing
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing, gradient_checkpointing_kwargs={"use_reentrant": True})
    
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        # This is a workaround for a bug in the current implementation of gradient checkpointing
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    if training_args.lora_enable:
        lora_namespan_exclude = training_args.lora_namespan_exclude
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(model, lora_namespan_exclude=lora_namespan_exclude, num_lora_modules=training_args.num_lora_modules),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
        )

        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA to the model...")
        model = get_peft_model(model, peft_config)

        # Peft maodel makes vision tower and projector freezed again.
        # Configuring fuction could be called here, but sometimes it does not work properly.
        # So I just made it this way.
        # Need to be fixed in the future.

        if not training_args.freeze_vision_tower:
            for name, param in model.named_parameters():
                if "vision_tower" in name:
                    param.requires_grad = True

        if not training_args.freeze_projector:
            for name, param in model.named_parameters():
                if "multi_modal_projector" in name:
                    param.requires_grad = True

    processor = AutoProcessor.from_pretrained(model_args.model_id)
        
    model.config.vision_lr = training_args.vision_lr
    model.config.projector_lr = training_args.projector_lr

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(
        processor=processor,
        data_args=data_args,
        gate_text_tokenizer=gate_text_tokenizer,
        gate_text_max_length=training_args.gate_text_max_length,
    )

    trainer = GemmaSFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True
    
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )

        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=False
        )

        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_state_dict.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)



if __name__ == "__main__":
    train()
