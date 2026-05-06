from pathlib import Path

DEBUG_DIR = Path(__file__).resolve().parent
BUNDLE_ROOT = DEBUG_DIR.parents[1]
import os
import json
import torch
import transformers
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from src.dataset import make_dpo_data_module
from src.params import DataArguments, DPOArguments
from src.trainer import GemmaDPOTrainer
from monkey_patch_forward import replace_gemma3_forward
from train.train_dpo import configure_llm, configure_vision_tower
from train.train_sft import configure_dual_input_gate, maybe_restore_dual_input_gate_from_checkpoint

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
replace_gemma3_forward(use_liger=False)
model_id = os.environ.get(
    "MODEL_ID",
    str(BUNDLE_ROOT / "checkpoints/gemma3_4b_stage2_gate_l1_mask_sdpa/checkpoint-10293"),
)
data_path = str(DEBUG_DIR / "test_raw_with_shortcut_answer_16.json")
image_folder = str(BUNDLE_ROOT / "data/playground_data/coco/train2014")
compute_dtype = torch.bfloat16

def load_model(tag):
    print('loading', tag, flush=True)
    m = Gemma3ForConditionalGeneration.from_pretrained(model_id, torch_dtype=compute_dtype, attn_implementation='sdpa').cuda()
    args = DPOArguments(
        output_dir=str(DEBUG_DIR / "probe_out"),
        use_dual_input_gate=True,
        gate_text_model_id=str(BUNDLE_ROOT / "models/siglip-so400m-patch14-384"),
        freeze_gate_text_encoder=True,
        freeze_llm=False,
        freeze_vision_tower=True,
        freeze_projector=False,
        disable_flash_attn2=True,
        attn_implementation='sdpa',
        bf16=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        remove_unused_columns=False,
        napo_loss_type='dyn_lq',
        napo_dyn_q_use_average=True,
        report_to=[],
    )
    configure_llm(m, args)
    configure_vision_tower(m, args, compute_dtype, torch.device('cuda'))
    tok = configure_dual_input_gate(m, args, compute_dtype)
    maybe_restore_dual_input_gate_from_checkpoint(m, model_id)
    m.config.use_cache = False
    return m, tok, args

policy, gate_tok, args = load_model('policy')
ref, ref_gate_tok, _ = load_model('ref')
processor = AutoProcessor.from_pretrained(model_id)
data_args = DataArguments(data_path=data_path, image_folder=image_folder, lazy_preprocess=True)
module = make_dpo_data_module(processor, data_args, gate_text_tokenizer=gate_tok, gate_text_max_length=64)
examples = [module['train_dataset'][0]]
batch = module['data_collator'](examples)
for k, v in batch.items():
    if torch.is_tensor(v):
        batch[k] = v.cuda()
        print(k, tuple(v.shape), v.dtype, 'finite', bool(torch.isfinite(v.float()).all()) if v.is_floating_point() else 'int')
print('chosen ids', batch['chosen_input_ids'].detach().cpu().tolist())
print('rejected ids', batch['rejected_input_ids'].detach().cpu().tolist())
print('chosen text', processor.tokenizer.decode(batch['chosen_input_ids'][0].detach().cpu(), skip_special_tokens=False))
print('rejected text', processor.tokenizer.decode(batch['rejected_input_ids'][0].detach().cpu(), skip_special_tokens=False))
trainer = GemmaDPOTrainer(model=policy, ref_model=ref, train_dataset=module['train_dataset'], eval_dataset=None, data_collator=module['data_collator'], processing_class=processor, args=args)
def manual_model_probe(model, use_labels: bool):
    concat = trainer.concatenated_inputs(batch, padding_value=trainer.padding_value)
    prompt_input_ids = concat["prompt_input_ids"]
    prompt_attention_mask = concat["prompt_attention_mask"]
    completion_input_ids = concat["completion_input_ids"]
    completion_attention_mask = concat["completion_attention_mask"]
    input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
    attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
    token_type_ids = None
    if "token_type_ids" in concat:
        completion_token_type_ids = torch.zeros_like(completion_input_ids)
        token_type_ids = torch.cat((concat["token_type_ids"], completion_token_type_ids), dim=1)
    kwargs = {
        "pixel_values": concat["pixel_values"],
        "gate_input_ids": concat["gate_input_ids"],
        "gate_attention_mask": concat["gate_attention_mask"],
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }
    if use_labels:
        kwargs["labels"] = input_ids.clone()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids, **kwargs)
    logits = out.logits.detach().float()
    print("manual_forward", "with_labels" if use_labels else "no_labels", "finite", bool(torch.isfinite(logits).all()), "min", float(torch.nan_to_num(logits, nan=0).min().cpu()), "max", float(torch.nan_to_num(logits, nan=0).max().cpu()))

manual_model_probe(policy, use_labels=False)
manual_model_probe(policy, use_labels=True)

row = json.load(open(data_path))[0]
image_path = os.path.join(image_folder, f"COCO_train2014_{int(row['image_id']):012d}.jpg")
image = Image.open(image_path).convert("RGB")
messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": row["question"]}]}]
eval_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
eval_inputs = processor(text=[eval_prompt], images=[[image]], return_tensors="pt", padding=True)
gate_tokens = gate_tok(
    [row["question"]],
    add_special_tokens=True,
    truncation=True,
    max_length=64,
    padding=True,
    return_attention_mask=True,
    return_tensors="pt",
)
eval_inputs["gate_input_ids"] = gate_tokens["input_ids"]
eval_inputs["gate_attention_mask"] = gate_tokens["attention_mask"]
eval_inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in eval_inputs.items()}
print("eval_prompt_len", eval_inputs["input_ids"].shape, "token_type_ones", int((eval_inputs.get("token_type_ids", torch.zeros_like(eval_inputs["input_ids"])) == 1).sum().item()))
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    eval_out = policy(**eval_inputs)
eval_logits = eval_out.logits.detach().float()
print("eval_style_forward finite", bool(torch.isfinite(eval_logits).all()), "min", float(torch.nan_to_num(eval_logits, nan=0).min().cpu()), "max", float(torch.nan_to_num(eval_logits, nan=0).max().cpu()))

with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    po = trainer.concatenated_forward(policy, batch)
    ro = trainer.concatenated_forward(ref, batch)
for name, out in [('policy', po), ('ref', ro)]:
    print('---', name)
    for k, v in out.items():
        if torch.is_tensor(v):
            vf = v.detach().float()
            print(k, vf.cpu().tolist(), 'finite', bool(torch.isfinite(vf).all()))
        else:
            print(k, v)
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss, metrics = trainer.get_batch_loss_metrics(policy, batch, train_eval='train')
print('loss', loss.detach().float().cpu().item(), 'finite', bool(torch.isfinite(loss.detach()).all()))
print('metrics', metrics)
