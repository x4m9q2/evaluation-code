from pathlib import Path

DEBUG_DIR = Path(__file__).resolve().parent
BUNDLE_ROOT = DEBUG_DIR.parents[1]
import os, json
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel, AutoTokenizer, Gemma3ForConditionalGeneration
from monkey_patch_forward import replace_gemma3_forward
from train.train_sft import maybe_restore_dual_input_gate_from_checkpoint
from src.gate_model.build_gate_model import DualInputGate

replace_gemma3_forward(use_liger=False)
model_id=str(BUNDLE_ROOT / "checkpoints/gemma3_4b_stage2_gate_l1_mask_sdpa")
image_folder=str(BUNDLE_ROOT / "data/playground_data/coco/train2014")
row=json.load(open(DEBUG_DIR / "test_raw_with_shortcut_answer_16.json"))[0]
model=Gemma3ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda().train()
siglip=AutoModel.from_pretrained(str(BUNDLE_ROOT / "models/siglip-so400m-patch14-384"), torch_dtype=torch.bfloat16).cuda()
model.siglip_text_model=siglip.text_model
model.gate=DualInputGate(model.config.vision_config.hidden_size, model.siglip_text_model.config.hidden_size).cuda().to(dtype=torch.bfloat16)
model.config.use_dual_input_gate=True
del siglip
maybe_restore_dual_input_gate_from_checkpoint(model, model_id)
processor=AutoProcessor.from_pretrained(model_id)
gate_tok=AutoTokenizer.from_pretrained(str(BUNDLE_ROOT / "models/siglip-so400m-patch14-384"))
image=Image.open(os.path.join(image_folder, f"COCO_train2014_{int(row['image_id']):012d}.jpg")).convert('RGB')

def run_case(name, prompts, images, gate_texts):
    inputs=processor(text=prompts, images=[[img] for img in images], return_tensors='pt', padding=True)
    gt=gate_tok(gate_texts, add_special_tokens=True, truncation=True, max_length=64, padding=True, return_attention_mask=True, return_tensors='pt')
    inputs['gate_input_ids']=gt['input_ids']
    inputs['gate_attention_mask']=gt['attention_mask']
    inputs={k:v.cuda() if torch.is_tensor(v) else v for k,v in inputs.items()}
    tti=inputs.get('token_type_ids')
    print('\nCASE', name, 'input', tuple(inputs['input_ids'].shape), 'toktype ones', int((tti==1).sum().item()) if tti is not None else None)
    print('last ids', inputs['input_ids'][:, -10:].detach().cpu().tolist())
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        out=model(**inputs)
    logits=out.logits.detach().float()
    print('finite', bool(torch.isfinite(logits).all()), 'nan', int(torch.isnan(logits).sum().cpu()), 'inf', int(torch.isinf(logits).sum().cpu()), 'min', float(torch.nan_to_num(logits,nan=0,posinf=0,neginf=0).min().cpu()), 'max', float(torch.nan_to_num(logits,nan=0,posinf=0,neginf=0).max().cpu()))

base_messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':row['question']}]}]
prompt=processor.apply_chat_template(base_messages, add_generation_prompt=True, tokenize=False)
run_case('eval_b1_prompt_only', [prompt], [image], [row['question']])
run_case('eval_b2_prompt_only_dup', [prompt,prompt], [image,image], [row['question'],row['question']])
run_case('eval_b1_prompt_plus_answer', [prompt+row['answer']+'<end_of_turn>\n'], [image], [row['question']])
run_case('eval_b2_chosen_rejected', [prompt+row['answer']+'<end_of_turn>\n', prompt+row['shortcut_answer']+'<end_of_turn>\n'], [image,image], [row['question'],row['question']])
