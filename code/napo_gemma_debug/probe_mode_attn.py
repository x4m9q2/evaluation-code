from pathlib import Path

DEBUG_DIR = Path(__file__).resolve().parent
BUNDLE_ROOT = DEBUG_DIR.parents[1]
import os, json, sys
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel, AutoTokenizer, Gemma3ForConditionalGeneration
from safetensors import safe_open
from transformers.modeling_utils import load_sharded_checkpoint
from monkey_patch_forward import replace_gemma3_forward
from train.train_sft import configure_dual_input_gate, maybe_restore_dual_input_gate_from_checkpoint
from src.params import DPOArguments

replace_gemma3_forward(use_liger=False)
model_id = os.environ['MODEL_ID']
attn = os.environ.get('ATTN', 'sdpa')
mode = os.environ.get('MODE', 'train')
image_folder = str(BUNDLE_ROOT / "data/playground_data/coco/train2014")
data_path = str(DEBUG_DIR / "test_raw_with_shortcut_answer_16.json")
row = json.load(open(data_path))[0]
compute_dtype=torch.bfloat16
print('load', model_id, 'attn', attn, 'mode', mode, flush=True)
model = Gemma3ForConditionalGeneration.from_pretrained(model_id, torch_dtype=compute_dtype, attn_implementation=attn).cuda()
args = DPOArguments(output_dir=str(DEBUG_DIR / "unused"), use_dual_input_gate=True, gate_text_model_id=str(BUNDLE_ROOT / "models/siglip-so400m-patch14-384"), freeze_gate_text_encoder=True, freeze_llm=False, freeze_vision_tower=True, freeze_projector=False, bf16=True)
# minimal gate attach/restore, no require-grad config needed here
siglip = AutoModel.from_pretrained(args.gate_text_model_id, torch_dtype=compute_dtype).cuda()
from src.gate_model.build_gate_model import DualInputGate
model.siglip_text_model = siglip.text_model
model.gate = DualInputGate(model.config.vision_config.hidden_size, model.siglip_text_model.config.hidden_size).cuda().to(dtype=compute_dtype)
model.config.use_dual_input_gate = True
del siglip
maybe_restore_dual_input_gate_from_checkpoint(model, model_id)
processor = AutoProcessor.from_pretrained(model_id)
gate_tok = AutoTokenizer.from_pretrained(args.gate_text_model_id)
image = Image.open(os.path.join(image_folder, f"COCO_train2014_{int(row['image_id']):012d}.jpg")).convert('RGB')
messages = [{"role":"user", "content":[{"type":"image", "image":image}, {"type":"text", "text":row['question']}]}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = processor(text=[prompt], images=[[image]], return_tensors='pt', padding=True)
gate_tokens = gate_tok([row['question']], add_special_tokens=True, truncation=True, max_length=64, padding=True, return_attention_mask=True, return_tensors='pt')
inputs['gate_input_ids']=gate_tokens['input_ids']
inputs['gate_attention_mask']=gate_tokens['attention_mask']
inputs={k:v.cuda() if torch.is_tensor(v) else v for k,v in inputs.items()}
if mode == 'eval':
    model.eval()
else:
    model.train()
with torch.set_grad_enabled(mode=='train'), torch.autocast('cuda', dtype=torch.bfloat16):
    out = model(**inputs)
logits = out.logits.detach().float()
finite = bool(torch.isfinite(logits).all())
print('finite', finite, 'nan_count', int(torch.isnan(logits).sum().cpu()), 'inf_count', int(torch.isinf(logits).sum().cpu()), 'min', float(torch.nan_to_num(logits, nan=0, posinf=0, neginf=0).min().cpu()), 'max', float(torch.nan_to_num(logits, nan=0, posinf=0, neginf=0).max().cpu()))
