import argparse
import math
import os
import json
import uuid

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import CLIPTokenizer

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path


def parse_torch_dtype(value):
    if value is None or value == "auto":
        return None
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    normalized = str(value).lower().replace("torch.", "")
    if normalized not in dtype_map:
        raise ValueError(f"Unsupported torch dtype: {value}")
    return dtype_map[normalized]


def infer_checkpoint_torch_dtype(model_path):
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return torch.float16
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return parse_torch_dtype(config.get("torch_dtype")) or torch.float16


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(
        self,
        questions,
        image_folder,
        tokenizer,
        image_processor,
        model_config,
        conv_mode,
        clip_tokenizer,
        patch_mask_coverage_by_qid=None,
    ):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode
        self.clip_tokenizer = clip_tokenizer
        self.clip_eos_token_id = clip_tokenizer.eos_token_id
        self.patch_mask_coverage_by_qid = patch_mask_coverage_by_qid

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        cur_prompt = line["text"]
        qs = cur_prompt
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        clip_ids = self.clip_tokenizer(
            cur_prompt,
            truncation=True,
            max_length=77,
        )
        clip_input_ids = clip_ids["input_ids"]
        clip_attn_mask = clip_ids["attention_mask"]
        if self.clip_eos_token_id is not None and self.clip_eos_token_id not in clip_input_ids:
            if len(clip_input_ids) >= 77:
                clip_input_ids[-1] = self.clip_eos_token_id
            else:
                clip_input_ids.append(self.clip_eos_token_id)
                clip_attn_mask.append(1)

        item = {
            "question_id": line["question_id"],
            "prompt": cur_prompt,
            "input_ids": input_ids,
            "image_tensor": image_tensor,
            "image_size": image.size,
            "clip_input_ids": torch.tensor((clip_input_ids + [0] * 77)[:77], dtype=torch.long),
            "clip_attn_mask": torch.tensor((clip_attn_mask + [0] * 77)[:77], dtype=torch.long),
        }
        if self.patch_mask_coverage_by_qid is not None:
            patch_mask_coverage = self.patch_mask_coverage_by_qid.get(int(line["question_id"]))
            if patch_mask_coverage is None:
                patch_mask_coverage = np.zeros((24 * 24,), dtype=np.float32)
            item["patch_mask_coverage"] = torch.tensor(patch_mask_coverage, dtype=torch.float32)
        return item

    def __len__(self):
        return len(self.questions)


class DataCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def _left_pad(self, tensors, pad_value):
        max_len = max(tensor.shape[0] for tensor in tensors)
        padded = tensors[0].new_full((len(tensors), max_len), pad_value)
        for idx, tensor in enumerate(tensors):
            padded[idx, -tensor.shape[0]:] = tensor
        return padded

    def __call__(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [torch.ones_like(item["input_ids"]) for item in batch]
        image_tensors = [item["image_tensor"] for item in batch]

        output = {
            "question_ids": [item["question_id"] for item in batch],
            "prompts": [item["prompt"] for item in batch],
            "input_ids": self._left_pad(input_ids, self.pad_token_id),
            "attention_mask": self._left_pad(attention_mask, 0),
            "image_tensors": torch.stack(image_tensors, dim=0),
            "image_sizes": [item["image_size"] for item in batch],
            "clip_input_ids": torch.stack([item["clip_input_ids"] for item in batch], dim=0),
            "clip_attn_mask": torch.stack([item["clip_attn_mask"] for item in batch], dim=0),
        }
        if "patch_mask_coverage" in batch[0]:
            output["patch_mask_coverage"] = torch.stack([item["patch_mask_coverage"] for item in batch], dim=0)
        return output


# DataLoader
def create_data_loader(
    questions,
    image_folder,
    tokenizer,
    image_processor,
    model_config,
    conv_mode,
    clip_tokenizer,
    patch_mask_coverage_by_qid=None,
    batch_size=1,
    num_workers=4,
):
    dataset = CustomDataset(
        questions,
        image_folder,
        tokenizer,
        image_processor,
        model_config,
        conv_mode,
        clip_tokenizer,
        patch_mask_coverage_by_qid=patch_mask_coverage_by_qid,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=DataCollator(tokenizer.pad_token_id),
    )
    return data_loader


def load_patch_mask_analysis(path):
    analysis = np.load(os.path.expanduser(path), allow_pickle=True)
    question_ids = analysis["question_ids"].astype(np.int64)
    coverage = analysis["coverage_ratio"].astype(np.float32).reshape(question_ids.shape[0], -1)
    return {int(qid): coverage[idx] for idx, qid in enumerate(question_ids.tolist())}


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    gate_override = None
    if args.force_use_dual_input_gate == "true":
        gate_override = True
    elif args.force_use_dual_input_gate == "false":
        gate_override = False
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        device_map=None,
        device='cuda',
        force_use_dual_input_gate=gate_override,
        torch_dtype=parse_torch_dtype(args.torch_dtype) or infer_checkpoint_torch_dtype(model_path),
    )
    tokenizer.padding_side = "left"
    model.config.tokenizer_padding_side = "left"
    print(
        "use_dual_input_gate="
        f"{getattr(model.get_model(), 'use_dual_input_gate', getattr(model.config, 'use_dual_input_gate', None))} "
        f"(override={args.force_use_dual_input_gate})"
    )
    model.get_model().runtime_patch_gate_suppress_ratio = float(args.gate_patch_suppress_ratio)
    if image_processor is None and hasattr(model, "get_vision_tower"):
        vision_tower = model.get_vision_tower()
        if vision_tower is not None:
            if not getattr(vision_tower, "is_loaded", True):
                vision_tower.load_model(device_map='cuda')
            image_processor = getattr(vision_tower, "image_processor", None)
    if image_processor is None:
        raise ValueError(
            f"Failed to initialize image_processor for model {model_path}. "
            "Please make sure the checkpoint includes a valid vision tower config."
        )
    clip_tokenizer = CLIPTokenizer.from_pretrained(model.config.mm_vision_tower)
    clip_eos_token_id = clip_tokenizer.eos_token_id

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    patch_mask_coverage_by_qid = None
    if args.patch_mask_analysis_path:
        patch_mask_coverage_by_qid = load_patch_mask_analysis(args.patch_mask_analysis_path)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(
        questions,
        args.image_folder,
        tokenizer,
        image_processor,
        model.config,
        args.conv_mode,
        clip_tokenizer,
        patch_mask_coverage_by_qid=patch_mask_coverage_by_qid,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    for batch in tqdm(data_loader, total=len(data_loader)):
        input_ids = batch["input_ids"].to(device='cuda', non_blocking=True)
        attention_mask = batch["attention_mask"].to(device='cuda', non_blocking=True)
        clip_input_ids = batch["clip_input_ids"].to(device='cuda', non_blocking=True)
        clip_attn_mask = batch["clip_attn_mask"].to(device='cuda', non_blocking=True)

        generate_kwargs = {}
        if args.begin_suppress_eos:
            eos_token_id = tokenizer.eos_token_id
            if eos_token_id is not None:
                generate_kwargs["begin_suppress_tokens"] = [eos_token_id]

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                images=batch["image_tensors"].to(dtype=next(model.parameters()).dtype, device='cuda', non_blocking=True),
                image_sizes=batch["image_sizes"],
                clip_input_ids=clip_input_ids,
                clip_attn_mask=clip_attn_mask,
                patch_mask_coverage=batch.get("patch_mask_coverage", None).to(device='cuda', non_blocking=True)
                if batch.get("patch_mask_coverage", None) is not None else None,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                **generate_kwargs)

        outputs = [output.strip() for output in tokenizer.batch_decode(output_ids, skip_special_tokens=True)]

        for idx, cur_prompt, output in zip(batch["question_ids"], batch["prompts"], outputs):
            ans_id = uuid.uuid4().hex
            ans_file.write(json.dumps({"question_id": idx,
                                       "prompt": cur_prompt,
                                       "text": output,
                                       "answer_id": ans_id,
                                       "model_id": model_name,
                                       "metadata": {}}) + "\n")
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--patch-mask-analysis-path", type=str, default=None)
    parser.add_argument("--gate-patch-suppress-ratio", type=float, default=0.0)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument(
        "--begin-suppress-eos",
        action="store_true",
        help="Suppress EOS only at the first generated token to avoid empty decoded answers.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
        default="bf16",
        help="Inference dtype. Defaults to bf16.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--force-use-dual-input-gate",
        type=str,
        choices=["auto", "true", "false"],
        default="auto",
        help="Override whether to enable the dual-input gate during inference.",
    )
    args = parser.parse_args()

    eval_model(args)
