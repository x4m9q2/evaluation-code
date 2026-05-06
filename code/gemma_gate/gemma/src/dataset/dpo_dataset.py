import os
from typing import Dict
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from src.params import DataArguments
from src.constants import (
    DEFAULT_START_TOKEN, 
    DEFAULT_END_TOKEN, 
    LLAVA_IMAGE_TOKEN,
    SYSTEM_MESSAGE,
)
from .data_utils import (
    encode_video, 
    pad_sequence, 
    get_image_info, 
    video_to_image_tokens, 
    replace_image_tokens,
)


def build_train2014_image_name(image_id: int) -> str:
    return f"COCO_train2014_{int(image_id):012d}.jpg"


def normalize_dpo_sample(sample: dict) -> dict:
    if {"prompt", "chosen", "rejected"}.issubset(sample.keys()):
        normalized = dict(sample)
        if "image" not in normalized and normalized.get("image_id") is not None:
            normalized["image"] = build_train2014_image_name(normalized["image_id"])
        return normalized

    if {"question", "answer", "shortcut_answer"}.issubset(sample.keys()):
        normalized = {
            "prompt": f"{LLAVA_IMAGE_TOKEN}\n{sample['question']}",
            "chosen": sample["answer"],
            "rejected": sample["shortcut_answer"],
        }
        for key in ("id", "question_id", "image_id", "answer_type"):
            if key in sample:
                normalized[key] = sample[key]
        if sample.get("image") is not None:
            normalized["image"] = sample["image"]
        elif sample.get("image_id") is not None:
            normalized["image"] = build_train2014_image_name(sample["image_id"])
        return normalized

    if {"question", "chosen", "rejected"}.issubset(sample.keys()):
        normalized = {
            "prompt": f"{LLAVA_IMAGE_TOKEN}\n{sample['question']}",
            "chosen": sample["chosen"],
            "rejected": sample["rejected"],
        }
        for key in ("id", "idx", "question_id", "image_id", "answer_type", "origin_dataset", "origin_split"):
            if key in sample:
                normalized[key] = sample[key]
        if sample.get("image") is not None:
            normalized["image"] = sample["image"]
        elif sample.get("image_id") is not None:
            normalized["image"] = build_train2014_image_name(sample["image_id"])
        return normalized

    return sample


class DPODataset(Dataset):
    """Dataset for DPO training."""

    def __init__(
            self,
            data_path: str | list,
            processor: transformers.ProcessorMixin,
            data_args: DataArguments,
            gate_text_tokenizer=None,
            gate_text_max_length: int = 64,
            padding=True,
    ):
        super(DPODataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.max_num_frames = data_args.max_num_frames
        self.gate_text_tokenizer = gate_text_tokenizer
        self.gate_text_max_length = gate_text_max_length

    def __len__(self):
        return len(self.list_data_dict)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = normalize_dpo_sample(self.list_data_dict[i])

        is_video = False
        num_frames = None
        pixel_values = None

        processor = self.processor
        if "image" in sources:
            image_files = sources["image"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]

            images = []

            for image_file in image_files:
                if not os.path.exists(image_file):
                    image_file = os.path.join(image_folder, image_file)
                images.append(Image.open(image_file).convert("RGB"))
            
            pixel_values = get_image_info(images, processor)

        elif "video" in sources:
            video_file = sources["video"]
            video_folder = self.data_args.image_folder

            if not os.path.exists(video_file):
                video_file = os.path.join(video_folder, video_file)

            images = encode_video(video_file, self.max_num_frames)
            
            is_video = True
            num_frames = len(images)
            pixel_values = get_image_info(images, processor)

        else:
            images = None

        all_input_ids = [torch.tensor([2])]
        all_rejected = []
        all_chosen = []

        user_prompt = video_to_image_tokens(sources['prompt'], num_frames) if is_video else sources['prompt']
        user_prompt = replace_image_tokens(user_prompt)

        if len(SYSTEM_MESSAGE) > 0:
            user_input = f"{DEFAULT_START_TOKEN}user\n{SYSTEM_MESSAGE}\n\n{user_prompt}{DEFAULT_END_TOKEN}\n{DEFAULT_START_TOKEN}assistant\n"
        else:
            user_input = f"{DEFAULT_START_TOKEN}user\n{user_prompt}{DEFAULT_END_TOKEN}\n{DEFAULT_START_TOKEN}assistant\n"

        chosen_response = f"{sources['chosen']}{DEFAULT_END_TOKEN}\n"
        rejected_response = f"{sources['rejected']}{DEFAULT_END_TOKEN}\n"

        prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids'].squeeze(0)
        chosen_input_ids = processor.tokenizer(chosen_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids'].squeeze(0)
        rejected_input_ids = processor.tokenizer(rejected_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids'].squeeze(0)
        
        
        all_input_ids.append(prompt_input_ids)
        all_chosen.append(chosen_input_ids)
        all_rejected.append(rejected_input_ids)

        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        chosen = torch.cat(all_chosen, dim=0).to(torch.long)
        rejected = torch.cat(all_rejected, dim=0).to(torch.long)

        data_dict = dict(
            prompt_input_ids=input_ids,
            chosen_input_ids=chosen,
            rejected_input_ids=rejected,
        )

        if pixel_values is not None:
            array_ids = input_ids
            token_type_ids = np.zeros_like(input_ids)
            token_type_ids[array_ids == processor.image_token_id] = 1
            token_type_ids = torch.tensor(token_type_ids)

            data_dict["pixel_values"] = pixel_values
            data_dict["token_type_ids"] = token_type_ids

        if self.gate_text_tokenizer is not None:
            gate_tokens = self.gate_text_tokenizer(
                sources["prompt"].replace(LLAVA_IMAGE_TOKEN, "").strip(),
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.gate_text_max_length,
            )
            gate_attention_mask = gate_tokens.get("attention_mask", torch.ones_like(gate_tokens["input_ids"]))
            data_dict["gate_input_ids"] = gate_tokens["input_ids"].squeeze(0).to(torch.long)
            data_dict["gate_attention_mask"] = gate_attention_mask.squeeze(0).to(torch.long)

        return data_dict
    
class DataCollatorForDPODataset(object):
    """Collate examples for DPO fine-tuning."""

    def __init__(self, pad_token_id: int, gate_pad_token_id: int | None = None):
        self.pad_token_id = pad_token_id
        self.gate_pad_token_id = 0 if gate_pad_token_id is None else gate_pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_chosen_ids = []
        batch_rejected_ids = []
        batch_pixel_values = []
        batch_token_type_ids = []
        batch_gate_input_ids = []
        batch_gate_attention_mask = []

        for example in examples:
            batch_input_ids.append(example["prompt_input_ids"])
            batch_chosen_ids.append(example["chosen_input_ids"])
            batch_rejected_ids.append(example["rejected_input_ids"])
            if "pixel_values" in example:
                batch_pixel_values.append(example["pixel_values"])
                batch_token_type_ids.append(example["token_type_ids"])
            if "gate_input_ids" in example:
                batch_gate_input_ids.append(example["gate_input_ids"])
                batch_gate_attention_mask.append(example["gate_attention_mask"])

        prompt_input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )
        chosen = pad_sequence(batch_chosen_ids, padding_side='right', padding_value=self.pad_token_id)
        rejected = pad_sequence(batch_rejected_ids, padding_side='right', padding_value=self.pad_token_id)

        prompt_attention_mask = prompt_input_ids != self.pad_token_id
        chosen_attention_mask = chosen != self.pad_token_id
        rejected_attention_mask = rejected != self.pad_token_id


        batch_dict = {
            'prompt_input_ids': prompt_input_ids,
            'prompt_attention_mask': prompt_attention_mask,
            'chosen_input_ids': chosen,
            'chosen_attention_mask': chosen_attention_mask,
            'rejected_input_ids': rejected,
            'rejected_attention_mask': rejected_attention_mask,
        }

        if len(batch_pixel_values) > 0:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            token_type_ids = pad_sequence(batch_token_type_ids, padding_side='right', padding_value=0)
            batch_dict.update(pixel_values=pixel_values, token_type_ids=token_type_ids)

        if len(batch_gate_input_ids) > 0:
            gate_input_ids = pad_sequence(
                batch_gate_input_ids, padding_side='right', padding_value=self.gate_pad_token_id
            )
            gate_attention_mask = pad_sequence(
                batch_gate_attention_mask, padding_side='right', padding_value=0
            )
            batch_dict.update(gate_input_ids=gate_input_ids, gate_attention_mask=gate_attention_mask)

        return batch_dict

def make_dpo_data_module(processor, data_args, gate_text_tokenizer=None, gate_text_max_length: int = 64):
    """Make dataset and collator for DPO fine-tuning."""
    dpo_dataset = DPODataset(
        data_path=data_args.data_path,
        processor=processor,
        data_args=data_args,
        gate_text_tokenizer=gate_text_tokenizer,
        gate_text_max_length=gate_text_max_length,
    )
    gate_pad_token_id = gate_text_tokenizer.pad_token_id if gate_text_tokenizer is not None else None
    data_collator = DataCollatorForDPODataset(
        pad_token_id=processor.tokenizer.pad_token_id,
        gate_pad_token_id=gate_pad_token_id,
    )

    return dict(train_dataset=dpo_dataset,
                eval_dataset=None,
                data_collator=data_collator)
