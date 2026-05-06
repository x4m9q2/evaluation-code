import json
import os
from typing import Dict, Sequence

import torch
import transformers

from PIL import Image
from torch.utils.data import Dataset

from llava import conversation as conversation_lib
from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN
from llava.mm_utils import process_images, tokenizer_image_token


class LazyGRPOAntiShortcutDataset(Dataset):
    """Single-question anti-shortcut dataset for GRPO training."""

    def __init__(
        self,
        clip_tokenizer,
        tokenizer: transformers.PreTrainedTokenizer,
        data_path: str,
        data_args,
    ):
        super().__init__()
        with open(data_path, "r") as f:
            self.list_data_dict = json.load(f)

        self.clip_tokenizer = clip_tokenizer
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            anti_len = len(sample["anti_shortcut"]["question"].split())
            length_list.append(anti_len + 256)
        return length_list

    @property
    def modality_lengths(self):
        return self.lengths

    def _resolve_image_path(self, image_name: str) -> str:
        if os.path.isabs(image_name):
            return image_name
        if self.data_args.image_folder is None:
            return image_name
        return os.path.join(self.data_args.image_folder, image_name)

    def _load_image(self, image_name: str):
        image_path = self._resolve_image_path(image_name)
        image = Image.open(image_path).convert("RGB")
        image_tensor = process_images([image], self.data_args.image_processor, self.data_args)[0]
        return image_tensor, image.size

    def _build_prompt(self, question: str) -> str:
        if getattr(self.data_args, "mm_use_im_start_end", False):
            question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
        else:
            question = DEFAULT_IMAGE_TOKEN + "\n" + question

        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def _build_clip_features(self, question: str) -> Dict[str, Sequence[int]]:
        return self.clip_tokenizer(
            question,
            truncation=True,
            max_length=77,
        )

    def __getitem__(self, i) -> Dict:
        sample = self.list_data_dict[i]
        image_tensor, image_size = self._load_image(sample["image"])

        question = sample["anti_shortcut"]["question"]
        prompt = self._build_prompt(question)
        prompt_input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            return_tensors="pt",
        )
        clip_features = self._build_clip_features(question)

        return {
            "group_id": str(sample.get("group_id", i)),
            "answer_type": sample.get("answer_type", "other"),
            "question": question,
            "answer": sample["anti_shortcut"]["answer"],
            "original_answer": sample["original"]["answer"],
            "image": image_tensor,
            "image_size": image_size,
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": torch.ones_like(prompt_input_ids),
            "clip_input_ids": torch.tensor(clip_features["input_ids"], dtype=torch.long),
            "clip_attn_mask": torch.tensor(clip_features["attention_mask"], dtype=torch.long),
        }


class DataCollatorForGRPOAntiShortcutDataset(object):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, clip_eos_token_id: int = 49407):
        self.tokenizer = tokenizer
        self.clip_eos_token_id = clip_eos_token_id

    def _pad_clip_features(self, feature_list: Sequence[torch.Tensor], max_len: int = 77) -> torch.Tensor:
        padded = []
        for feature in feature_list:
            feature = feature[:max_len]
            if feature.shape[0] < max_len:
                pad = torch.zeros(max_len - feature.shape[0], dtype=feature.dtype)
                feature = torch.cat([feature, pad], dim=0)
            padded.append(feature)
        return torch.stack(padded, dim=0)

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        prompt_input_ids = [instance["prompt_input_ids"] for instance in instances]
        prompt_attention_mask = [instance["prompt_attention_mask"] for instance in instances]
        clip_input_ids = [instance["clip_input_ids"] for instance in instances]
        clip_attn_mask = [instance["clip_attn_mask"] for instance in instances]

        prompt_input_ids = torch.nn.utils.rnn.pad_sequence(
            prompt_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        prompt_attention_mask = torch.nn.utils.rnn.pad_sequence(
            prompt_attention_mask,
            batch_first=True,
            padding_value=0,
        )
        prompt_input_ids = prompt_input_ids[:, : self.tokenizer.model_max_length]
        prompt_attention_mask = prompt_attention_mask[:, : self.tokenizer.model_max_length]

        clip_input_ids = self._pad_clip_features(clip_input_ids)
        clip_attn_mask = self._pad_clip_features(clip_attn_mask)
        eos_ok = clip_input_ids.eq(self.clip_eos_token_id).any(dim=1)

        images = [instance["image"] for instance in instances]
        if all(image.shape == images[0].shape for image in images):
            images = torch.stack(images, dim=0)

        batch = {
            "group_ids": [instance["group_id"] for instance in instances],
            "answer_types": [instance["answer_type"] for instance in instances],
            "questions": [instance["question"] for instance in instances],
            "answers": [instance["answer"] for instance in instances],
            "original_answers": [instance["original_answer"] for instance in instances],
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "clip_input_ids": clip_input_ids,
            "clip_attn_mask": clip_attn_mask,
            "images": images,
            "image_sizes": [instance["image_size"] for instance in instances],
            "skip_batch": torch.tensor(bool((~eos_ok).any().item()), dtype=torch.bool),
        }
        return batch


def make_grpo_anti_data_module(clip_tokenizer, tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    train_dataset = LazyGRPOAntiShortcutDataset(
        clip_tokenizer=clip_tokenizer,
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_args=data_args,
    )
    eval_dataset = None
    if getattr(data_args, "eval_data_path", None):
        eval_dataset = LazyGRPOAntiShortcutDataset(
            clip_tokenizer=clip_tokenizer,
            tokenizer=tokenizer,
            data_path=data_args.eval_data_path,
            data_args=data_args,
        )

    data_collator = DataCollatorForGRPOAntiShortcutDataset(
        tokenizer=tokenizer,
        clip_eos_token_id=clip_tokenizer.eos_token_id,
    )
    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }
