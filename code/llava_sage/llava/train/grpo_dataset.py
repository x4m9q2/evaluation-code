import json
import os
from typing import Dict, List, Sequence

import torch
import transformers

from PIL import Image
from torch.utils.data import Dataset

from llava import conversation as conversation_lib
from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN
from llava.mm_utils import process_images, tokenizer_image_token


class LazyGRPOPairDataset(Dataset):
    """Pair-level dataset for anti-shortcut GRPO training."""

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
            original_len = len(sample["original"]["question"].split())
            anti_shortcut_len = len(sample["anti_shortcut"]["question"].split())
            length_list.append(original_len + anti_shortcut_len + 256)
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

    def _build_clip_features(self, question: str) -> Dict[str, List[int]]:
        return self.clip_tokenizer(
            question,
            truncation=True,
            max_length=77,
        )

    def _build_pair_sample(self, question: str, answer: str) -> Dict:
        prompt = self._build_prompt(question)
        prompt_input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            return_tensors="pt",
        )
        clip_features = self._build_clip_features(question)
        return {
            "question": question,
            "answer": answer,
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": torch.ones_like(prompt_input_ids),
            "clip_input_ids": torch.tensor(clip_features["input_ids"], dtype=torch.long),
            "clip_attn_mask": torch.tensor(clip_features["attention_mask"], dtype=torch.long),
        }

    def __getitem__(self, i) -> Dict:
        sample = self.list_data_dict[i]
        image_tensor, image_size = self._load_image(sample["image"])

        original_pair = self._build_pair_sample(
            question=sample["original"]["question"],
            answer=sample["original"]["answer"],
        )
        anti_shortcut_pair = self._build_pair_sample(
            question=sample["anti_shortcut"]["question"],
            answer=sample["anti_shortcut"]["answer"],
        )

        return {
            "group_id": str(sample.get("group_id", i)),
            "answer_type": sample.get("answer_type", "other"),
            "image": image_tensor,
            "image_size": image_size,
            "pairs": [
                {"pair_role": "original", **original_pair},
                {"pair_role": "anti_shortcut", **anti_shortcut_pair},
            ],
        }


class DataCollatorForGRPODataset(object):
    """Collate pair-level groups into a flat prompt batch for GRPO."""

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
        flat_pairs = []
        group_ids = []
        answer_types = []
        image_sizes = []

        for instance in instances:
            for pair in instance["pairs"]:
                flat_pairs.append({
                    **pair,
                    "image": instance["image"],
                    "image_size": instance["image_size"],
                })
            group_ids.append(instance["group_id"])
            answer_types.append(instance["answer_type"])
            image_sizes.extend([instance["image_size"], instance["image_size"]])

        prompt_input_ids = [pair["prompt_input_ids"] for pair in flat_pairs]
        prompt_attention_mask = [pair["prompt_attention_mask"] for pair in flat_pairs]
        clip_input_ids = [pair["clip_input_ids"] for pair in flat_pairs]
        clip_attn_mask = [pair["clip_attn_mask"] for pair in flat_pairs]

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

        images = [pair["image"] for pair in flat_pairs]
        if all(image.shape == images[0].shape for image in images):
            images = torch.stack(images, dim=0)

        batch = {
            "group_ids": group_ids,
            "pair_roles": [pair["pair_role"] for pair in flat_pairs],
            "questions": [pair["question"] for pair in flat_pairs],
            "answers": [pair["answer"] for pair in flat_pairs],
            "answer_types": answer_types,
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "clip_input_ids": clip_input_ids,
            "clip_attn_mask": clip_attn_mask,
            "images": images,
            "image_sizes": image_sizes,
            "skip_batch": torch.tensor(bool((~eos_ok).any().item()), dtype=torch.bool),
        }
        return batch


def make_grpo_data_module(clip_tokenizer, tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    train_dataset = LazyGRPOPairDataset(
        clip_tokenizer=clip_tokenizer,
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_args=data_args,
    )
    data_collator = DataCollatorForGRPODataset(
        tokenizer=tokenizer,
        clip_eos_token_id=clip_tokenizer.eos_token_id,
    )
    return {
        "train_dataset": train_dataset,
        "eval_dataset": None,
        "data_collator": data_collator,
    }
