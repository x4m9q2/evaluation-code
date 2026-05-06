import copy
import os
from typing import Dict, Optional
import re
import pathlib
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
    SYSTEM_MESSAGE,
    IGNORE_INDEX,
    LLAVA_IMAGE_TOKEN,
    LLAVA_VIDEO_TOKEN,
)
from .data_utils import (
    encode_video, 
    pad_sequence, 
    get_image_info,
    llava_to_openai
)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        gate_text_tokenizer=None,
        gate_text_max_length: int = 64,
        padding=True,
    ):
        super(SupervisedDataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = load_supervised_data(data_path)
        else:
            list_data_dict = data_path

        self.processor = processor
        self.list_data_dict = [
            normalize_multimodal_sample(sample, image_folder=data_args.image_folder)
            for sample in list_data_dict
        ]
        self.data_args = data_args
        self.padding = padding
        self.max_num_frames = data_args.max_num_frames
        self.gate_text_tokenizer = gate_text_tokenizer
        self.gate_text_max_length = gate_text_max_length
        self.patch_mask_coverage = None
        self.patch_mask_question_id_to_row = {}
        self.patch_mask_full_rows = None
        if data_args.patch_mask_analysis_path:
            (
                self.patch_mask_coverage,
                self.patch_mask_question_id_to_row,
                self.patch_mask_full_rows,
            ) = load_patch_mask_analysis(data_args.patch_mask_analysis_path)

    def __len__(self):
        return len(self.list_data_dict)

    def _extract_gate_text(self, conversations) -> str:
        parts = []
        for turn in conversations:
            if turn.get("from") != "human":
                continue
            text = turn.get("value", "")
            text = text.replace(LLAVA_IMAGE_TOKEN, " ")
            text = text.replace(LLAVA_VIDEO_TOKEN, " ")
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                parts.append(text)
        return "\n".join(parts)

    def _build_patch_mask_coverage(self, sample: Dict) -> np.ndarray | None:
        if self.patch_mask_coverage is None:
            return None
        sample_answer_type = str(sample.get("answer_type", ""))
        question_id = int(sample.get("question_id", -1))
        row_idx = self.patch_mask_question_id_to_row.get(question_id, None)
        if (
            row_idx is None
            or (self.data_args.disable_number_mask_loss and sample_answer_type == "number")
        ):
            return np.zeros((self.patch_mask_coverage.shape[1],), dtype=np.float32)
        patch_mask_coverage = self.patch_mask_coverage[row_idx]
        if self.patch_mask_full_rows is not None and bool(self.patch_mask_full_rows[row_idx]):
            return np.zeros((self.patch_mask_coverage.shape[1],), dtype=np.float32)
        return np.asarray(patch_mask_coverage, dtype=np.float32).copy()

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sample = self.list_data_dict[i]

        is_video = False
        num_frames = None
        pixel_values = None

        processor = self.processor
        if "image" in sample:
            image_files = sample["image"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]

            images = []

            for image_file in image_files:
                if not os.path.exists(image_file):
                    image_file = os.path.join(image_folder, image_file)
                images.append(Image.open(image_file).convert("RGB"))
            
            pixel_values = get_image_info(images, processor)

        elif "video" in sample:
            video_file = sample["video"]
            video_folder = self.data_args.image_folder

            if not os.path.exists(video_file):
                video_file = os.path.join(video_folder, video_file)

            images = encode_video(video_file, self.max_num_frames)
            
            is_video = True
            num_frames = len(images)
            pixel_values = get_image_info(images, processor)

        else:
            images = None

        sources = copy.deepcopy(llava_to_openai(sample['conversations'], is_video=is_video, num_frames=num_frames))

        all_input_ids = [torch.tensor([2])] # bos token id
        all_labels = [torch.tensor([-100])] # ignore bos token

        for idx, j in enumerate(range(0, len(sources), 2)):
            user_input = sources[j]
            gpt_response = sources[j + 1]

            if idx == 0 and len(SYSTEM_MESSAGE) > 0:
                user_input = f"{DEFAULT_START_TOKEN}{user_input['role']}\n{SYSTEM_MESSAGE}\n\n{user_input['content']}{DEFAULT_END_TOKEN}\n{DEFAULT_START_TOKEN}{gpt_response['role']}\n"
                gpt_response = f"{gpt_response['content']}{DEFAULT_END_TOKEN}\n"

            else:
                user_input = f"{DEFAULT_START_TOKEN}{user_input['role']}\n{user_input['content']}{DEFAULT_END_TOKEN}\n{DEFAULT_START_TOKEN}{gpt_response['role']}\n"
                gpt_response = f"{gpt_response['content']}{DEFAULT_END_TOKEN}\n"

            prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']
            response_input_ids = processor.tokenizer(gpt_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
            labels = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),  
                    response_input_ids.squeeze(0),
                ],
                dim=0,
            )

            all_input_ids.append(input_ids)
            all_labels.append(labels)
        
        # There is no need for eos tokens in the input_ids
        # Gemma3 does not use them
        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)

        attention_mask = (input_ids > -1000000).to(torch.long)

        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        if pixel_values is not None:
            array_ids = input_ids
            token_type_ids = np.zeros_like(input_ids)
            token_type_ids[array_ids == processor.image_token_id] = 1
            token_type_ids = torch.tensor(token_type_ids)

            data_dict["pixel_values"] = pixel_values
            data_dict["token_type_ids"] = token_type_ids
            patch_mask_coverage = self._build_patch_mask_coverage(sample)
            if patch_mask_coverage is not None:
                data_dict["patch_mask_coverage"] = torch.from_numpy(patch_mask_coverage)

        if self.gate_text_tokenizer is not None:
            gate_text = self._extract_gate_text(sample["conversations"])
            gate_tokens = self.gate_text_tokenizer(
                gate_text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.gate_text_max_length,
                padding=False,
                return_attention_mask=True,
                return_tensors="pt",
            )
            gate_attention_mask = gate_tokens.get("attention_mask", torch.ones_like(gate_tokens["input_ids"]))
            data_dict["gate_input_ids"] = gate_tokens["input_ids"].squeeze(0).to(torch.long)
            data_dict["gate_attention_mask"] = gate_attention_mask.squeeze(0).to(torch.long)
            
        return data_dict

class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int, gate_pad_token_id: int | None = None):
        self.pad_token_id = pad_token_id
        self.gate_pad_token_id = 0 if gate_pad_token_id is None else gate_pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_token_type_ids = []
        batch_gate_input_ids = []
        batch_gate_attention_mask = []
        batch_patch_mask_coverage = []
        
        for example in examples:
            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])
            if "pixel_values" in example:
                batch_pixel_values.append(example["pixel_values"])
                batch_token_type_ids.append(example["token_type_ids"])
            if "gate_input_ids" in example:
                batch_gate_input_ids.append(example["gate_input_ids"])
                batch_gate_attention_mask.append(example["gate_attention_mask"])
            if "patch_mask_coverage" in example:
                batch_patch_mask_coverage.append(example["patch_mask_coverage"])
        
        input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )

        attention_mask = input_ids != self.pad_token_id
        labels = pad_sequence(batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX)
        
        batch_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
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

        if len(batch_patch_mask_coverage) > 0:
            batch_dict["patch_mask_coverage"] = torch.stack(batch_patch_mask_coverage, dim=0)

        return batch_dict

def make_supervised_data_module(processor, data_args, gate_text_tokenizer=None, gate_text_max_length: int = 64):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = SupervisedDataset(
        data_path=data_args.data_path,
        processor=processor,
        data_args=data_args,
        gate_text_tokenizer=gate_text_tokenizer,
        gate_text_max_length=gate_text_max_length,
    )
    gate_pad_token_id = gate_text_tokenizer.pad_token_id if gate_text_tokenizer is not None else None
    data_collator = DataCollatorForSupervisedDataset(
        pad_token_id=processor.tokenizer.pad_token_id,
        gate_pad_token_id=gate_pad_token_id,
    )

    return dict(train_dataset=sft_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def build_train2014_image_name(image_id: int) -> str:
    return f"COCO_train2014_{int(image_id):012d}.jpg"


def load_supervised_data(data_path: str):
    if str(data_path).endswith(".jsonl"):
        rows = []
        with open(data_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with open(data_path, "r") as f:
        return json.load(f)


def _as_int(value, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_vqa_image(image_id: int, image_folder: Optional[str]) -> str:
    filename = build_train2014_image_name(image_id)
    candidates = [
        filename,
        os.path.join("coco", "train2014", filename),
    ]
    if image_folder:
        for candidate in candidates:
            if os.path.exists(os.path.join(image_folder, candidate)):
                return candidate
    return candidates[0]


def _infer_sage_as_source(sample: Dict) -> str:
    image_path = str(sample.get("image_path", ""))
    if {"question", "answer", "image_id"}.issubset(sample.keys()):
        return "vqa"
    if "/gqa/" in image_path or image_path.startswith("gqa/"):
        return "gqa"
    if "/vg/" in image_path or image_path.startswith("vg/"):
        return "vg"
    return str(sample.get("data_source", "")).lower()


def _resolve_vg_image(image_id: int, image_folder: Optional[str]) -> str:
    candidates = [
        os.path.join("vg", "VG_100K", f"{image_id}.jpg"),
        os.path.join("vg", "VG_100K_2", f"{image_id}.jpg"),
    ]
    if image_folder:
        for candidate in candidates:
            if os.path.exists(os.path.join(image_folder, candidate)):
                return candidate
    return candidates[0]


def normalize_multimodal_sample(sample: Dict, image_folder: Optional[str] = None) -> Dict:
    if "conversations" in sample:
        normalized = copy.deepcopy(sample)
        if "image" not in normalized and normalized.get("image_id") is not None:
            normalized["image"] = _resolve_vqa_image(_as_int(normalized["image_id"]), image_folder)
        return normalized

    if {"question", "answer", "image_id"}.issubset(sample.keys()):
        question = str(sample["question"]).strip()
        answer = str(sample["answer"]).strip()
        image_id = _as_int(sample["image_id"])
        return {
            "question_id": _as_int(sample.get("question_id", -1)),
            "image_id": image_id,
            "image": _resolve_vqa_image(image_id, image_folder),
            "answer_type": sample.get("answer_type", "other"),
            "data_source": sample.get("data_source", "vqa"),
            "mask_supervision": sample.get("mask_supervision", ""),
            "conversations": [
                {"from": "human", "value": f"{LLAVA_IMAGE_TOKEN}\n{question}"},
                {"from": "gpt", "value": answer},
            ],
        }

    if {"generated_question", "generated_answer", "image_id"}.issubset(sample.keys()):
        question = str(sample["generated_question"]).strip()
        answer = str(sample["generated_answer"]).strip()
        image_id = _as_int(sample["image_id"])
        source = _infer_sage_as_source(sample)
        if source == "gqa":
            image = os.path.join("gqa", "images", f"{image_id}.jpg")
        elif source == "vg":
            image = _resolve_vg_image(image_id, image_folder)
        else:
            image = str(sample.get("image_path", ""))
        return {
            "question_id": _as_int(sample.get("question_id", -1)),
            "image_id": image_id,
            "image": image,
            "answer_type": sample.get("answer_type", "other"),
            "data_source": sample.get("data_source", source),
            "mask_supervision": sample.get("mask_supervision", ""),
            "conversations": [
                {"from": "human", "value": f"{LLAVA_IMAGE_TOKEN}\n{question}"},
                {"from": "gpt", "value": answer},
            ],
        }

    return copy.deepcopy(sample)


def load_patch_mask_analysis(patch_mask_analysis_path: str):
    analysis = np.load(patch_mask_analysis_path, allow_pickle=True)
    coverage_ratio = analysis["coverage_ratio"].astype(np.float32).reshape(analysis["coverage_ratio"].shape[0], -1)
    full_mask_rows = np.zeros((coverage_ratio.shape[0],), dtype=np.bool_)
    question_id_to_row = {}

    if "question_ids" in analysis:
        qids = analysis["question_ids"].tolist()
        for idx, qid in enumerate(qids):
            question_id_to_row.setdefault(int(qid), idx)
    else:
        for idx, image_name in enumerate(analysis["image_names"]):
            stem = pathlib.Path(str(image_name)).stem
            question_id_str = stem.rsplit("_", 1)[0]
            if question_id_str.isdigit():
                question_id_to_row.setdefault(int(question_id_str), idx)

    has_mask = analysis["has_mask"].astype(np.bool_).reshape(analysis["has_mask"].shape[0], -1) if "has_mask" in analysis else None
    if has_mask is not None:
        full_mask_rows = has_mask.all(axis=1)

    return coverage_ratio, question_id_to_row, full_mask_rows
