# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
import sys
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
from transformers import CLIPTokenizer, AutoTokenizer
import torch
import numpy as np

import transformers
import tokenizers

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.model.builder import checkpoint_has_gate_weights
from llava.mm_utils import tokenizer_image_token

from PIL import Image


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    use_dual_input_gate: bool = field(default=False)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    eval_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to validation data for loss evaluation."},
    )
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    patch_mask_analysis_path: Optional[str] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    mask_patch_loss_weight: float = field(default=0.0)
    gate_l1_loss_weight: float = field(default=0.0)
    lr_scheduler_total_steps_scale: float = field(
        default=1.0,
        metadata={
            "help": "Scale scheduler horizon without changing actual max_steps/epochs."
        },
    )


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector', 'model.gate.']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    clip_tokenizer,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    question = sources[0][0]['value']

    # 去掉开头的 <image>\n
    # replace 比较稳妥，或者使用 removeprefix (Python 3.9+)
    if question.startswith("<image>\n"):
        clean_question = question.replace("<image>\n", "", 1)
    else:
        clean_question = question
    clip_ids = clip_tokenizer(
        clean_question,
        truncation=True,
        max_length=77,
    )
        # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        clip_ids = clip_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    clip_tokenizer,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, clip_tokenizer,tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


def build_train2014_image_name(image_id: int) -> str:
    return f"COCO_train2014_{int(image_id):012d}.jpg"


def load_supervised_data(data_path: str) -> List[Dict]:
    if data_path.endswith(".jsonl"):
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
                {"from": "human", "value": f"{DEFAULT_IMAGE_TOKEN}\n{question}"},
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
                {"from": "human", "value": f"{DEFAULT_IMAGE_TOKEN}\n{question}"},
                {"from": "gpt", "value": answer},
            ],
        }

    return copy.deepcopy(sample)


def load_patch_mask_analysis(patch_mask_analysis_path: str):
    analysis = np.load(patch_mask_analysis_path, allow_pickle=True)
    coverage_ratio = analysis["coverage_ratio"].astype(np.float32).reshape(analysis["coverage_ratio"].shape[0], -1)
    has_mask = analysis["has_mask"].astype(np.bool_).reshape(analysis["has_mask"].shape[0], -1)
    question_id_to_row = {}
    full_mask_rows = np.zeros((coverage_ratio.shape[0],), dtype=np.bool_)
    if "question_ids" in analysis:
        for idx, question_id in enumerate(analysis["question_ids"]):
            question_id_to_row.setdefault(_as_int(question_id), idx)
    elif "image_names" in analysis:
        for idx, image_name in enumerate(analysis["image_names"]):
            stem = pathlib.Path(str(image_name)).stem
            question_id_str = stem.rsplit("_", 1)[0]
            if question_id_str.isdigit():
                question_id_to_row.setdefault(int(question_id_str), idx)
    for idx in range(has_mask.shape[0]):
        # Degenerate masks that activate every patch are effectively full-image suppression.
        if bool(has_mask[idx].all()):
            full_mask_rows[idx] = True
    return coverage_ratio, question_id_to_row, full_mask_rows


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, clip_tokenizer,data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = load_supervised_data(data_path)

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.clip_tokenizer = clip_tokenizer
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.list_data_dict = [
            normalize_multimodal_sample(sample, image_folder=data_args.image_folder)
            for sample in list_data_dict
        ]
        self.patch_mask_coverage = None
        self.patch_mask_question_id_to_row = {}
        self.patch_mask_full_rows = None
        if data_args.patch_mask_analysis_path:
            (
                self.patch_mask_coverage,
                self.patch_mask_question_id_to_row,
                self.patch_mask_full_rows,
            ) = load_patch_mask_analysis(
                data_args.patch_mask_analysis_path
            )
            matched = sum(
                int(sample.get("question_id", -1)) in self.patch_mask_question_id_to_row
                for sample in self.list_data_dict
            )
            full_mask_count = int(self.patch_mask_full_rows.sum()) if self.patch_mask_full_rows is not None else 0
            rank0_print(
                f"Loaded patch mask analysis from {data_args.patch_mask_analysis_path}: "
                f"matched {matched}/{len(self.list_data_dict)} samples; "
                f"degenerate_full_mask_rows={full_mask_count}"
            )

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.clip_tokenizer,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             clip_ids = data_dict['clip_ids'],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        row_idx = None
        sample_answer_type = str(self.list_data_dict[i].get("answer_type", ""))
        sample_mask_supervision = str(self.list_data_dict[i].get("mask_supervision", ""))
        if self.patch_mask_coverage is not None:
            question_id = int(self.list_data_dict[i].get("question_id", -1))
            row_idx = self.patch_mask_question_id_to_row.get(question_id, None)
            if (
                row_idx is None
                or sample_answer_type == "number"
            ):
                patch_mask_coverage = np.zeros((self.patch_mask_coverage.shape[1],), dtype=np.float32)
            else:
                patch_mask_coverage = self.patch_mask_coverage[row_idx]
                if self.patch_mask_full_rows is not None and bool(self.patch_mask_full_rows[row_idx]):
                    patch_mask_coverage = np.zeros((self.patch_mask_coverage.shape[1],), dtype=np.float32)
            data_dict['patch_mask_coverage'] = torch.from_numpy(
                np.asarray(patch_mask_coverage, dtype=np.float32).copy()
            )
        data_dict["question_id"] = int(self.list_data_dict[i].get("question_id", -1))
        data_dict["data_source"] = str(self.list_data_dict[i].get("data_source", ""))
        data_dict["answer_type"] = sample_answer_type
        mask_supervision = sample_mask_supervision
        if self.patch_mask_coverage is not None and row_idx is not None:
            mask_supervision = "sam3_patch_mask"
        if (
            self.patch_mask_coverage is not None
            and row_idx is not None
            and self.patch_mask_full_rows is not None
            and bool(self.patch_mask_full_rows[row_idx])
        ):
            mask_supervision = "none"
        if sample_answer_type == "number":
            mask_supervision = "none"
        data_dict["mask_supervision"] = mask_supervision
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    clip_eos_token_id: int = 49407

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids,clip_input, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids","clip_ids", "labels"))
        MAX_LEN = 77

        # 暴力截断或补零，不检查输入是否合法
        # 长度多了就切掉，少了就加 [0]
        input_ids_fixed = [
            (ids + [0] * MAX_LEN)[:MAX_LEN] 
            for ids in [item['input_ids'] for item in clip_input]
        ]

        attn_mask_fixed = [
            (mask + [0] * MAX_LEN)[:MAX_LEN] 
            for mask in [item['attention_mask'] for item in clip_input]
        ]
        clip_input_ids = torch.tensor(input_ids_fixed)
        clip_attn_mask = torch.tensor(attn_mask_fixed)
        eos_ok = clip_input_ids.eq(self.clip_eos_token_id).any(dim=1)
        skip_batch = bool((~eos_ok).any().item())

        # 转换为 Tensor
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            clip_input_ids = clip_input_ids,
            clip_attn_mask = clip_attn_mask,
            skip_batch = torch.tensor(skip_batch, dtype=torch.bool),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        if 'patch_mask_coverage' in instances[0]:
            batch['patch_mask_coverage'] = torch.stack(
                [instance['patch_mask_coverage'] for instance in instances], dim=0
            )

        if 'question_id' in instances[0]:
            batch['question_ids'] = torch.tensor(
                [int(instance['question_id']) for instance in instances], dtype=torch.long
            )
        if 'data_source' in instances[0]:
            batch['data_sources'] = [str(instance['data_source']) for instance in instances]
        if 'answer_type' in instances[0]:
            batch['answer_types'] = [str(instance['answer_type']) for instance in instances]
        if 'mask_supervision' in instances[0]:
            batch['mask_supervisions'] = [str(instance['mask_supervision']) for instance in instances]

        return batch


def make_supervised_data_module(clip_tokenizer,tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(clip_tokenizer=clip_tokenizer,tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    eval_dataset = None
    if data_args.eval_data_path:
        eval_dataset = LazySupervisedDataset(
            clip_tokenizer=clip_tokenizer,
            tokenizer=tokenizer,
            data_path=data_args.eval_data_path,
            data_args=data_args,
        )
    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer,
        clip_eos_token_id=clip_tokenizer.eos_token_id,
    )
    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)


def train(attn_implementation=None):
    global local_rank

    training_arg_fields = getattr(TrainingArguments, "__dataclass_fields__", {})
    if (
        "--evaluation_strategy" in sys.argv
        and "evaluation_strategy" not in training_arg_fields
        and "eval_strategy" in training_arg_fields
    ):
        sys.argv = [
            "--eval_strategy" if arg == "--evaluation_strategy" else arg
            for arg in sys.argv
        ]
    elif (
        "--eval_strategy" in sys.argv
        and "eval_strategy" not in training_arg_fields
        and "evaluation_strategy" in training_arg_fields
    ):
        sys.argv = [
            "--evaluation_strategy" if arg == "--eval_strategy" else arg
            for arg in sys.argv
        ]

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    if training_args.max_steps > 0:
        warmup_steps = training_args.get_warmup_steps(training_args.max_steps)
        if warmup_steps >= training_args.max_steps:
            if training_args.local_rank in (0, -1):
                print(
                    f"[warn] warmup_steps ({warmup_steps}) >= max_steps ({training_args.max_steps}). "
                    "All optimization steps may run with zero LR. Auto-setting warmup_steps=0, warmup_ratio=0.0."
                )
            training_args.warmup_steps = 0
            training_args.warmup_ratio = 0.0
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
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
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            config = LlavaConfig.from_pretrained(model_args.model_name_or_path)
            config.mm_vision_tower = model_args.vision_tower
            base_checkpoint_has_gate = checkpoint_has_gate_weights(model_args.model_name_or_path)
            config.use_dual_input_gate = False
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
            model.get_model().use_dual_input_gate = base_checkpoint_has_gate
            model.config.use_dual_input_gate = base_checkpoint_has_gate
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

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
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
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
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        model.get_model().use_dual_input_gate = bool(model_args.use_dual_input_gate)
        model.config.use_dual_input_gate = bool(model_args.use_dual_input_gate)
        if model_args.use_dual_input_gate:
            gate_module = model.get_model().ensure_dual_input_gate()
            projector_param = next(model.get_model().mm_projector.parameters())
            gate_module.to(dtype=projector_param.dtype, device=projector_param.device)
        else:
            gate_module = getattr(model.get_model(), "gate", None)
            if gate_module is not None:
                gate_module.requires_grad_(False)
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
            gate_module = getattr(model.get_model(), "gate", None)
            if gate_module is not None:
                for p in gate_module.parameters():
                    p.requires_grad = True


        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False
            gate_module = getattr(model.get_model(), "gate", None)
            if gate_module is not None:
                for p in gate_module.parameters():
                    p.requires_grad = False
        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.mask_patch_loss_weight = training_args.mask_patch_loss_weight
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

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
    clip_tokenizer = CLIPTokenizer.from_pretrained(model_args.vision_tower)
    data_module = make_supervised_data_module(clip_tokenizer=clip_tokenizer,tokenizer=tokenizer,
                                              data_args=data_args)
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)
    init_weight_to_save = None
    if os.environ.get("SAVE_INIT_TRAINABLES", "0") == "1":
        init_keys_to_match = ['mm_projector', 'model.gate.']
        init_weight_to_save = get_mm_adapter_state_maybe_zero_3(
            model.named_parameters(), init_keys_to_match
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            os.makedirs(training_args.output_dir, exist_ok=True)
            torch.save(init_weight_to_save, os.path.join(training_args.output_dir, 'init_trainables.bin'))
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"✅ [trainable] name: {name} | shape: {list(param.shape)} | dtype: {param.dtype}")
    # Only let rank0 initialize/watch wandb, otherwise multi-GPU launches multiple runs.
    if (training_args.local_rank in (0, -1)) and ("wandb" in training_args.report_to):
        import wandb
        if wandb.run is None:
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", getattr(training_args, 'wandb_project', 'huggingface')),
                name=os.environ.get("WANDB_NAME", training_args.output_dir.split('/')[-1]),
                config=vars(training_args)
            )
        gate_module = model.get_model().gate if hasattr(model, "get_model") else None
        if gate_module is not None:
            wandb.watch(gate_module, log="gradients", log_freq=100)
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    if os.environ.get("DEBUG_DELTA_AFTER_TRAIN", "0") == "1" and init_weight_to_save is not None:
        cur_keys_to_match = ['mm_projector', 'model.gate.']
        cur_weight_to_save = get_mm_adapter_state_maybe_zero_3(
            model.named_parameters(), cur_keys_to_match
        )
        total_changed = 0
        total_numel = 0
        for k in sorted(set(init_weight_to_save.keys()) & set(cur_weight_to_save.keys())):
            d = (cur_weight_to_save[k].float() - init_weight_to_save[k].float()).abs()
            changed = int((d > 0).sum().item())
            total_changed += changed
            total_numel += int(d.numel())
            if training_args.local_rank in (0, -1):
                print(
                    f"[debug-delta] {k}: changed={changed}/{d.numel()} "
                    f"max_abs={float(d.max().item())} mean_abs={float(d.mean().item())}"
                )
        if training_args.local_rank in (0, -1):
            ratio = (total_changed / total_numel) if total_numel > 0 else 0.0
            print(f"[debug-delta] total_changed={total_changed}/{total_numel} ratio={ratio}")
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
