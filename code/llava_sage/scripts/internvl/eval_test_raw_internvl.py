#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run InternVL2.5-8B on /path/to/sage_repro_bundle/test_raw.json-style VQA data."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--question-file", required=True, help="JSON or JSONL with question_id/image_id/question fields.")
    parser.add_argument("--image-folder", required=True, help="COCO image root, e.g. data/images/coco/train2014")
    parser.add_argument("--answers-file", required=True, help="Output JSONL prediction path.")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--max-num", type=int, default=12, help="Max dynamic image tiles.")
    parser.add_argument("--use-flash-attn", action="store_true")
    parser.add_argument("--prompt-suffix", default="", help="Optional suffix appended after each question.")
    return parser.parse_args()


def load_json_or_jsonl(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        raise ValueError(f"Expected list JSON in {path}")
    except json.JSONDecodeError:
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows


def split_list(lst: list[dict], n: int) -> list[list[dict]]:
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst: list[dict], n: int, k: int) -> list[dict]:
    chunks = split_list(lst, n)
    if k < 0 or k >= len(chunks):
        raise IndexError(f"chunk_idx {k} out of range for {len(chunks)} chunks")
    return chunks[k]


def build_transform(input_size: int):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> list[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))

    return processed_images


def load_image(image_file: Path, input_size: int = 448, max_num: int = 12) -> torch.Tensor:
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


class TestRawDataset(Dataset):
    def __init__(
        self,
        questions: list[dict],
        image_folder: Path,
        input_size: int,
        max_num: int,
        prompt_suffix: str,
    ):
        self.questions = questions
        self.image_folder = image_folder
        self.input_size = input_size
        self.max_num = max_num
        self.prompt_suffix = prompt_suffix.strip()

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, index: int) -> dict:
        item = self.questions[index]
        image_id = int(item["image_id"])
        image_path = self.image_folder / f"COCO_train2014_{image_id:012d}.jpg"
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        question = str(item["question"]).strip()
        prompt = f"<image>\n{question}"
        if self.prompt_suffix:
            prompt = f"{prompt}\n{self.prompt_suffix}"

        pixel_values = load_image(image_path, input_size=self.input_size, max_num=self.max_num)
        return {
            "question_id": int(item["question_id"]),
            "prompt": question,
            "chat_prompt": prompt,
            "pixel_values": pixel_values,
        }


def collate_fn(batch: list[dict]) -> dict:
    return {
        "question_ids": [item["question_id"] for item in batch],
        "prompts": [item["prompt"] for item in batch],
        "chat_prompts": [item["chat_prompt"] for item in batch],
        "pixel_values": torch.cat([item["pixel_values"] for item in batch], dim=0),
        "num_patches_list": [item["pixel_values"].shape[0] for item in batch],
    }


def slice_batch(batch: dict, start: int, end: int) -> dict:
    patch_start = sum(batch["num_patches_list"][:start])
    patch_end = sum(batch["num_patches_list"][:end])
    return {
        "question_ids": batch["question_ids"][start:end],
        "prompts": batch["prompts"][start:end],
        "chat_prompts": batch["chat_prompts"][start:end],
        "pixel_values": batch["pixel_values"][patch_start:patch_end],
        "num_patches_list": batch["num_patches_list"][start:end],
    }


def is_oom_error(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "CUDA out of memory" in str(exc)


def cleanup_cuda_oom() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def run_batch_chat_once(
    model,
    tokenizer,
    batch: dict,
    generation_config: dict,
) -> list[str]:
    pixel_values = batch["pixel_values"].to(device="cuda", dtype=torch.bfloat16, non_blocking=True)
    try:
        with torch.inference_mode():
            return model.batch_chat(
                tokenizer,
                pixel_values,
                batch["chat_prompts"],
                generation_config,
                num_patches_list=batch["num_patches_list"],
                history=None,
                return_history=False,
            )
    finally:
        del pixel_values


def build_single_example_batch_with_cap(batch: dict, patch_cap: int) -> dict:
    assert len(batch["question_ids"]) == 1
    pixel_values = batch["pixel_values"]
    total_patches = int(batch["num_patches_list"][0])
    if patch_cap >= total_patches:
        capped_pixel_values = pixel_values
    elif patch_cap <= 1:
        capped_pixel_values = pixel_values[-1:].clone()
    else:
        # Keep a few spatial tiles plus the thumbnail tile for global context.
        capped_pixel_values = torch.cat([pixel_values[: patch_cap - 1], pixel_values[-1:]], dim=0)

    return {
        "question_ids": batch["question_ids"],
        "prompts": batch["prompts"],
        "chat_prompts": batch["chat_prompts"],
        "pixel_values": capped_pixel_values,
        "num_patches_list": [int(capped_pixel_values.shape[0])],
    }


def run_single_example_with_patch_fallback(
    model,
    tokenizer,
    batch: dict,
    generation_config: dict,
    depth: int,
) -> list[str]:
    total_patches = int(batch["num_patches_list"][0])
    patch_caps = []
    for cap in [total_patches, 8, 6, 4, 2, 1]:
        cap = min(total_patches, cap)
        if cap not in patch_caps:
            patch_caps.append(cap)

    last_oom_message = "unknown CUDA OOM"
    indent = "  " * depth
    question_id = batch["question_ids"][0]
    for patch_cap in patch_caps:
        candidate_batch = build_single_example_batch_with_cap(batch, patch_cap)
        if patch_cap != total_patches:
            print(
                f"{indent}Retrying question_id={question_id} with {patch_cap} patch(es) "
                f"(original={total_patches})",
                flush=True,
            )
        try:
            return run_batch_chat_once(model, tokenizer, candidate_batch, generation_config)
        except RuntimeError as exc:
            if not is_oom_error(exc):
                raise
            last_oom_message = str(exc)
        cleanup_cuda_oom()

    raise RuntimeError(
        f"CUDA OOM persisted for question_id={question_id} after retrying patch caps "
        f"{patch_caps}: {last_oom_message}"
    )


def run_batch_chat_with_retry(
    model,
    tokenizer,
    batch: dict,
    generation_config: dict,
    depth: int = 0,
) -> list[str]:
    indent = "  " * depth
    try:
        return run_batch_chat_once(model, tokenizer, batch, generation_config)
    except RuntimeError as exc:
        if not is_oom_error(exc):
            raise
    cleanup_cuda_oom()
    if len(batch["question_ids"]) <= 1:
        return run_single_example_with_patch_fallback(
            model, tokenizer, batch, generation_config, depth=depth + 1
        )

    split_point = len(batch["question_ids"]) // 2
    print(
        f"{indent}OOM at batch size {len(batch['question_ids'])}, splitting into "
        f"{split_point} + {len(batch['question_ids']) - split_point}",
        flush=True,
    )
    left_batch = slice_batch(batch, 0, split_point)
    right_batch = slice_batch(batch, split_point, len(batch["question_ids"]))
    left_responses = run_batch_chat_with_retry(model, tokenizer, left_batch, generation_config, depth + 1)
    cleanup_cuda_oom()
    right_responses = run_batch_chat_with_retry(model, tokenizer, right_batch, generation_config, depth + 1)
    return left_responses + right_responses


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    question_file = Path(args.question_file)
    questions = load_json_or_jsonl(question_file)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=args.use_flash_attn,
        trust_remote_code=True,
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)

    dataset = TestRawDataset(
        questions=questions,
        image_folder=Path(args.image_folder),
        input_size=args.input_size,
        max_num=args.max_num,
        prompt_suffix=args.prompt_suffix,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    generation_config = {
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "top_p": args.top_p,
    }
    if args.temperature > 0:
        generation_config["temperature"] = args.temperature

    answers_path = Path(args.answers_file)
    answers_path.parent.mkdir(parents=True, exist_ok=True)

    with answers_path.open("w", encoding="utf-8") as f:
        for batch in tqdm(dataloader, total=len(dataloader), desc=f"chunk{args.chunk_idx}"):
            responses = run_batch_chat_with_retry(model, tokenizer, batch, generation_config)

            for question_id, prompt, response in zip(batch["question_ids"], batch["prompts"], responses):
                row = {
                    "question_id": question_id,
                    "prompt": prompt,
                    "text": str(response).strip(),
                    "model_id": os.path.basename(os.path.abspath(args.model_path.rstrip("/"))),
                    "metadata": {},
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()


if __name__ == "__main__":
    main()
