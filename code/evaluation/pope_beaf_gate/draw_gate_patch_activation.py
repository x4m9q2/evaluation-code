#!/usr/bin/env python3
import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPTokenizer

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


OBJECT_WORDS = {
    "person",
    "people",
    "man",
    "woman",
    "child",
    "boy",
    "girl",
    "car",
    "truck",
    "bus",
    "train",
    "bicycle",
    "motorcycle",
    "boat",
    "airplane",
    "traffic light",
    "stop sign",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw dual-input gate patch activation heatmaps.")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--pope-question-file", type=Path, default=Path("/path/to/sage_repro_bundle/playground/data/eval/pope/llava_pope_test.jsonl"))
    parser.add_argument("--pope-image-folder", type=Path, default=Path("/path/to/sage_repro_bundle/playground/data/eval/pope/val2014"))
    parser.add_argument("--beaf-qna-path", type=Path, default=Path("/path/to/sage_repro_bundle/playground/data/eval/beaf/BEAF_downloads/beaf_qna.json"))
    parser.add_argument("--beaf-image-folder", type=Path, default=Path("/path/to/sage_repro_bundle/playground/data/eval/beaf/beaf_dataset_ver1"))
    parser.add_argument("--output-dir", type=Path, default=Path("/path/to/sage_repro_bundle/analysis/gate_patch_activation_nonumbermaskloss_20260430"))
    parser.add_argument("--num-images", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json_array(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Expected JSON array: {path}")
    return rows


def object_hits(question: str) -> List[str]:
    text = question.lower()
    return sorted({word for word in OBJECT_WORDS if word in text})


def build_candidates(args: argparse.Namespace) -> List[dict]:
    by_image: Dict[str, dict] = {}

    if args.pope_question_file.exists():
        pope_rows = load_jsonl(args.pope_question_file)
        grouped = defaultdict(list)
        for row in pope_rows:
            grouped[row["image"]].append(row)
        for image_name, rows in grouped.items():
            path = args.pope_image_folder / image_name
            if not path.exists():
                continue
            hits = set()
            for row in rows:
                hits.update(object_hits(row.get("text", "")))
            by_image[str(path)] = {
                "source": "pope",
                "image_path": str(path),
                "image_name": image_name,
                "question": "Describe the main objects and their layout in this image.",
                "question_count": len(rows),
                "object_hits": sorted(hits),
                "file_size": path.stat().st_size,
            }

    if args.beaf_qna_path.exists():
        beaf_rows = load_json_array(args.beaf_qna_path)
        grouped = defaultdict(list)
        for row in beaf_rows:
            if row.get("orig_img") is True:
                grouped[row["image"]].append(row)
        for image_name, rows in grouped.items():
            path = args.beaf_image_folder / image_name
            if not path.exists():
                continue
            hits = set()
            for row in rows:
                hits.update(object_hits(row.get("question", "")))
            current = by_image.get(str(path))
            payload = {
                "source": "beaf",
                "image_path": str(path),
                "image_name": image_name,
                "question": "Describe the main objects and their layout in this image.",
                "question_count": len(rows),
                "object_hits": sorted(hits),
                "file_size": path.stat().st_size,
            }
            if current is None or (len(payload["object_hits"]), payload["question_count"], payload["file_size"]) > (
                len(current["object_hits"]),
                current["question_count"],
                current["file_size"],
            ):
                by_image[str(path)] = payload

    candidates = list(by_image.values())
    candidates.sort(
        key=lambda item: (
            len(item["object_hits"]),
            item["question_count"],
            item["file_size"],
        ),
        reverse=True,
    )
    return candidates


def clip_prompt_ids(clip_tokenizer: CLIPTokenizer, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
    clip_ids = clip_tokenizer(prompt, truncation=True, max_length=77)
    input_ids = clip_ids["input_ids"]
    attn_mask = clip_ids["attention_mask"]
    eos_id = clip_tokenizer.eos_token_id
    if eos_id is not None and eos_id not in input_ids:
        if len(input_ids) >= 77:
            input_ids[-1] = eos_id
        else:
            input_ids.append(eos_id)
            attn_mask.append(1)
    input_ids = (input_ids + [0] * 77)[:77]
    attn_mask = (attn_mask + [0] * 77)[:77]
    return (
        torch.tensor([input_ids], dtype=torch.long, device="cuda"),
        torch.tensor([attn_mask], dtype=torch.long, device="cuda"),
    )


def infer_grid(values: np.ndarray) -> tuple[int, int]:
    length = int(values.shape[0])
    side = int(round(math.sqrt(length)))
    if side * side == length:
        return side, side
    for height in range(side, 0, -1):
        if length % height == 0:
            return height, length // height
    return 1, length


def colorize_heatmap(norm: np.ndarray) -> Image.Image:
    norm = np.clip(norm, 0.0, 1.0)
    red = (255 * norm).astype(np.uint8)
    green = (255 * (1.0 - np.abs(norm - 0.55) / 0.55).clip(0.0, 1.0)).astype(np.uint8)
    blue = (255 * (1.0 - norm)).astype(np.uint8)
    alpha = np.full(norm.shape, 190, dtype=np.uint8)
    rgba = np.stack([red, green, blue, alpha], axis=-1)
    return Image.fromarray(rgba, mode="RGBA")


def draw_caption(image: Image.Image, lines: List[str]) -> Image.Image:
    width, height = image.size
    pad = 12
    line_height = 18
    caption_h = pad * 2 + line_height * len(lines)
    canvas = Image.new("RGB", (width, height + caption_h), (255, 255, 255))
    canvas.paste(image.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    y = height + pad
    for line in lines:
        draw.text((pad, y), line[:180], fill=(20, 20, 20), font=font)
        y += line_height
    return canvas


def save_visuals(image: Image.Image, activation: np.ndarray, out_prefix: Path, caption_lines: List[str]) -> dict:
    grid_h, grid_w = infer_grid(activation)
    grid = activation.reshape(grid_h, grid_w)
    min_v = float(grid.min())
    max_v = float(grid.max())
    mean_v = float(grid.mean())
    std_v = float(grid.std())
    norm = (grid - min_v) / (max_v - min_v + 1e-8)

    heat_small = colorize_heatmap(norm)
    resample_nearest = getattr(Image.Resampling, "NEAREST", Image.NEAREST)
    resample_bilinear = getattr(Image.Resampling, "BILINEAR", Image.BILINEAR)
    heat = heat_small.resize(image.size, resample=resample_bilinear)
    overlay = Image.alpha_composite(image.convert("RGBA"), heat).convert("RGB")
    heat_rgb = heat.convert("RGB")

    image.save(out_prefix.with_suffix(".original.jpg"), quality=92)
    heat_grid = heat_small.resize((grid_w * 24, grid_h * 24), resample=resample_nearest).convert("RGB")
    heat_grid.save(out_prefix.with_suffix(".heatmap_grid.png"))
    draw_caption(heat_rgb, caption_lines).save(out_prefix.with_suffix(".heatmap.png"))
    draw_caption(overlay, caption_lines).save(out_prefix.with_suffix(".overlay.png"))
    np.save(out_prefix.with_suffix(".activation.npy"), grid.astype(np.float32))

    return {
        "grid_shape": [grid_h, grid_w],
        "activation_min": min_v,
        "activation_max": max_v,
        "activation_mean": mean_v,
        "activation_std": std_v,
    }


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates = build_candidates(args)
    if not candidates:
        raise RuntimeError("No candidate images found for gate visualization")
    selected = candidates[: args.num_images]
    (args.output_dir / "selected_candidates.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    disable_torch_init()
    model_name = get_model_name_from_path(str(args.model_path))
    tokenizer, model, image_processor, _ = load_pretrained_model(
        str(args.model_path),
        None,
        model_name,
        device_map=None,
        device="cuda",
        force_use_dual_input_gate=True,
    )
    tokenizer.padding_side = "left"
    model.eval()
    model.get_model().runtime_patch_gate_suppress_ratio = 0.0
    clip_tokenizer = CLIPTokenizer.from_pretrained(model.config.mm_vision_tower)

    metadata = []
    for idx, item in enumerate(selected):
        image_path = Path(item["image_path"])
        image = Image.open(image_path).convert("RGB")
        question = item["question"]
        qs = DEFAULT_IMAGE_TOKEN + "\n" + question
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to("cuda")
        attention_mask = torch.ones_like(input_ids, device="cuda")
        image_tensor = process_images([image], image_processor, model.config)[0].unsqueeze(0).to(dtype=torch.float16, device="cuda")
        clip_input_ids, clip_attn_mask = clip_prompt_ids(clip_tokenizer, question)

        gate_module = getattr(model.get_model(), "gate", None)
        if gate_module is None:
            raise RuntimeError("Loaded model does not expose model.gate")
        gate_module.current_gate_patch_activation = None

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                images=image_tensor,
                image_sizes=[image.size],
                clip_input_ids=clip_input_ids,
                clip_attn_mask=clip_attn_mask,
                do_sample=False,
                temperature=0.0,
                num_beams=1,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
        answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        activation_tensor = gate_module.current_gate_patch_activation
        if activation_tensor is None:
            raise RuntimeError(f"No gate patch activation captured for {image_path}")
        activation = activation_tensor.detach().float().cpu().numpy()[0]

        prefix = args.output_dir / f"{idx:02d}_{image_path.stem}"
        object_summary = ", ".join(item["object_hits"][:16]) if item["object_hits"] else "n/a"
        caption = [
            f"{image_path.name} source={item['source']} q_count={item['question_count']} objects={object_summary}",
            f"Q: {question}",
            f"A: {answer}",
        ]
        stats = save_visuals(image, activation, prefix, caption)
        row = {
            **item,
            "question": question,
            "answer": answer,
            **stats,
            "files": {
                "original": str(prefix.with_suffix(".original.jpg")),
                "overlay": str(prefix.with_suffix(".overlay.png")),
                "heatmap": str(prefix.with_suffix(".heatmap.png")),
                "heatmap_grid": str(prefix.with_suffix(".heatmap_grid.png")),
                "activation": str(prefix.with_suffix(".activation.npy")),
            },
        }
        metadata.append(row)
        print(f"[done] {idx + 1}/{len(selected)} {image_path.name} mean={stats['activation_mean']:.6f}", flush=True)

    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[done] gate visualizations: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
