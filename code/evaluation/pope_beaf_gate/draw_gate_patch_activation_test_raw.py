#!/usr/bin/env python3
import argparse
import json
import math
from collections import defaultdict
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
    "train",
    "platform",
    "skateboarder",
    "skateboard",
    "bike",
    "bicycle",
    "bikes",
    "bottle",
    "bottles",
    "counter",
    "cap",
    "caps",
    "car",
    "truck",
    "bus",
    "boat",
    "airplane",
    "horse",
    "dog",
    "cat",
    "bird",
    "bench",
    "chair",
    "table",
    "couch",
    "bed",
    "sink",
    "toilet",
    "tv",
    "laptop",
    "phone",
    "book",
    "clock",
    "vase",
    "umbrella",
    "backpack",
    "bag",
    "suitcase",
    "ball",
    "bat",
    "glove",
    "frisbee",
    "skis",
    "snowboard",
    "surfboard",
    "racket",
    "kite",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "cup",
    "glass",
    "pizza",
    "sandwich",
    "cake",
    "banana",
    "apple",
    "orange",
    "broccoli",
    "carrot",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw gate activation heatmaps from test_raw result rows.")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument(
        "--result-path",
        type=Path,
        default=Path("/path/to/sage_repro_bundle/infer_result_test_raw_nonumbermaskloss_epoch3/checkpoint-5148/test_raw_with_shortcut_answer.json"),
    )
    parser.add_argument(
        "--fallback-data-path",
        type=Path,
        default=Path("/path/to/sage_repro_bundle/test_data/test_raw_with_shortcut_answer.json"),
    )
    parser.add_argument("--image-folder", type=Path, default=Path("data/images/coco/train2014"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/path/to/sage_repro_bundle/analysis/gate_patch_activation_test_raw_nonumbermaskloss_20260430"),
    )
    parser.add_argument("--num-images", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--prefer", choices=["mixed", "shortcut", "correct", "wrong"], default="mixed")
    return parser.parse_args()


def load_rows(result_path: Path, fallback_data_path: Path) -> List[dict]:
    path = result_path if result_path.exists() else fallback_data_path
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Expected JSON array in {path}")
    return rows


def norm_text(value) -> str:
    return str(value if value is not None else "").strip().lower()


def object_hits(question: str) -> List[str]:
    text = question.lower()
    return sorted({word for word in OBJECT_WORDS if word in text})


def image_path_for(row: dict, image_folder: Path) -> Path:
    if row.get("image"):
        image_name = str(row["image"])
    else:
        image_name = f"COCO_train2014_{int(row['image_id']):012d}.jpg"
    path = image_folder / image_name
    if path.exists():
        return path
    alt = Path("/path/to/sage_repro_bundle/sam3/train2014") / image_name
    if alt.exists():
        return alt
    return path


def build_candidates(rows: List[dict], image_folder: Path, prefer: str) -> List[dict]:
    grouped: Dict[int, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["image_id"])].append(row)

    candidates = []
    for image_id, image_rows in grouped.items():
        image_path = image_path_for(image_rows[0], image_folder)
        if not image_path.exists():
            continue
        hits = set()
        type_counts = defaultdict(int)
        correct_count = 0
        shortcut_count = 0
        wrong_count = 0
        for row in image_rows:
            hits.update(object_hits(row.get("question", "")))
            type_counts[str(row.get("answer_type", ""))] += 1
            pred = norm_text(row.get("model_pred"))
            ans = norm_text(row.get("answer"))
            shortcut = norm_text(row.get("shortcut_answer"))
            if pred and pred == ans:
                correct_count += 1
            elif pred and pred == shortcut:
                shortcut_count += 1
            elif pred:
                wrong_count += 1
        ranked_rows = sorted(
            image_rows,
            key=lambda row: (
                len(object_hits(row.get("question", ""))),
                norm_text(row.get("model_pred")) == norm_text(row.get("shortcut_answer")),
                norm_text(row.get("model_pred")) == norm_text(row.get("answer")),
            ),
            reverse=True,
        )
        selected_row = ranked_rows[0]
        prefer_score = {
            "mixed": correct_count + shortcut_count + wrong_count,
            "shortcut": shortcut_count,
            "correct": correct_count,
            "wrong": wrong_count,
        }[prefer]
        candidates.append(
            {
                "image_id": image_id,
                "image_path": str(image_path),
                "question_count": len(image_rows),
                "object_hits": sorted(hits),
                "answer_type_counts": dict(type_counts),
                "correct_count": correct_count,
                "shortcut_count": shortcut_count,
                "wrong_count": wrong_count,
                "prefer_score": prefer_score,
                "file_size": image_path.stat().st_size,
                "row": selected_row,
            }
        )

    candidates.sort(
        key=lambda item: (
            item["prefer_score"],
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
    return (
        torch.tensor([(input_ids + [0] * 77)[:77]], dtype=torch.long, device="cuda"),
        torch.tensor([(attn_mask + [0] * 77)[:77]], dtype=torch.long, device="cuda"),
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
    return Image.fromarray(np.stack([red, green, blue, alpha], axis=-1), mode="RGBA")


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

    resample_nearest = getattr(Image.Resampling, "NEAREST", Image.NEAREST)
    resample_bilinear = getattr(Image.Resampling, "BILINEAR", Image.BILINEAR)
    heat_small = colorize_heatmap(norm)
    heat = heat_small.resize(image.size, resample=resample_bilinear)
    overlay = Image.alpha_composite(image.convert("RGBA"), heat).convert("RGB")
    heat_rgb = heat.convert("RGB")

    image.save(out_prefix.with_suffix(".original.jpg"), quality=92)
    heat_small.resize((grid_w * 24, grid_h * 24), resample=resample_nearest).convert("RGB").save(
        out_prefix.with_suffix(".heatmap_grid.png")
    )
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.result_path, args.fallback_data_path)
    candidates = build_candidates(rows, args.image_folder, args.prefer)
    if not candidates:
        raise RuntimeError("No test_raw candidate images found")
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
        row = item["row"]
        image_path = Path(item["image_path"])
        image = Image.open(image_path).convert("RGB")
        question = row["question"]
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
        regenerated_pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        activation_tensor = gate_module.current_gate_patch_activation
        if activation_tensor is None:
            raise RuntimeError(f"No gate patch activation captured for {image_path}")
        activation = activation_tensor.detach().float().cpu().numpy()[0]

        prefix = args.output_dir / f"{idx:02d}_{image_path.stem}_qid{int(row['question_id'])}"
        objects = ", ".join(item["object_hits"][:16]) if item["object_hits"] else "n/a"
        caption = [
            f"{image_path.name} qid={row['question_id']} image_questions={item['question_count']} objects={objects}",
            f"Q: {question}",
            f"GT: {row.get('answer')} | shortcut: {row.get('shortcut_answer')} | stored_pred: {row.get('model_pred')} | regen: {regenerated_pred}",
        ]
        stats = save_visuals(image, activation, prefix, caption)
        out_row = {
            **{k: v for k, v in item.items() if k != "row"},
            "question_id": int(row["question_id"]),
            "question": question,
            "answer": row.get("answer"),
            "shortcut_answer": row.get("shortcut_answer"),
            "answer_type": row.get("answer_type"),
            "stored_model_pred": row.get("model_pred"),
            "regenerated_pred": regenerated_pred,
            **stats,
            "files": {
                "original": str(prefix.with_suffix(".original.jpg")),
                "overlay": str(prefix.with_suffix(".overlay.png")),
                "heatmap": str(prefix.with_suffix(".heatmap.png")),
                "heatmap_grid": str(prefix.with_suffix(".heatmap_grid.png")),
                "activation": str(prefix.with_suffix(".activation.npy")),
            },
        }
        metadata.append(out_row)
        print(
            f"[done] {idx + 1}/{len(selected)} qid={row['question_id']} "
            f"mean={stats['activation_mean']:.6f}",
            flush=True,
        )

    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[done] test_raw gate visualizations: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
