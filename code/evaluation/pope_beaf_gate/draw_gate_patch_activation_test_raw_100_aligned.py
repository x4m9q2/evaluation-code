#!/usr/bin/env python3
import argparse
import csv
import html
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image
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
    "bike",
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
    "bag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "ball",
    "kite",
    "baseball bat",
    "bat",
    "baseball glove",
    "glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "racket",
    "bottle",
    "wine glass",
    "glass",
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
    "table",
    "dining table",
    "potted plant",
    "bed",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "phone",
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
    "platform",
    "counter",
    "cap",
}

OBJECT_PATTERNS = {
    word: re.compile(r"(?<![a-z0-9])" + re.escape(word) + r"(?![a-z0-9])")
    for word in sorted(OBJECT_WORDS, key=len, reverse=True)
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw 100 aligned dual-input gate patch activation heatmaps for test_raw rows."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            "/path/to/sage_repro_bundle/checkpoints/"
            "finetune_stage2_qwenratio_oldbase_sam3_qwenkeepmask_nonumbermaskloss_full_bs32_20260425_175907/"
            "checkpoint-5148"
        ),
    )
    parser.add_argument(
        "--result-path",
        type=Path,
        default=Path("/path/to/sage_repro_bundle/infer_result/checkpoint-5148/test_raw_with_shortcut_answer.json"),
    )
    parser.add_argument(
        "--fallback-data-path",
        type=Path,
        default=Path("/path/to/sage_repro_bundle/test_data/test_raw_with_shortcut_answer.json"),
    )
    parser.add_argument("--image-folder", type=Path, default=Path("/root/train2014"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/path/to/sage_repro_bundle/analysis/gate_patch_activation_test_raw_nonumbermaskloss_100_aligned_20260430"),
    )
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--prefer", choices=["mixed", "shortcut", "correct", "wrong"], default="mixed")
    parser.add_argument("--overlay-alpha", type=int, default=170)
    parser.add_argument("--html-thumb-width", type=int, default=250)
    return parser.parse_args()


def load_rows(result_path: Path, fallback_data_path: Path) -> Tuple[List[dict], Path]:
    path = result_path if result_path.exists() else fallback_data_path
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Expected JSON array in {path}")
    return rows, path


def norm_text(value) -> str:
    return str(value if value is not None else "").strip().lower()


def object_hits(question: str) -> List[str]:
    text = question.lower()
    return sorted({word for word, pattern in OBJECT_PATTERNS.items() if pattern.search(text)})


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


def result_bucket(row: dict) -> str:
    pred = norm_text(row.get("model_pred"))
    answer = norm_text(row.get("answer"))
    shortcut = norm_text(row.get("shortcut_answer"))
    if pred and pred == answer:
        return "correct"
    if pred and pred == shortcut:
        return "shortcut"
    if pred:
        return "wrong"
    return "empty"


def row_rank(row: dict, prefer: str) -> tuple:
    bucket = result_bucket(row)
    prefer_score = 1 if bucket == prefer else 0
    if prefer == "mixed":
        prefer_score = 1 if bucket != "empty" else 0
    return (
        prefer_score,
        len(object_hits(row.get("question", ""))),
        str(row.get("answer_type", "")) != "number",
        int(row.get("question_id", 0)),
    )


def build_candidates(rows: List[dict], image_folder: Path, prefer: str) -> List[dict]:
    grouped: Dict[int, List[dict]] = defaultdict(list)
    for row in rows:
        if "image_id" not in row:
            continue
        grouped[int(row["image_id"])].append(row)

    candidates = []
    for image_id, image_rows in grouped.items():
        image_path = image_path_for(image_rows[0], image_folder)
        if not image_path.exists():
            continue

        hits = set()
        type_counts = defaultdict(int)
        bucket_counts = defaultdict(int)
        for row in image_rows:
            hits.update(object_hits(row.get("question", "")))
            type_counts[str(row.get("answer_type", ""))] += 1
            bucket_counts[result_bucket(row)] += 1

        selected_row = max(image_rows, key=lambda row: row_rank(row, prefer))
        prefer_score = (
            sum(bucket_counts.values()) if prefer == "mixed" else bucket_counts.get(prefer, 0)
        )
        candidates.append(
            {
                "image_id": image_id,
                "image_path": str(image_path),
                "question_count": len(image_rows),
                "object_hits": sorted(hits),
                "answer_type_counts": dict(type_counts),
                "bucket_counts": dict(bucket_counts),
                "prefer_score": int(prefer_score),
                "file_size": image_path.stat().st_size,
                "row": selected_row,
            }
        )

    candidates.sort(
        key=lambda item: (
            len(item["object_hits"]),
            item["question_count"],
            item["prefer_score"],
            item["file_size"],
        ),
        reverse=True,
    )
    return candidates


def clip_prompt_ids(clip_tokenizer: CLIPTokenizer, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
    clip_ids = clip_tokenizer(prompt, truncation=True, max_length=77)
    input_ids = list(clip_ids["input_ids"])
    attn_mask = list(clip_ids["attention_mask"])
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


def normalize_grid(grid: np.ndarray) -> Tuple[np.ndarray, dict]:
    min_v = float(grid.min())
    max_v = float(grid.max())
    mean_v = float(grid.mean())
    std_v = float(grid.std())
    norm = (grid - min_v) / (max_v - min_v + 1e-8)
    return np.clip(norm, 0.0, 1.0), {
        "activation_min": min_v,
        "activation_max": max_v,
        "activation_mean": mean_v,
        "activation_std": std_v,
        "normalization": "per_image_minmax",
    }


def colorize_heatmap(norm: np.ndarray, alpha: int) -> Image.Image:
    norm = np.clip(norm, 0.0, 1.0)
    red = (255 * norm).astype(np.uint8)
    green = (255 * (1.0 - np.abs(norm - 0.55) / 0.55).clip(0.0, 1.0)).astype(np.uint8)
    blue = (255 * (1.0 - norm)).astype(np.uint8)
    alpha_arr = np.full(norm.shape, int(np.clip(alpha, 0, 255)), dtype=np.uint8)
    return Image.fromarray(np.stack([red, green, blue, alpha_arr], axis=-1), mode="RGBA")


def pad_square_mapping(size: Tuple[int, int]) -> dict:
    width, height = size
    side = max(width, height)
    pad_left = (side - width) // 2
    pad_top = (side - height) // 2
    pad_right = side - width - pad_left
    pad_bottom = side - height - pad_top
    return {
        "original_width": width,
        "original_height": height,
        "padded_square_side": side,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "pad_right": pad_right,
        "pad_bottom": pad_bottom,
        "crop_box_on_padded_square": [pad_left, pad_top, pad_left + width, pad_top + height],
    }


def heatmap_on_original(
    heat_small_rgba: Image.Image,
    original_size: Tuple[int, int],
    image_aspect_ratio: str | None,
) -> Tuple[Image.Image, dict]:
    resample_bilinear = getattr(Image.Resampling, "BILINEAR", Image.BILINEAR)
    if image_aspect_ratio == "pad":
        mapping = pad_square_mapping(original_size)
        side = int(mapping["padded_square_side"])
        heat_padded = heat_small_rgba.resize((side, side), resample=resample_bilinear)
        heat = heat_padded.crop(tuple(mapping["crop_box_on_padded_square"]))
        mapping["alignment_method"] = (
            "LLaVA pad mode: upsample 24x24 gate grid to the preprocessor padded square, "
            "then crop out the original image region."
        )
    else:
        heat = heat_small_rgba.resize(original_size, resample=resample_bilinear)
        mapping = {
            "original_width": original_size[0],
            "original_height": original_size[1],
            "alignment_method": "Direct resize because model image_aspect_ratio is not pad.",
        }
    mapping["output_size_matches_original"] = list(heat.size) == list(original_size)
    return heat, mapping


def ensure_dirs(output_dir: Path) -> dict:
    paths = {
        "originals": output_dir / "originals",
        "overlays": output_dir / "overlays",
        "heatmaps": output_dir / "heatmaps",
        "activations": output_dir / "activations",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_visuals(
    image: Image.Image,
    activation: np.ndarray,
    file_prefix: str,
    output_paths: dict,
    image_aspect_ratio: str | None,
    overlay_alpha: int,
) -> dict:
    grid_h, grid_w = infer_grid(activation)
    grid = activation.reshape(grid_h, grid_w)
    norm, stats = normalize_grid(grid)

    heat_small = colorize_heatmap(norm, overlay_alpha)
    heat_rgba, mapping = heatmap_on_original(heat_small, image.size, image_aspect_ratio)
    overlay = Image.alpha_composite(image.convert("RGBA"), heat_rgba).convert("RGB")
    heat_rgb = heat_rgba.convert("RGB")

    original_path = output_paths["originals"] / f"{file_prefix}.jpg"
    overlay_path = output_paths["overlays"] / f"{file_prefix}.png"
    heatmap_path = output_paths["heatmaps"] / f"{file_prefix}.png"
    activation_path = output_paths["activations"] / f"{file_prefix}.npy"

    image.save(original_path, quality=95)
    overlay.save(overlay_path)
    heat_rgb.save(heatmap_path)
    np.save(activation_path, grid.astype(np.float32))

    return {
        "grid_shape": [grid_h, grid_w],
        **stats,
        "alignment": mapping,
        "files": {
            "original": str(original_path),
            "overlay": str(overlay_path),
            "heatmap": str(heatmap_path),
            "activation": str(activation_path),
        },
    }


def decode_generated(tokenizer, output_ids: torch.Tensor, input_len: int) -> str:
    if output_ids.ndim == 2 and output_ids.shape[1] > input_len:
        token_ids = output_ids[:, input_len:]
    else:
        token_ids = output_ids
    return tokenizer.batch_decode(token_ids, skip_special_tokens=True)[0].strip()


def relpath(path: str, base: Path) -> str:
    return str(Path(path).relative_to(base))


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[dict], output_dir: Path) -> None:
    fieldnames = [
        "idx",
        "image_id",
        "question_id",
        "answer_type",
        "question",
        "answer",
        "shortcut_answer",
        "stored_model_pred",
        "regenerated_pred",
        "result_bucket",
        "object_hits",
        "question_count",
        "activation_min",
        "activation_max",
        "activation_mean",
        "activation_std",
        "original",
        "overlay",
        "heatmap",
        "activation",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {name: row.get(name, "") for name in fieldnames}
            flat["object_hits"] = ";".join(row.get("object_hits", []))
            for key in ("original", "overlay", "heatmap", "activation"):
                flat[key] = relpath(row["files"][key], output_dir)
            writer.writerow(flat)


def write_html(path: Path, rows: List[dict], output_dir: Path, thumb_width: int) -> None:
    style = """
body{font-family:Arial,sans-serif;margin:18px;background:#f7f7f2;color:#202020}
table{border-collapse:collapse;width:100%%;font-size:13px}
th,td{border:1px solid #d0d0c8;padding:8px;vertical-align:top}
th{position:sticky;top:0;background:#ece9df;z-index:1}
img{max-width:%dpx;height:auto;display:block}
.meta{max-width:360px}
.small{color:#666;font-size:12px}
""" % thumb_width
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Gate Patch Activation Test Raw 100</title>",
        f"<style>{style}</style></head><body>",
        "<h1>Gate Patch Activation Test Raw 100</h1>",
        "<p>Images contain no question/answer text. Text is shown only in this index.</p>",
        "<table><thead><tr>",
        "<th>#</th><th>Original</th><th>Overlay</th><th>Heatmap</th><th>Question / Answers</th><th>Stats</th>",
        "</tr></thead><tbody>",
    ]
    for row in rows:
        files = row["files"]
        parts.extend(
            [
                "<tr>",
                f"<td>{row['idx']}</td>",
                f"<td><img src='{html.escape(relpath(files['original'], output_dir))}'></td>",
                f"<td><img src='{html.escape(relpath(files['overlay'], output_dir))}'></td>",
                f"<td><img src='{html.escape(relpath(files['heatmap'], output_dir))}'></td>",
                "<td class='meta'>",
                f"<b>qid:</b> {row['question_id']}<br>",
                f"<b>question:</b> {html.escape(row['question'])}<br>",
                f"<b>GT:</b> {html.escape(str(row.get('answer')))}<br>",
                f"<b>shortcut:</b> {html.escape(str(row.get('shortcut_answer')))}<br>",
                f"<b>stored model:</b> {html.escape(str(row.get('stored_model_pred')))}<br>",
                f"<b>regen:</b> {html.escape(str(row.get('regenerated_pred')))}<br>",
                f"<b>answer_type:</b> {html.escape(str(row.get('answer_type')))}<br>",
                f"<b>objects:</b> {html.escape(', '.join(row.get('object_hits', [])))}",
                "</td>",
                "<td class='small'>",
                f"grid={row['grid_shape']}<br>",
                f"mean={row['activation_mean']:.6f}<br>",
                f"min={row['activation_min']:.6f}<br>",
                f"max={row['activation_max']:.6f}<br>",
                f"std={row['activation_std']:.6f}<br>",
                f"align_ok={row['alignment']['output_size_matches_original']}<br>",
                f"crop={row['alignment'].get('crop_box_on_padded_square', '')}",
                "</td>",
                "</tr>",
            ]
        )
    parts.extend(["</tbody></table></body></html>"])
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_paths = ensure_dirs(args.output_dir)

    rows, source_path = load_rows(args.result_path, args.fallback_data_path)
    candidates = build_candidates(rows, args.image_folder, args.prefer)
    if not candidates:
        raise RuntimeError("No test_raw candidate images found")
    selected = candidates[: args.num_examples]
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
    image_aspect_ratio = getattr(model.config, "image_aspect_ratio", None)
    clip_tokenizer = CLIPTokenizer.from_pretrained(model.config.mm_vision_tower)

    metadata = []
    for idx, item in enumerate(selected):
        row = item["row"]
        image_path = Path(item["image_path"])
        image = Image.open(image_path).convert("RGB")
        question = str(row["question"])
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

        regenerated_pred = decode_generated(tokenizer, output_ids, input_ids.shape[1])
        activation_tensor = gate_module.current_gate_patch_activation
        if activation_tensor is None:
            raise RuntimeError(f"No gate patch activation captured for {image_path}")
        activation = activation_tensor.detach().float().cpu().numpy()[0]

        qid = int(row["question_id"])
        file_prefix = f"{idx:03d}_{image_path.stem}_qid{qid}"
        stats_and_files = save_visuals(
            image=image,
            activation=activation,
            file_prefix=file_prefix,
            output_paths=output_paths,
            image_aspect_ratio=image_aspect_ratio,
            overlay_alpha=args.overlay_alpha,
        )
        out_row = {
            "idx": idx,
            "image_id": int(item["image_id"]),
            "image_path": str(image_path),
            "question_count": int(item["question_count"]),
            "object_hits": item["object_hits"],
            "answer_type_counts": item["answer_type_counts"],
            "bucket_counts": item["bucket_counts"],
            "question_id": qid,
            "question": question,
            "answer": row.get("answer"),
            "shortcut_answer": row.get("shortcut_answer"),
            "answer_type": row.get("answer_type"),
            "stored_model_pred": row.get("model_pred"),
            "regenerated_pred": regenerated_pred,
            "result_bucket": result_bucket(row),
            **stats_and_files,
        }
        metadata.append(out_row)
        print(
            f"[done] {idx + 1}/{len(selected)} qid={qid} "
            f"mean={out_row['activation_mean']:.6f} align_ok={out_row['alignment']['output_size_matches_original']}",
            flush=True,
        )

    manifest = {
        "model_path": str(args.model_path),
        "source_path": str(source_path),
        "image_folder": str(args.image_folder),
        "num_examples": len(metadata),
        "image_aspect_ratio": image_aspect_ratio,
        "activation_definition": "gate.abs().mean(channel) per patch; sigmoid gate is non-negative, so abs equals mean.",
        "alignment_note": "For image_aspect_ratio=pad, heatmaps are resized to the padded square and cropped back to the original image box.",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_jsonl(args.output_dir / "index.jsonl", metadata)
    write_csv(args.output_dir / "index.csv", metadata, args.output_dir)
    write_html(args.output_dir / "index.html", metadata, args.output_dir, args.html_thumb_width)
    print(f"[done] aligned test_raw gate visualizations: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
