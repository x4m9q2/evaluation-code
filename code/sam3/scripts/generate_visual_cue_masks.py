#!/usr/bin/env python3
"""
Generate SAM3 masks for visual cues linked from image_rule.json -> rules.json.

This version supports batched inference (`--batch-size`) by flattening each
(question_id, visual_cue) pair as one inference unit.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.eval.postprocessors import PostProcessImage
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    Image as SAMImage,
    InferenceMetadata,
)
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    NormalizeAPI,
    RandomResizeAPI,
    ToTensorAPI,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate visual-cue masks for sampled question records."
    )
    parser.add_argument(
        "--image-rule-json",
        type=Path,
        default=Path("image_rule.json"),
    )
    parser.add_argument(
        "--rules-json",
        type=Path,
        default=Path("rules.json"),
    )
    parser.add_argument(
        "--train2014-dir",
        type=Path,
        default=Path("train2014"),
    )
    parser.add_argument(
        "--val2014-dir",
        type=Path,
        default=Path("train2014"),
    )
    parser.add_argument("--sample-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--checkpoint-path", type=str, default="sam3_ckpt/sam3.pt")
    parser.add_argument("--no-load-from-hf", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/visual_cue_masks_32"),
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_prompt(cue: str) -> str:
    # e.g. "racket,racquet" -> "racket"
    parts = [x.strip() for x in cue.split(",") if x.strip()]
    return parts[0] if parts else cue.strip()


def safe_stem(text: str) -> str:
    txt = text.strip().replace(" ", "_")
    txt = re.sub(r"[^0-9a-zA-Z_\-]+", "_", txt)
    return txt[:80] if txt else "empty"


def resolve_image_path(image_id: int, train_dir: Path, val_dir: Path) -> Optional[Path]:
    train_path = train_dir / f"COCO_train2014_{image_id:012d}.jpg"
    if train_path.exists():
        return train_path
    val_path = val_dir / f"COCO_val2014_{image_id:012d}.jpg"
    if val_path.exists():
        return val_path
    return None


def create_transform(resolution: int) -> ComposeAPI:
    return ComposeAPI(
        transforms=[
            RandomResizeAPI(
                sizes=resolution,
                max_size=resolution,
                square=True,
                consistent_transform=False,
            ),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def create_datapoint(
    image: Image.Image, prompt: str, query_id: int, image_height: int, image_width: int
) -> Datapoint:
    datapoint = Datapoint(
        find_queries=[],
        images=[SAMImage(data=image, objects=[], size=[image_height, image_width])],
    )
    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=prompt,
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=query_id,
                original_image_id=query_id,
                original_category_id=1,
                original_size=[image_height, image_width],
                object_id=0,
                frame_index=0,
            ),
        )
    )
    return datapoint


def chunk(items: List[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def tensor_mask_to_2d(mask_tensor: torch.Tensor, h: int, w: int) -> np.ndarray:
    masks = mask_tensor.detach().cpu()
    if masks.numel() == 0:
        return np.zeros((h, w), dtype=np.uint8)

    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]  # [N, H, W]
    if masks.ndim == 3:
        merged = masks.any(dim=0).to(torch.uint8).numpy() * 255
    elif masks.ndim == 2:
        merged = masks.to(torch.uint8).numpy() * 255
    else:
        raise ValueError(f"Unexpected mask shape: {tuple(mask_tensor.shape)}")

    if merged.shape != (h, w):
        merged = np.array(
            Image.fromarray(merged).resize((w, h), resample=Image.NEAREST), dtype=np.uint8
        )
    return merged


def result_masks_to_2d(result_masks: Any, h: int, w: int) -> np.ndarray:
    if result_masks is None:
        return np.zeros((h, w), dtype=np.uint8)

    if isinstance(result_masks, torch.Tensor):
        return tensor_mask_to_2d(result_masks, h, w)

    if isinstance(result_masks, list):
        merged = np.zeros((h, w), dtype=np.uint8)
        for item in result_masks:
            if isinstance(item, torch.Tensor):
                merged = np.maximum(merged, tensor_mask_to_2d(item, h, w))
        return merged

    return np.zeros((h, w), dtype=np.uint8)


def count_instances(result_masks: Any) -> int:
    if result_masks is None:
        return 0
    if isinstance(result_masks, torch.Tensor):
        if result_masks.ndim >= 3:
            return int(result_masks.shape[0])
        return int(result_masks.numel() > 0)
    if isinstance(result_masks, list):
        total = 0
        for item in result_masks:
            if isinstance(item, torch.Tensor):
                if item.ndim >= 3:
                    total += int(item.shape[0])
                elif item.ndim == 2:
                    total += 1
        return total
    return 0


def make_overlay(image_rgb: np.ndarray, mask_2d: np.ndarray) -> np.ndarray:
    overlay = image_rgb.copy()
    red = np.zeros_like(overlay)
    red[:, :, 0] = 255
    alpha = 0.45
    mask_bool = mask_2d > 0
    overlay[mask_bool] = (
        (1.0 - alpha) * overlay[mask_bool] + alpha * red[mask_bool]
    ).astype(np.uint8)
    return overlay


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    if args.no_load_from_hf and args.checkpoint_path is None:
        if os.getenv("SAM3_CHECKPOINT_PATH") is None:
            raise ValueError(
                "--no-load-from-hf is set but no checkpoint is available. "
                "Set --checkpoint-path or SAM3_CHECKPOINT_PATH."
            )

    image_rule = load_json(args.image_rule_json)
    rules_obj = load_json(args.rules_json)

    records: List[Dict[str, Any]] = image_rule["results"]
    rules_list: List[Dict[str, Any]] = rules_obj["rules"]
    rule_map: Dict[int, Dict[str, Any]] = {int(r["rule_id"]): r for r in rules_list}

    # Build valid question-level samples.
    valid_records: List[Tuple[Dict[str, Any], Path, int, List[str]]] = []
    for rec in records:
        rule_id = int(rec["rule_id"])
        if rule_id <= 0:
            continue
        rule = rule_map.get(rule_id)
        if rule is None:
            continue
        visual_cues = [str(x) for x in rule.get("visual_cues", []) if str(x).strip()]
        if not visual_cues:
            continue
        image_path = resolve_image_path(
            int(rec["image_id"]), args.train2014_dir, args.val2014_dir
        )
        if image_path is None:
            continue
        valid_records.append((rec, image_path, rule_id, visual_cues))

    if not valid_records:
        raise RuntimeError("No valid records found for sampling.")

    random.seed(args.seed)
    random.shuffle(valid_records)
    sampled = valid_records[: args.sample_size]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    masks_root = args.output_dir / "masks"
    masks_root.mkdir(parents=True, exist_ok=True)

    model = build_sam3_image_model(
        device=args.device,
        eval_mode=True,
        checkpoint_path=args.checkpoint_path,
        load_from_HF=not args.no_load_from_hf,
    )
    transform = create_transform(args.resolution)
    postprocessor = PostProcessImage(
        max_dets_per_img=-1,
        iou_type="segm",
        use_original_sizes_box=True,
        use_original_sizes_mask=True,
        convert_mask_to_rle=False,
        detection_threshold=args.score_thresh,
        to_cpu=True,
    )
    device = torch.device(args.device)

    # Stream questions into fixed-size cue batches to avoid loading all images into RAM.
    summary_items: Dict[int, Dict[str, Any]] = {}
    active_items: Dict[int, Dict[str, Any]] = {}
    pending_units: List[Dict[str, Any]] = []

    processed_units = 0
    total_units = sum(len(visual_cues) for _, _, _, visual_cues in sampled)
    next_query_id = 1

    def process_batch(batch_units: List[Dict[str, Any]]) -> None:
        nonlocal processed_units
        if not batch_units:
            return

        datapoints: List[Datapoint] = []
        for unit in batch_units:
            item = active_items[unit["item_uid"]]
            dp = create_datapoint(
                image=item["image_pil"],
                prompt=unit["cue_prompt"],
                query_id=unit["query_id"],
                image_height=item["height"],
                image_width=item["width"],
            )
            dp = transform(dp)
            datapoints.append(dp)

        batch = collate_fn_api(datapoints, dict_key="batch")["batch"]
        batch = copy_data_to_device(batch, device, non_blocking=True)
        with torch.inference_mode():
            outputs = model(batch)
        processed = postprocessor.process_results(outputs, batch.find_metadatas)

        for unit in batch_units:
            query_id = unit["query_id"]
            item = active_items[unit["item_uid"]]
            pred = processed.get(query_id, {})
            masks = pred.get("masks")
            mask_2d = result_masks_to_2d(masks, item["height"], item["width"])
            num_instances = count_instances(masks)

            cue_file = (
                item["item_dir"]
                / f"cue_{unit['cue_idx']:02d}_{safe_stem(unit['cue_prompt'])}.png"
            )
            Image.fromarray(mask_2d, mode="L").save(cue_file)
            item["merged_mask"] = np.maximum(item["merged_mask"], mask_2d)
            item["cue_outputs"].append(
                {
                    "cue_raw": unit["cue_raw"],
                    "cue_prompt": unit["cue_prompt"],
                    "num_instances": num_instances,
                    "mask_path": str(cue_file),
                }
            )
            item["remaining_cues"] -= 1
            processed_units += 1
            if item["remaining_cues"] == 0:
                merged_mask_path = item["item_dir"] / "merged_mask.png"
                Image.fromarray(item["merged_mask"], mode="L").save(merged_mask_path)

                image_np = np.array(item["image_pil"])
                overlay = make_overlay(image_np, item["merged_mask"])
                overlay_path = item["item_dir"] / "overlay.png"
                Image.fromarray(overlay).save(overlay_path)

                summary_items[item["index"]] = {
                    "index": item["index"],
                    "question_id": item["question_id"],
                    "image_id": item["image_id"],
                    "image_path": item["image_path"],
                    "rule_id": item["rule_id"],
                    "visual_cues": item["visual_cues"],
                    "cue_outputs": item["cue_outputs"],
                    "merged_mask_path": str(merged_mask_path),
                    "overlay_path": str(overlay_path),
                }
                del active_items[unit["item_uid"]]

        print(f"processed cue units: {processed_units}/{total_units}", flush=True)

    for idx, (rec, image_path, rule_id, visual_cues) in enumerate(sampled, start=1):
        question_id = int(rec["question_id"])
        image_id = int(rec["image_id"])
        item_dir = masks_root / f"q_{question_id}"
        item_dir.mkdir(parents=True, exist_ok=True)

        image_pil = Image.open(image_path).convert("RGB")
        width, height = image_pil.size
        active_items[idx] = {
            "index": idx,
            "question_id": question_id,
            "image_id": image_id,
            "image_path": str(image_path),
            "rule_id": rule_id,
            "visual_cues": visual_cues,
            "item_dir": item_dir,
            "image_pil": image_pil,
            "height": height,
            "width": width,
            "merged_mask": np.zeros((height, width), dtype=np.uint8),
            "cue_outputs": [],
            "remaining_cues": len(visual_cues),
        }

        for cue_idx, cue_raw in enumerate(visual_cues):
            cue_prompt = normalize_prompt(cue_raw)
            pending_units.append(
                {
                    "query_id": next_query_id,
                    "item_uid": idx,
                    "cue_idx": cue_idx,
                    "cue_raw": cue_raw,
                    "cue_prompt": cue_prompt,
                }
            )
            next_query_id += 1

        while len(pending_units) >= args.batch_size:
            process_batch(pending_units[: args.batch_size])
            pending_units = pending_units[args.batch_size:]

    while pending_units:
        process_batch(pending_units[: args.batch_size])
        pending_units = pending_units[args.batch_size:]

    if active_items:
        raise RuntimeError(f"Found unfinished items after inference: {len(active_items)}")

    # Finalize summary.
    summary: Dict[str, Any] = {
        "sample_size_requested": args.sample_size,
        "sample_size_actual": len(sampled),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": args.device,
        "score_thresh": args.score_thresh,
        "items": [summary_items[i] for i in sorted(summary_items)],
    }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
