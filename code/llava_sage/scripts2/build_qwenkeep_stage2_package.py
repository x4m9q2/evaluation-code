#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


DEFAULT_BUNDLE_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the qwenkeep stage2 VQA package: keep SAM3 masks for filtered train_raw "
            "rows, keep removed rows without masks, and append VQAv2 rows without masks."
        )
    )
    parser.add_argument(
        "--train-raw",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT / "data/stage2/train_raw.json",
    )
    parser.add_argument(
        "--keep-json",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT
        / "analysis/visual_cue_question_filter_qwen35_strict4/full_all_shards/merged/keep.json",
    )
    parser.add_argument(
        "--remove-json",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT
        / "analysis/visual_cue_question_filter_qwen35_strict4/full_all_shards/merged/remove.json",
    )
    parser.add_argument(
        "--vqav2-train",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT / "data/stage2/vqa_train2014.json",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT / "outputs/sam3_train_raw_llava_union_masks/masks",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT / "data/stage2/train_raw_mixed_qwenratio_oldbase_sam3_plus_vqa.json",
    )
    parser.add_argument(
        "--output-mask-npz",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT / "data/stage2/patch_mask_analysis_train_raw_qwenkeep_sam3_compat.npz",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT / "data/stage2/train_raw_mixed_qwenratio_oldbase_sam3_plus_vqa.summary.json",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT / "models/llava-v1.5-7b/config.json",
    )
    parser.add_argument(
        "--vision-config",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT / "models/clip-vit-large-patch14-336/config.json",
    )
    parser.add_argument(
        "--preprocessor-config",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT / "models/clip-vit-large-patch14-336/preprocessor_config.json",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="If set, shuffle the final mixed JSON rows with this seed.",
    )
    parser.add_argument("--limit-keep", type=int, default=None)
    parser.add_argument("--limit-remove", type=int, default=None)
    parser.add_argument("--limit-vqa", type=int, default=None)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_train_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": int(row["question_id"]),
        "question": str(row["question"]).strip(),
        "image_id": int(row["image_id"]),
        "answer": str(row["answer"]).strip(),
        "answer_type": row.get("answer_type", "other"),
    }


def load_mask(mask_path: Path) -> np.ndarray:
    mask = np.array(Image.open(mask_path).convert("L"))
    return (mask >= 128).astype(np.float32)


def pad_to_square(mask: np.ndarray) -> tuple[np.ndarray, int, int, int]:
    height, width = mask.shape
    side = max(height, width)
    pad_top = (side - height) // 2
    pad_left = (side - width) // 2
    square = np.zeros((side, side), dtype=np.float32)
    square[pad_top : pad_top + height, pad_left : pad_left + width] = mask
    return square, pad_top, pad_left, side


def build_patch_npz(
    mask_rows: list[dict[str, Any]],
    mask_dir: Path,
    model_config_path: Path,
    vision_config_path: Path,
    preprocessor_config_path: Path,
    output_path: Path,
    metadata_extra: dict[str, Any],
) -> None:
    model_cfg = load_json(model_config_path)
    vision_cfg = load_json(vision_config_path)["vision_config"]
    preprocessor_cfg = load_json(preprocessor_config_path)

    image_aspect_ratio = model_cfg.get("image_aspect_ratio", "square")
    if image_aspect_ratio != "pad":
        raise ValueError(f"Expected image_aspect_ratio=pad, got {image_aspect_ratio!r}")

    image_size = int(vision_cfg["image_size"])
    patch_size = int(vision_cfg["patch_size"])
    grid_size = image_size // patch_size
    crop_size = preprocessor_cfg["crop_size"]
    crop_height = int(crop_size["height"] if isinstance(crop_size, dict) else crop_size)
    if crop_height != image_size:
        raise ValueError(f"Unexpected processor crop height {crop_height}; expected {image_size}.")

    mask_rows = sorted(mask_rows, key=lambda row: int(row["question_id"]))
    num_rows = len(mask_rows)
    image_names = np.empty(num_rows, dtype=object)
    question_ids = np.empty(num_rows, dtype=np.int64)
    image_ids = np.empty(num_rows, dtype=np.int64)
    original_widths = np.empty(num_rows, dtype=np.int32)
    original_heights = np.empty(num_rows, dtype=np.int32)
    padded_sides = np.empty(num_rows, dtype=np.int32)
    pad_tops = np.empty(num_rows, dtype=np.int32)
    pad_lefts = np.empty(num_rows, dtype=np.int32)
    mask_pixel_counts = np.empty(num_rows, dtype=np.int64)
    matched_instance_counts = np.ones(num_rows, dtype=np.int32)
    coverage_ratio = np.empty((num_rows, grid_size, grid_size), dtype=np.float32)
    has_mask = np.empty((num_rows, grid_size, grid_size), dtype=np.bool_)

    for idx, row in enumerate(mask_rows):
        qid = int(row["question_id"])
        image_id = int(row["image_id"])
        mask_path = mask_dir / f"{qid}.png"
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        mask = load_mask(mask_path)
        square_mask, pad_top, pad_left, side = pad_to_square(mask)
        coverage = F.adaptive_avg_pool2d(
            torch.from_numpy(square_mask).unsqueeze(0).unsqueeze(0),
            (grid_size, grid_size),
        ).squeeze(0).squeeze(0).numpy().astype(np.float32)
        contains_mask = coverage > 0.0

        image_names[idx] = f"{qid}_{image_id}.png"
        question_ids[idx] = qid
        image_ids[idx] = image_id
        original_heights[idx], original_widths[idx] = mask.shape
        padded_sides[idx] = side
        pad_tops[idx] = pad_top
        pad_lefts[idx] = pad_left
        mask_pixel_counts[idx] = int(contains_mask.sum())
        coverage_ratio[idx] = coverage
        has_mask[idx] = contains_mask

    metadata = {
        "model_config": str(model_config_path),
        "vision_config": str(vision_config_path),
        "preprocessor_config": str(preprocessor_config_path),
        "mask_dir": str(mask_dir),
        "image_aspect_ratio": image_aspect_ratio,
        "image_size": image_size,
        "patch_size": patch_size,
        "grid_size": grid_size,
        "num_patches_per_image": grid_size * grid_size,
        "patch_order": "row-major (coverage_ratio[i, row, col])",
        "coverage_ratio_definition": "fraction of each visual patch covered by the binary mask after LLaVA's pad-to-square preprocessing",
        "contains_mask_definition": "coverage_ratio > 0",
        "image_name_format": "<question_id>_<image_id>.png",
        "question_ids_definition": "explicit question ids aligned with train_raw_filtered_masked rows sorted by question_id",
        "mask_pixel_counts_definition": "count of active visual patches, not raw pixel count",
        "matched_instance_counts_definition": "set to 1 for SAM3 union masks",
        **metadata_extra,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        metadata_json=np.array(json.dumps(metadata, ensure_ascii=True)),
        image_names=image_names,
        question_ids=question_ids,
        image_ids=image_ids,
        original_widths=original_widths,
        original_heights=original_heights,
        padded_sides=padded_sides,
        pad_tops=pad_tops,
        pad_lefts=pad_lefts,
        mask_pixel_counts=mask_pixel_counts,
        matched_instance_counts=matched_instance_counts,
        coverage_ratio=coverage_ratio,
        has_mask=has_mask,
    )


def maybe_limit(rows: list[Any], limit: int | None) -> list[Any]:
    if limit is None:
        return rows
    return rows[:limit]


def main() -> None:
    args = parse_args()

    train_raw = [normalize_train_row(x) for x in load_json(args.train_raw)]
    train_by_qid = {int(x["question_id"]): x for x in train_raw}
    if len(train_by_qid) != len(train_raw):
        raise ValueError("Duplicate question_id in train_raw.")

    keep_rows = maybe_limit(load_json(args.keep_json), args.limit_keep)
    remove_rows = maybe_limit(load_json(args.remove_json), args.limit_remove)
    vqa_rows = maybe_limit(
        [normalize_train_row(x) for x in load_json(args.vqav2_train)],
        args.limit_vqa,
    )

    keep_qids = [int(x["question_id"]) for x in keep_rows]
    remove_qids = [int(x["question_id"]) for x in remove_rows]
    keep_qid_set = set(keep_qids)
    remove_qid_set = set(remove_qids)
    if keep_qid_set & remove_qid_set:
        overlap = sorted(keep_qid_set & remove_qid_set)[:20]
        raise ValueError(f"keep/remove overlap: {overlap}")

    train_qids = set(train_by_qid)
    missing_train_qids = (keep_qid_set | remove_qid_set) - train_qids
    if missing_train_qids:
        raise ValueError(f"Missing question ids in train_raw: {sorted(missing_train_qids)[:20]}")

    if args.limit_keep is None and args.limit_remove is None and (keep_qid_set | remove_qid_set) != train_qids:
        missing = sorted(train_qids - (keep_qid_set | remove_qid_set))[:20]
        extra = sorted((keep_qid_set | remove_qid_set) - train_qids)[:20]
        raise ValueError(
            "Filter coverage mismatch: "
            f"missing_from_filter={len(train_qids - (keep_qid_set | remove_qid_set))} sample={missing} "
            f"extra_in_filter={len((keep_qid_set | remove_qid_set) - train_qids)} sample={extra}"
        )

    mixed_rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    missing_mask_qids: list[int] = []

    for qid in keep_qids:
        row = dict(train_by_qid[qid])
        mask_path = args.mask_dir / f"{qid}.png"
        if mask_path.exists():
            row["data_source"] = "train_raw_filtered_masked"
            row["mask_supervision"] = "sam3_patch_mask"
            mask_rows.append(row)
        else:
            row["data_source"] = "train_raw_filtered_missingmask_nomask"
            row["mask_supervision"] = "none"
            missing_mask_qids.append(qid)
        mixed_rows.append(row)

    for qid in remove_qids:
        row = dict(train_by_qid[qid])
        row["data_source"] = "train_raw_removed_nomask"
        row["mask_supervision"] = "none"
        mixed_rows.append(row)

    for row in vqa_rows:
        out = dict(row)
        out["data_source"] = "vqa_train2014_nomask"
        out["mask_supervision"] = "none"
        mixed_rows.append(out)

    if args.shuffle_seed is not None:
        random.Random(args.shuffle_seed).shuffle(mixed_rows)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(mixed_rows, f, ensure_ascii=False)

    build_patch_npz(
        mask_rows=mask_rows,
        mask_dir=args.mask_dir,
        model_config_path=args.model_config,
        vision_config_path=args.vision_config,
        preprocessor_config_path=args.preprocessor_config,
        output_path=args.output_mask_npz,
        metadata_extra={
            "compat_source": "qwen_keep_only",
            "compat_source_json": str(args.output_json),
            "compat_keep_json": str(args.keep_json),
            "compat_remove_json": str(args.remove_json),
        },
    )

    summary = {
        "output_json": str(args.output_json),
        "output_mask_npz": str(args.output_mask_npz),
        "train_raw_total": len(train_raw),
        "train_raw_keep_total": len(keep_qids),
        "train_raw_remove_total": len(remove_qids),
        "train_raw_masked_total": len(mask_rows),
        "train_raw_missingmask_nomask_total": len(missing_mask_qids),
        "vqa_total": len(vqa_rows),
        "mixed_total": len(mixed_rows),
        "shuffle_seed": args.shuffle_seed,
        "sources": dict(Counter(x["data_source"] for x in mixed_rows)),
        "mask_supervision_counts": dict(Counter(x["mask_supervision"] for x in mixed_rows)),
        "missing_mask_qids_head": missing_mask_qids[:20],
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
