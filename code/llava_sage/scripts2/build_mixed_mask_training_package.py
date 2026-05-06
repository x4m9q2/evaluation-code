#!/usr/bin/env python3
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


DEFAULT_BUNDLE_ROOT = Path(__file__).resolve().parents[3]


DEFAULT_COARSE_CUES = [
    "dining table",
    "person",
    "chair",
    "bed",
    "couch",
    "bench",
    "tv",
    "bowl",
    "cell phone",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a mixed train package where filtered train_raw rows keep mask supervision, "
            "while removed or demoted rows and VQAv2 rows stay in the data without mask supervision."
        )
    )
    parser.add_argument("--train-raw", type=Path, default=DEFAULT_BUNDLE_ROOT / "data/stage2/train_raw.json")
    parser.add_argument(
        "--keep-json",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT / "analysis/visual_cue_question_filter_qwen35_strict4/full_all_shards/merged/keep.json",
    )
    parser.add_argument(
        "--remove-json",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT / "analysis/visual_cue_question_filter_qwen35_strict4/full_all_shards/merged/remove.json",
    )
    parser.add_argument("--vqav2-train", type=Path, default=DEFAULT_BUNDLE_ROOT / "data/stage2/vqa_train2014.json")
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT / "outputs/sam3_train_raw_llava_union_masks/masks",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-mask-npz",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--coarse-cues",
        nargs="*",
        default=DEFAULT_COARSE_CUES,
        help="Masked-keep rows containing any of these cues will be demoted to no-mask.",
    )
    parser.add_argument("--shuffle-seed", type=int, default=20260419)
    parser.add_argument("--no-shuffle", action="store_true")
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
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_mask(mask_path: Path):
    mask = np.array(Image.open(mask_path).convert("L"))
    # SAM3 union PNGs are saved as white foreground on black background.
    return (mask >= 128).astype(np.float32)


def pad_to_square(mask: np.ndarray):
    height, width = mask.shape
    side = max(height, width)
    pad_top = (side - height) // 2
    pad_left = (side - width) // 2
    square = np.zeros((side, side), dtype=np.float32)
    square[pad_top:pad_top + height, pad_left:pad_left + width] = mask
    return square, pad_top, pad_left, side


def compute_patch_coverage(square_mask: np.ndarray, grid_size: int):
    mask_tensor = torch.from_numpy(square_mask).unsqueeze(0).unsqueeze(0)
    coverage = F.adaptive_avg_pool2d(mask_tensor, (grid_size, grid_size))
    return coverage.squeeze(0).squeeze(0).numpy().astype(np.float32)


def build_patch_npz(mask_paths, model_config_path: Path, vision_config_path: Path, preprocessor_config_path: Path, output_path: Path):
    model_cfg = load_json(model_config_path)
    vision_cfg = load_json(vision_config_path)["vision_config"]
    preprocessor_cfg = load_json(preprocessor_config_path)

    image_aspect_ratio = model_cfg.get("image_aspect_ratio", "square")
    if image_aspect_ratio != "pad":
        raise ValueError(f"Expected pad aspect ratio, got {image_aspect_ratio!r}")

    image_size = int(vision_cfg["image_size"])
    patch_size = int(vision_cfg["patch_size"])
    grid_size = image_size // patch_size

    processor_crop_size = preprocessor_cfg["crop_size"]
    crop_height = int(processor_crop_size["height"] if isinstance(processor_crop_size, dict) else processor_crop_size)
    if crop_height != image_size:
        raise ValueError(f"Unexpected crop size {crop_height}; expected {image_size}.")

    num_images = len(mask_paths)
    image_names = np.empty(num_images, dtype=object)
    image_ids = np.empty(num_images, dtype=np.int64)
    original_widths = np.empty(num_images, dtype=np.int32)
    original_heights = np.empty(num_images, dtype=np.int32)
    padded_sides = np.empty(num_images, dtype=np.int32)
    pad_tops = np.empty(num_images, dtype=np.int32)
    pad_lefts = np.empty(num_images, dtype=np.int32)
    coverage_ratio = np.empty((num_images, grid_size, grid_size), dtype=np.float32)
    has_mask = np.empty((num_images, grid_size, grid_size), dtype=np.bool_)

    for idx, mask_path in enumerate(mask_paths):
        mask = load_mask(mask_path)
        square_mask, pad_top, pad_left, side = pad_to_square(mask)
        coverage = compute_patch_coverage(square_mask, grid_size)
        contains_mask = coverage > 0.0

        image_names[idx] = mask_path.name
        image_ids[idx] = int(mask_path.stem.rsplit("_", 1)[-1]) if "_" in mask_path.stem else -1
        original_heights[idx], original_widths[idx] = mask.shape
        padded_sides[idx] = side
        pad_tops[idx] = pad_top
        pad_lefts[idx] = pad_left
        coverage_ratio[idx] = coverage
        has_mask[idx] = contains_mask

        if (idx + 1) % 10000 == 0:
            print(f"mask_npz processed={idx + 1}/{num_images}")

    metadata = {
        "model_config": str(model_config_path),
        "vision_config": str(vision_config_path),
        "preprocessor_config": str(preprocessor_config_path),
        "mask_dir": str(mask_paths[0].parent if mask_paths else ""),
        "image_aspect_ratio": image_aspect_ratio,
        "image_size": image_size,
        "patch_size": patch_size,
        "grid_size": grid_size,
        "num_patches_per_image": grid_size * grid_size,
        "patch_order": "row-major (coverage_ratio[i, row, col])",
        "coverage_ratio_definition": "fraction of each visual patch covered by the binary mask after LLaVA's pad-to-square preprocessing",
        "contains_mask_definition": "coverage_ratio > 0",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        metadata_json=np.array(json.dumps(metadata, ensure_ascii=True)),
        image_names=image_names,
        image_ids=image_ids,
        original_widths=original_widths,
        original_heights=original_heights,
        padded_sides=padded_sides,
        pad_tops=pad_tops,
        pad_lefts=pad_lefts,
        coverage_ratio=coverage_ratio,
        has_mask=has_mask,
    )


def normalize_train_row(row):
    return {
        "question_id": int(row["question_id"]),
        "question": str(row["question"]).strip(),
        "image_id": int(row["image_id"]),
        "answer": str(row["answer"]).strip(),
        "answer_type": row.get("answer_type", "other"),
    }


def main():
    args = parse_args()
    coarse_cues = set(args.coarse_cues)

    train_raw = [normalize_train_row(x) for x in load_json(args.train_raw)]
    train_by_qid = {int(x["question_id"]): x for x in train_raw}
    if len(train_by_qid) != len(train_raw):
        raise ValueError("Duplicate question_id in train_raw.")

    keep_rows = load_json(args.keep_json)
    remove_rows = load_json(args.remove_json)
    vqa_rows = [normalize_train_row(x) for x in load_json(args.vqav2_train)]

    keep_qids = {int(x["question_id"]) for x in keep_rows}
    remove_qids = {int(x["question_id"]) for x in remove_rows}
    all_filter_qids = keep_qids | remove_qids
    train_qids = set(train_by_qid)
    if all_filter_qids != train_qids:
        missing = sorted(train_qids - all_filter_qids)[:20]
        extra = sorted(all_filter_qids - train_qids)[:20]
        raise ValueError(
            f"Filter qid coverage mismatch: missing_from_filter={len(train_qids - all_filter_qids)} sample={missing} "
            f"extra_in_filter={len(all_filter_qids - train_qids)} sample={extra}"
        )

    demoted_qids = set()
    for row in keep_rows:
        cues = set(row.get("visual_cues") or [])
        if cues & coarse_cues:
            demoted_qids.add(int(row["question_id"]))

    masked_qids_target = sorted(keep_qids - demoted_qids)
    demoted_qids = sorted(demoted_qids)
    removed_qids = sorted(remove_qids)

    mask_paths = []
    missing_mask_qids = []
    for qid in masked_qids_target:
        mask_path = args.mask_dir / f"{qid}.png"
        if mask_path.exists():
            mask_paths.append(mask_path)
        else:
            missing_mask_qids.append(qid)

    masked_qids = {int(p.stem) for p in mask_paths}
    missing_mask_qids = sorted(missing_mask_qids)

    mixed_rows = []

    for qid in sorted(masked_qids):
        row = dict(train_by_qid[qid])
        row["data_source"] = "train_raw_filtered_masked"
        row["mask_supervision"] = "sam3_patch_mask"
        mixed_rows.append(row)

    for qid in demoted_qids:
        row = dict(train_by_qid[qid])
        row["data_source"] = "train_raw_filtered_demoted_nomask"
        row["mask_supervision"] = "none"
        mixed_rows.append(row)

    for qid in removed_qids:
        row = dict(train_by_qid[qid])
        row["data_source"] = "train_raw_removed_nomask"
        row["mask_supervision"] = "none"
        mixed_rows.append(row)

    for qid in missing_mask_qids:
        row = dict(train_by_qid[qid])
        row["data_source"] = "train_raw_filtered_missingmask_nomask"
        row["mask_supervision"] = "none"
        mixed_rows.append(row)

    vqa_offset = max(train_qids) + 1
    for row in vqa_rows:
        remapped = dict(row)
        remapped["question_id"] = int(remapped["question_id"]) + vqa_offset
        remapped["data_source"] = "vqa_train2014_nomask"
        remapped["mask_supervision"] = "none"
        mixed_rows.append(remapped)

    if not args.no_shuffle:
        random.Random(args.shuffle_seed).shuffle(mixed_rows)

    unique_qids = {int(x["question_id"]) for x in mixed_rows}
    if len(unique_qids) != len(mixed_rows):
        raise ValueError("Output contains duplicate question_id values.")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(mixed_rows, f, ensure_ascii=False)

    build_patch_npz(
        sorted(mask_paths, key=lambda p: int(p.stem)),
        args.model_config,
        args.vision_config,
        args.preprocessor_config,
        args.output_mask_npz,
    )

    summary = {
        "output_json": str(args.output_json),
        "output_mask_npz": str(args.output_mask_npz),
        "train_raw_total": len(train_raw),
        "train_raw_keep_total": len(keep_qids),
        "train_raw_remove_total": len(remove_qids),
        "train_raw_masked_total": len(masked_qids),
        "train_raw_demoted_nomask_total": len(demoted_qids),
        "train_raw_missingmask_nomask_total": len(missing_mask_qids),
        "train_raw_removed_nomask_total": len(removed_qids),
        "vqa_total": len(vqa_rows),
        "mixed_total": len(mixed_rows),
        "coarse_cues": sorted(coarse_cues),
        "shuffle_seed": None if args.no_shuffle else args.shuffle_seed,
        "sources": dict(Counter(x["data_source"] for x in mixed_rows)),
        "mask_supervision_counts": dict(Counter(x["mask_supervision"] for x in mixed_rows)),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
