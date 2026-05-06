#!/usr/bin/env python3
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def default_root() -> Path:
    return Path(__file__).resolve().parents[3]


ROOT = default_root()
MODEL_CONFIG = ROOT / "models/llava-v1.5-7b/config.json"
VISION_CONFIG = ROOT / "models/clip-vit-large-patch14-336/config.json"
PREPROCESSOR_CONFIG = ROOT / "models/clip-vit-large-patch14-336/preprocessor_config.json"


DATASETS = {
    "gqa": {
        "input_json": ROOT / "data2/GQA/GQA_filtered_sampled_10000.json",
        "filter_root": ROOT / "analysis/gqa_sampled10000_qwen35_filter",
        "mask_dir": ROOT / "analysis/gqa_sampled10000_sam3_union_masks/masks",
        "output_json": ROOT / "data2/GQA/GQA_filtered_sampled_10000_qwenkeep_sam3_nonumbermask.json",
        "output_npz": ROOT / "patch_mask_analysis_gqa_sampled10000_qwenkeep_sam3_nonumbermask_compat.npz",
        "summary_json": ROOT / "analysis/gqa_sampled10000_qwen35_filter/qwenkeep_sam3_nonumbermask_package_summary.json",
    },
    "vg": {
        "input_json": ROOT / "data2/vg/vg_filtered_sampled_10000.json",
        "filter_root": ROOT / "analysis/vg_sampled10000_qwen35_filter",
        "mask_dir": ROOT / "analysis/vg_sampled10000_sam3_union_masks/masks",
        "output_json": ROOT / "data2/vg/vg_filtered_sampled_10000_qwenkeep_sam3_nonumbermask.json",
        "output_npz": ROOT / "patch_mask_analysis_vg_sampled10000_qwenkeep_sam3_nonumbermask_compat.npz",
        "summary_json": ROOT / "analysis/vg_sampled10000_qwen35_filter/qwenkeep_sam3_nonumbermask_package_summary.json",
    },
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def sort_key(row):
    return int(row["question_id"])


def merge_filter_outputs(filter_root: Path):
    merged_dir = filter_root / "merged"
    if merged_dir.exists() and not (filter_root / "run_00").exists():
        keep_rows = load_json(merged_dir / "keep.json")
        remove_rows = load_json(merged_dir / "remove.json")
        summary_path = merged_dir / "summary.json"
        summary = load_json(summary_path) if summary_path.exists() else {
            "filter_root": str(filter_root),
            "total_rows": len(keep_rows) + len(remove_rows),
            "kept_rows": len(keep_rows),
            "removed_rows": len(remove_rows),
            "removed_ratio": len(remove_rows) / max(1, len(keep_rows) + len(remove_rows)),
            "source": "premerged",
        }
        return keep_rows, remove_rows, summary

    keep_rows = []
    remove_rows = []
    audit_rows = []
    shard_summaries = []
    for shard in range(4):
        run_dir = filter_root / f"run_{shard:02d}"
        keep_rows.extend(load_json(run_dir / "keep.json"))
        remove_rows.extend(load_json(run_dir / "remove.json"))
        shard_summaries.append(load_json(run_dir / "summary.json"))
        with (run_dir / "audit.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    audit_rows.append(json.loads(line))

    keep_rows.sort(key=sort_key)
    remove_rows.sort(key=sort_key)
    audit_rows.sort(key=sort_key)

    keep_qids = {int(x["question_id"]) for x in keep_rows}
    remove_qids = {int(x["question_id"]) for x in remove_rows}
    if keep_qids & remove_qids:
        overlap = sorted(keep_qids & remove_qids)[:20]
        raise ValueError(f"keep/remove overlap: {overlap}")
    if len(keep_qids) != len(keep_rows) or len(remove_qids) != len(remove_rows):
        raise ValueError("Duplicate question_id in merged filter outputs.")

    summary = {
        "filter_root": str(filter_root),
        "total_rows": len(keep_rows) + len(remove_rows),
        "kept_rows": len(keep_rows),
        "removed_rows": len(remove_rows),
        "removed_ratio": len(remove_rows) / max(1, len(keep_rows) + len(remove_rows)),
        "shards": shard_summaries,
    }

    write_json(merged_dir / "keep.json", keep_rows)
    write_json(merged_dir / "remove.json", remove_rows)
    write_json(merged_dir / "summary.json", summary)
    with (merged_dir / "audit.jsonl").open("w", encoding="utf-8") as f:
        for row in audit_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return keep_rows, remove_rows, summary


def normalize_dataset_row(row):
    question = row.get("generated_question") or row.get("question") or row.get("text") or ""
    answer = row.get("generated_answer") or row.get("answer") or row.get("original_answer") or ""
    image_path = row.get("image_path") or row.get("image") or ""
    return {
        **row,
        "question_id": int(row["question_id"]),
        "image_id": int(row["image_id"]),
        "question": str(question).strip(),
        "answer": str(answer).strip(),
        "image": str(image_path),
        "answer_type": row.get("answer_type", "other"),
    }


def load_mask(mask_path: Path):
    mask = np.array(Image.open(mask_path).convert("L"))
    return (mask >= 128).astype(np.float32)


def pad_to_square(mask: np.ndarray):
    height, width = mask.shape
    side = max(height, width)
    pad_top = (side - height) // 2
    pad_left = (side - width) // 2
    square = np.zeros((side, side), dtype=np.float32)
    square[pad_top : pad_top + height, pad_left : pad_left + width] = mask
    return square, pad_top, pad_left, side


def build_patch_npz(mask_rows, mask_dir: Path, output_path: Path, metadata_extra):
    model_cfg = load_json(MODEL_CONFIG)
    vision_cfg = load_json(VISION_CONFIG)["vision_config"]
    preprocessor_cfg = load_json(PREPROCESSOR_CONFIG)

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
        "model_config": str(MODEL_CONFIG),
        "vision_config": str(VISION_CONFIG),
        "preprocessor_config": str(PREPROCESSOR_CONFIG),
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
        "question_ids_definition": "explicit question ids aligned with rows",
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


def build_dataset_package(name, cfg):
    rows = [normalize_dataset_row(x) for x in load_json(cfg["input_json"])]
    rows_by_qid = {int(x["question_id"]): x for x in rows}
    if len(rows_by_qid) != len(rows):
        raise ValueError(f"{name}: duplicate question_id in input json.")

    keep_rows, remove_rows, filter_summary = merge_filter_outputs(cfg["filter_root"])
    keep_qids = {int(x["question_id"]) for x in keep_rows}
    remove_qids = {int(x["question_id"]) for x in remove_rows}
    filter_qids = keep_qids | remove_qids
    input_qids = set(rows_by_qid)
    if filter_qids != input_qids:
        raise ValueError(
            f"{name}: filter coverage mismatch "
            f"missing={len(input_qids - filter_qids)} extra={len(filter_qids - input_qids)}"
        )

    missing_masks = []
    number_mask_removed = []
    output_rows = []
    mask_rows = []
    for qid in sorted(input_qids):
        row = dict(rows_by_qid[qid])
        if qid in keep_qids:
            mask_path = cfg["mask_dir"] / f"{qid}.png"
            if mask_path.exists():
                if str(row.get("answer_type", "")) == "number":
                    row["data_source"] = f"{name}_qwenkeep_number_nomask"
                    row["mask_supervision"] = "none"
                    number_mask_removed.append(qid)
                else:
                    row["data_source"] = f"{name}_qwenkeep_sam3_masked"
                    row["mask_supervision"] = "sam3_patch_mask"
                    mask_rows.append(row)
            else:
                row["data_source"] = f"{name}_qwenkeep_missingmask_nomask"
                row["mask_supervision"] = "none"
                missing_masks.append(qid)
        else:
            row["data_source"] = f"{name}_qwenremove_nomask"
            row["mask_supervision"] = "none"
        output_rows.append(row)

    write_json(cfg["output_json"], output_rows)
    build_patch_npz(
        mask_rows,
        cfg["mask_dir"],
        cfg["output_npz"],
        {
            "dataset": name,
            "input_json": str(cfg["input_json"]),
            "output_json": str(cfg["output_json"]),
            "qwen_filter_merged_dir": str(cfg["filter_root"] / "merged"),
            "compat_source": "qwen_keep_nonumbermask_gqa_vg_sam3",
            "nonumbermask_rule": "keep all JSON rows, but set answer_type == 'number' mask_supervision to none and drop those mask rows from NPZ",
        },
    )

    summary = {
        "dataset": name,
        "input_json": str(cfg["input_json"]),
        "output_json": str(cfg["output_json"]),
        "output_mask_npz": str(cfg["output_npz"]),
        "input_total": len(rows),
        "qwen_keep_total": len(keep_qids),
        "qwen_remove_total": len(remove_qids),
        "masked_total": len(mask_rows),
        "missing_mask_nomask_total": len(missing_masks),
        "number_mask_removed_total": len(number_mask_removed),
        "mask_supervision_counts": dict(Counter(x["mask_supervision"] for x in output_rows)),
        "data_source_counts": dict(Counter(x["data_source"] for x in output_rows)),
        "missing_mask_qids_head": missing_masks[:20],
        "number_mask_removed_qids_head": number_mask_removed[:20],
        "filter_summary": filter_summary,
    }
    write_json(cfg["summary_json"], summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    for name, cfg in DATASETS.items():
        build_dataset_package(name, cfg)


if __name__ == "__main__":
    main()
