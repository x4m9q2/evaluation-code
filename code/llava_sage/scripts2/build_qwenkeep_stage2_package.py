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
            "Build the qwenkeep stage2 VQA package: keep SAM3 masks for filtered generated-train "
            "rows, keep removed rows without masks, and append original VQA rows from CMSV "
            "train/val/test without masks."
        )
    )
    parser.add_argument(
        "--generated-train",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT / "data/shortcut_pipeline/vqa_v2_cmsv/train.json",
        help="Generated train split used for supervised rows and Qwen keep/remove filtering.",
    )
    parser.add_argument(
        "--stage2-input-json",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT / "data/shortcut_pipeline/cross_modality_qa_input.json",
        help=(
            "Optional full stage-2 candidate JSON used to backfill generated-train rows "
            "when --generated-train points to a split train.json that does not cover all "
            "keep/remove question ids."
        ),
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
        "--original-split",
        type=Path,
        action="append",
        default=None,
        help=(
            "CMSV split JSON files containing original_question/original_answer. "
            "If omitted, defaults to data/shortcut_pipeline/vqa_v2_cmsv/{train,val,test}.json."
        ),
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT / "data/shortcut_pipeline/union_mask/masks",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT / "data/stage2/train_raw_mixed_qwenratio_oldbase_sam3_plus_vqa_nonumbermask.json",
    )
    parser.add_argument(
        "--output-mask-npz",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT / "data/stage2/patch_mask_analysis_train_raw_qwenkeep_sam3_nonumbermask_compat.npz",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT / "data/stage2/train_raw_mixed_qwenratio_oldbase_sam3_plus_vqa_nonumbermask.summary.json",
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
    parser.add_argument("--limit-original", type=int, default=None)
    parser.add_argument(
        "--original-question-id-offset",
        type=int,
        default=None,
        help=(
            "Offset added to source question_id for appended original no-mask rows. "
            "Default: max generated-train question_id + 1."
        ),
    )
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


def load_stage2_input_rows(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    rows = payload.get("results", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list or a dict with 'results' in stage2 input: {path}")
    return [normalize_train_row(x) for x in rows]


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


def default_original_splits() -> list[Path]:
    base = DEFAULT_BUNDLE_ROOT / "data/shortcut_pipeline/vqa_v2_cmsv"
    return [base / "train.json", base / "val.json", base / "test.json"]


def build_original_rows(split_paths: list[Path]) -> tuple[list[dict[str, Any]], Counter]:
    original_rows: list[dict[str, Any]] = []
    original_by_split: Counter = Counter()
    seen_keys: set[tuple[int, str, str]] = set()

    for path in split_paths:
        split_name = path.stem.strip().lower() or "unknown"
        rows = load_json(path)
        if not isinstance(rows, list):
            raise ValueError(f"Expected a list in original split: {path}")
        for row in rows:
            question = str(row.get("original_question", "")).strip()
            answer = str(row.get("original_answer", "")).strip()
            image_id = row.get("image_id")
            answer_type = str(row.get("answer_type", "other")).strip() or "other"
            if not question or not answer or image_id is None:
                raise ValueError(
                    f"Missing original_question/original_answer/image_id in {path} "
                    f"for question_id={row.get('question_id')}"
                )

            image_id_int = int(image_id)
            dedup_key = (image_id_int, question, answer)
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            original_rows.append(
                {
                    "question_id": int(row["question_id"]),
                    "source_question_id": int(row["question_id"]),
                    "question": question,
                    "image_id": image_id_int,
                    "answer": answer,
                    "answer_type": answer_type,
                    "origin_split": split_name,
                }
            )
            original_by_split[split_name] += 1

    return original_rows, original_by_split


def main() -> None:
    args = parse_args()
    original_split_paths = args.original_split or default_original_splits()

    train_raw = [normalize_train_row(x) for x in load_json(args.generated_train)]
    train_by_qid = {int(x["question_id"]): x for x in train_raw}
    if len(train_by_qid) != len(train_raw):
        raise ValueError("Duplicate question_id in generated_train.")

    keep_rows = maybe_limit(load_json(args.keep_json), args.limit_keep)
    remove_rows = maybe_limit(load_json(args.remove_json), args.limit_remove)
    original_rows, original_by_split = build_original_rows(original_split_paths)
    original_rows = maybe_limit(original_rows, args.limit_original)
    original_by_split = Counter(str(row.get("origin_split", "unknown")) for row in original_rows)

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
        if args.stage2_input_json.exists():
            stage2_input_rows = load_stage2_input_rows(args.stage2_input_json)
            stage2_by_qid = {int(x["question_id"]): x for x in stage2_input_rows}
            for qid in sorted(missing_train_qids):
                row = stage2_by_qid.get(qid)
                if row is not None:
                    train_by_qid[qid] = row
            train_qids = set(train_by_qid)
            missing_train_qids = (keep_qid_set | remove_qid_set) - train_qids
        if missing_train_qids:
            raise ValueError(
                "Missing question ids in generated_train even after stage2-input backfill: "
                f"{sorted(missing_train_qids)[:20]}"
            )

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
    number_mask_removed_qids: list[int] = []

    for qid in keep_qids:
        row = dict(train_by_qid[qid])
        mask_path = args.mask_dir / f"{qid}.png"
        if mask_path.exists():
            if str(row.get("answer_type", "")) == "number":
                row["data_source"] = "train_raw_filtered_number_nomask"
                row["mask_supervision"] = "none"
                number_mask_removed_qids.append(qid)
            else:
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

    original_qid_offset = (
        args.original_question_id_offset
        if args.original_question_id_offset is not None
        else (max(train_qids) + 1 if train_qids else 1)
    )
    for idx, row in enumerate(original_rows):
        out = dict(row)
        # Keep original no-mask rows outside the generated-train qid space so mask matching
        # cannot accidentally activate on them.
        out["question_id"] = original_qid_offset + idx
        out["data_source"] = f"vqa_original_{out.get('origin_split', 'unknown')}_nomask"
        out["mask_supervision"] = "none"
        mixed_rows.append(out)

    qid_counts = Counter(int(x["question_id"]) for x in mixed_rows)
    duplicate_qids = [qid for qid, count in qid_counts.items() if count > 1]
    if duplicate_qids:
        raise ValueError(f"Output contains duplicate question_id values: {duplicate_qids[:20]}")

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
            "compat_source": "qwen_keep_nonumbermask",
            "compat_source_json": str(args.output_json),
            "compat_generated_train": str(args.generated_train),
            "compat_keep_json": str(args.keep_json),
            "compat_remove_json": str(args.remove_json),
            "compat_original_splits": [str(path) for path in original_split_paths],
            "original_question_id_offset": original_qid_offset,
            "nonumbermask_rule": "keep all JSON rows, but set answer_type == 'number' mask_supervision to none and drop those mask rows from NPZ",
            "mixed_sample_rule": "generated train questions plus original questions reconstructed from generated train/val/test splits; original rows are no-mask and question_id-offset to avoid NPZ mask collisions",
        },
    )

    summary = {
        "output_json": str(args.output_json),
        "output_mask_npz": str(args.output_mask_npz),
        "generated_train": str(args.generated_train),
        "original_splits": [str(path) for path in original_split_paths],
        "generated_train_total": len(train_raw),
        "generated_train_keep_total": len(keep_qids),
        "generated_train_remove_total": len(remove_qids),
        "generated_train_masked_total": len(mask_rows),
        "generated_train_missingmask_nomask_total": len(missing_mask_qids),
        "generated_train_number_mask_removed_total": len(number_mask_removed_qids),
        "original_total": len(original_rows),
        "original_by_split": dict(original_by_split),
        "original_question_id_offset": original_qid_offset,
        "mixed_total": len(mixed_rows),
        "shuffle_seed": args.shuffle_seed,
        "sources": dict(Counter(x["data_source"] for x in mixed_rows)),
        "mask_supervision_counts": dict(Counter(x["mask_supervision"] for x in mixed_rows)),
        "missing_mask_qids_head": missing_mask_qids[:20],
        "number_mask_removed_qids_head": number_mask_removed_qids[:20],
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
