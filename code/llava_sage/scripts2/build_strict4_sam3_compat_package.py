#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_BUNDLE_ROOT = Path(__file__).resolve().parents[3]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebuild strict4 SAM3 mixed package with old-package-compatible NPZ fields."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT / "outputs/strict4_sam3/train_raw_mixed_strict4_sam3_allkeep_plus_removed_plus_vqa.json",
    )
    parser.add_argument(
        "--input-mask-npz",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT / "outputs/strict4_sam3/patch_mask_analysis_train_raw_strict4_sam3_allkeep.npz",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT / "outputs/strict4_sam3/train_raw_mixed_strict4_sam3_allkeep_plus_removed_plus_vqa_compat.json",
    )
    parser.add_argument(
        "--output-mask-npz",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT / "outputs/strict4_sam3/patch_mask_analysis_train_raw_strict4_sam3_allkeep_compat.npz",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with args.input_json.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=True)

    analysis = np.load(args.input_mask_npz, allow_pickle=True)
    metadata = json.loads(str(analysis["metadata_json"]))

    by_qid = {int(row["question_id"]): row for row in rows}
    image_names_in = analysis["image_names"]
    num_rows = len(image_names_in)

    question_ids = np.empty(num_rows, dtype=np.int64)
    image_ids = np.empty(num_rows, dtype=np.int64)
    image_names = np.empty(num_rows, dtype=object)
    mask_pixel_counts = np.empty(num_rows, dtype=np.int64)
    matched_instance_counts = np.ones(num_rows, dtype=np.int32)

    has_mask = analysis["has_mask"].astype(np.bool_)

    for idx, image_name in enumerate(image_names_in):
        stem = Path(str(image_name)).stem
        if not stem.isdigit():
            raise ValueError(f"Unexpected SAM3 mask image name: {image_name!r}")
        qid = int(stem)
        row = by_qid.get(qid)
        if row is None:
            raise KeyError(f"Question id {qid} from NPZ not found in mixed json.")
        image_id = int(row["image_id"])
        question_ids[idx] = qid
        image_ids[idx] = image_id
        image_names[idx] = f"{qid}_{image_id}.png"
        mask_pixel_counts[idx] = int(has_mask[idx].sum())

    metadata["compat_source_npz"] = str(args.input_mask_npz)
    metadata["compat_mode"] = "old_package_like_fields"
    metadata["image_name_format"] = "<question_id>_<image_id>.png"
    metadata["question_ids_definition"] = "explicit question ids aligned with rows"
    metadata["mask_pixel_counts_definition"] = "count of active visual patches, not raw pixel count"
    metadata["matched_instance_counts_definition"] = "set to 1 for SAM3 union masks"

    args.output_mask_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_mask_npz,
        metadata_json=np.array(json.dumps(metadata, ensure_ascii=True)),
        image_names=image_names,
        question_ids=question_ids,
        image_ids=image_ids,
        original_widths=analysis["original_widths"].astype(np.int32),
        original_heights=analysis["original_heights"].astype(np.int32),
        padded_sides=analysis["padded_sides"].astype(np.int32),
        pad_tops=analysis["pad_tops"].astype(np.int32),
        pad_lefts=analysis["pad_lefts"].astype(np.int32),
        mask_pixel_counts=mask_pixel_counts,
        matched_instance_counts=matched_instance_counts,
        coverage_ratio=analysis["coverage_ratio"].astype(np.float32),
        has_mask=has_mask,
    )

    print(f"wrote json: {args.output_json}")
    print(f"wrote npz: {args.output_mask_npz}")
    print(f"rows: {len(rows)} masked_npz_rows: {num_rows}")


if __name__ == "__main__":
    main()
