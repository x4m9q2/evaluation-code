#!/usr/bin/env python3
import json
from collections import Counter
from pathlib import Path

import numpy as np


def default_root() -> Path:
    return Path(__file__).resolve().parents[3]


ROOT = default_root()

DATASETS = {
    "gqa": {
        "input_json": ROOT / "data2/GQA/GQA_filtered_sampled_10000_qwenkeep_sam3.json",
        "input_npz": ROOT / "patch_mask_analysis_gqa_sampled10000_qwenkeep_sam3_compat.npz",
        "output_json": ROOT / "data2/GQA/GQA_filtered_sampled_10000_qwenkeep_sam3_nonumber.json",
        "output_npz": ROOT / "patch_mask_analysis_gqa_sampled10000_qwenkeep_sam3_nonumber_compat.npz",
        "summary_json": ROOT / "analysis/gqa_sampled10000_qwen35_filter/qwenkeep_sam3_nonumber_summary.json",
    },
    "vg": {
        "input_json": ROOT / "data2/vg/vg_filtered_sampled_10000_qwenkeep_sam3.json",
        "input_npz": ROOT / "patch_mask_analysis_vg_sampled10000_qwenkeep_sam3_compat.npz",
        "output_json": ROOT / "data2/vg/vg_filtered_sampled_10000_qwenkeep_sam3_nonumber.json",
        "output_npz": ROOT / "patch_mask_analysis_vg_sampled10000_qwenkeep_sam3_nonumber_compat.npz",
        "summary_json": ROOT / "analysis/vg_sampled10000_qwen35_filter/qwenkeep_sam3_nonumber_summary.json",
    },
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    for name, cfg in DATASETS.items():
        rows = load_json(cfg["input_json"])
        kept_rows = [row for row in rows if str(row.get("answer_type", "")) != "number"]
        kept_qids = {int(row["question_id"]) for row in kept_rows}

        analysis = np.load(cfg["input_npz"], allow_pickle=True)
        question_ids = analysis["question_ids"].astype(np.int64)
        keep_mask = np.array([int(qid) in kept_qids for qid in question_ids], dtype=np.bool_)

        cfg["output_json"].parent.mkdir(parents=True, exist_ok=True)
        write_json(cfg["output_json"], kept_rows)

        metadata = json.loads(str(analysis["metadata_json"].item()))
        metadata["nonumber_source_json"] = str(cfg["input_json"])
        metadata["nonumber_source_npz"] = str(cfg["input_npz"])
        metadata["nonumber_rule"] = "drop rows with answer_type == 'number' and keep NPZ rows whose question_id remains in JSON"

        cfg["output_npz"].parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cfg["output_npz"],
            metadata_json=np.array(json.dumps(metadata, ensure_ascii=True)),
            image_names=analysis["image_names"][keep_mask],
            question_ids=analysis["question_ids"][keep_mask],
            image_ids=analysis["image_ids"][keep_mask],
            original_widths=analysis["original_widths"][keep_mask],
            original_heights=analysis["original_heights"][keep_mask],
            padded_sides=analysis["padded_sides"][keep_mask],
            pad_tops=analysis["pad_tops"][keep_mask],
            pad_lefts=analysis["pad_lefts"][keep_mask],
            mask_pixel_counts=analysis["mask_pixel_counts"][keep_mask],
            matched_instance_counts=analysis["matched_instance_counts"][keep_mask],
            coverage_ratio=analysis["coverage_ratio"][keep_mask],
            has_mask=analysis["has_mask"][keep_mask],
        )

        remaining_masked_qids = set(int(qid) for qid in analysis["question_ids"][keep_mask].tolist())
        json_masked_qids = {
            int(row["question_id"])
            for row in kept_rows
            if row.get("mask_supervision") == "sam3_patch_mask"
        }
        if remaining_masked_qids != json_masked_qids:
            raise ValueError(f"{name}: JSON/NPZ masked qids are not aligned.")

        summary = {
            "dataset": name,
            "input_json": str(cfg["input_json"]),
            "input_npz": str(cfg["input_npz"]),
            "output_json": str(cfg["output_json"]),
            "output_npz": str(cfg["output_npz"]),
            "input_total": len(rows),
            "output_total": len(kept_rows),
            "removed_number_total": len(rows) - len(kept_rows),
            "input_answer_type_counts": dict(Counter(str(row.get("answer_type", "")) for row in rows)),
            "output_answer_type_counts": dict(Counter(str(row.get("answer_type", "")) for row in kept_rows)),
            "input_mask_supervision_counts": dict(Counter(str(row.get("mask_supervision", "")) for row in rows)),
            "output_mask_supervision_counts": dict(Counter(str(row.get("mask_supervision", "")) for row in kept_rows)),
            "input_npz_rows": int(question_ids.shape[0]),
            "output_npz_rows": int(keep_mask.sum()),
        }
        write_json(cfg["summary_json"], summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
