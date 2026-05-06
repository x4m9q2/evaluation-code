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
        "output_json": ROOT / "data2/GQA/GQA_filtered_sampled_10000_qwenkeep_sam3_nonumbermask.json",
        "output_npz": ROOT / "patch_mask_analysis_gqa_sampled10000_qwenkeep_sam3_nonumbermask_compat.npz",
        "summary_json": ROOT / "analysis/gqa_sampled10000_qwen35_filter/qwenkeep_sam3_nonumbermask_summary.json",
    },
    "vg": {
        "input_json": ROOT / "data2/vg/vg_filtered_sampled_10000_qwenkeep_sam3.json",
        "input_npz": ROOT / "patch_mask_analysis_vg_sampled10000_qwenkeep_sam3_compat.npz",
        "output_json": ROOT / "data2/vg/vg_filtered_sampled_10000_qwenkeep_sam3_nonumbermask.json",
        "output_npz": ROOT / "patch_mask_analysis_vg_sampled10000_qwenkeep_sam3_nonumbermask_compat.npz",
        "summary_json": ROOT / "analysis/vg_sampled10000_qwen35_filter/qwenkeep_sam3_nonumbermask_summary.json",
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
        output_rows = []
        number_mask_removed = 0
        for row in rows:
            out = dict(row)
            if str(out.get("answer_type", "")) == "number" and out.get("mask_supervision") == "sam3_patch_mask":
                out["mask_supervision"] = "none"
                out["data_source"] = f"{name}_qwenkeep_number_nomask"
                number_mask_removed += 1
            output_rows.append(out)

        allowed_mask_qids = {
            int(row["question_id"])
            for row in output_rows
            if row.get("mask_supervision") == "sam3_patch_mask"
        }

        analysis = np.load(cfg["input_npz"], allow_pickle=True)
        question_ids = analysis["question_ids"].astype(np.int64)
        keep_mask = np.array([int(qid) in allowed_mask_qids for qid in question_ids], dtype=np.bool_)

        metadata = json.loads(str(analysis["metadata_json"].item()))
        metadata["nonumbermask_source_json"] = str(cfg["input_json"])
        metadata["nonumbermask_source_npz"] = str(cfg["input_npz"])
        metadata["nonumbermask_rule"] = "keep all JSON rows, but set answer_type == 'number' mask_supervision to none and drop those mask rows from NPZ"

        cfg["output_json"].parent.mkdir(parents=True, exist_ok=True)
        write_json(cfg["output_json"], output_rows)

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

        npz_qids = set(int(qid) for qid in analysis["question_ids"][keep_mask].tolist())
        if npz_qids != allowed_mask_qids:
            raise ValueError(f"{name}: JSON/NPZ masked qids are not aligned.")

        summary = {
            "dataset": name,
            "input_json": str(cfg["input_json"]),
            "input_npz": str(cfg["input_npz"]),
            "output_json": str(cfg["output_json"]),
            "output_npz": str(cfg["output_npz"]),
            "json_total": len(output_rows),
            "number_rows_total": sum(str(row.get("answer_type", "")) == "number" for row in output_rows),
            "number_mask_removed": number_mask_removed,
            "input_mask_supervision_counts": dict(Counter(str(row.get("mask_supervision", "")) for row in rows)),
            "output_mask_supervision_counts": dict(Counter(str(row.get("mask_supervision", "")) for row in output_rows)),
            "input_answer_type_x_mask_counts": {
                f"{answer_type}|{mask_supervision}": count
                for (answer_type, mask_supervision), count in Counter(
                    (str(row.get("answer_type", "")), str(row.get("mask_supervision", ""))) for row in rows
                ).items()
            },
            "output_answer_type_x_mask_counts": {
                f"{answer_type}|{mask_supervision}": count
                for (answer_type, mask_supervision), count in Counter(
                    (str(row.get("answer_type", "")), str(row.get("mask_supervision", ""))) for row in output_rows
                ).items()
            },
            "input_npz_rows": int(question_ids.shape[0]),
            "output_npz_rows": int(keep_mask.sum()),
        }
        write_json(cfg["summary_json"], summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
