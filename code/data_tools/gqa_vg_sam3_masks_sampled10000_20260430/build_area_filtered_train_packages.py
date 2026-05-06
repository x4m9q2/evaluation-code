#!/usr/bin/env python3
import json
from collections import Counter
from pathlib import Path

import numpy as np


def default_root() -> Path:
    return Path(__file__).resolve().parents[3]


ROOT = default_root()
GQA_IMAGE_ROOT = ROOT / "data/images/gqa/images"
VG_IMAGE_ROOTS = [
    ROOT / "data/images/vg/VG_100K",
    ROOT / "data/images/vg/VG_100K_2",
]

DATASETS = {
    "gqa": {
        "input_json": ROOT / "data2/GQA/GQA_filtered_sampled_10000_qwenkeep_sam3_nonumbermask.json",
        "input_npz": ROOT / "patch_mask_analysis_gqa_sampled10000_qwenkeep_sam3_nonumbermask_compat.npz",
        "image_roots": [GQA_IMAGE_ROOT],
        "out_dir": ROOT / "data2/GQA",
        "npz_prefix": ROOT / "patch_mask_analysis_gqa_sampled10000_qwenkeep_sam3_nonumbermask",
        "summary_dir": ROOT / "analysis/gqa_sampled10000_qwen35_filter",
    },
    "vg": {
        "input_json": ROOT / "data2/vg/vg_filtered_sampled_10000_qwenkeep_sam3_nonumbermask.json",
        "input_npz": ROOT / "patch_mask_analysis_vg_sampled10000_qwenkeep_sam3_nonumbermask_compat.npz",
        "image_roots": VG_IMAGE_ROOTS,
        "out_dir": ROOT / "data2/vg",
        "npz_prefix": ROOT / "patch_mask_analysis_vg_sampled10000_qwenkeep_sam3_nonumbermask",
        "summary_dir": ROOT / "analysis/vg_sampled10000_qwen35_filter",
    },
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def resolve_image_path(image_id: int, roots: list[Path]) -> str:
    for root in roots:
        candidate = root / f"{image_id}.jpg"
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"Could not resolve image_id={image_id} in roots={roots}")


def to_llava_row(dataset: str, row: dict, image_path: str) -> dict:
    question = str(row.get("generated_question") or row.get("question") or row.get("text") or "").strip()
    answer = str(row.get("generated_answer") or row.get("answer") or row.get("original_answer") or "").strip()
    if not question or not answer:
        raise ValueError(f"Missing question/answer for question_id={row.get('question_id')}")
    return {
        "id": f"{dataset}_{int(row['question_id'])}",
        "dataset": dataset,
        "question_id": int(row["question_id"]),
        "image_id": int(row["image_id"]),
        "answer_type": str(row.get("answer_type", "other")),
        "data_source": str(row.get("data_source", "")),
        "mask_supervision": str(row.get("mask_supervision", "")),
        "image": image_path,
        "source_image_path": str(row.get("image_path", "")),
        "visual_cues": row.get("visual_cues", []),
        "conversations": [
            {"from": "human", "value": f"<image>\n{question}"},
            {"from": "gpt", "value": answer},
        ],
    }


def build_for_threshold(dataset: str, cfg: dict, max_active_patch_frac: float):
    tag = f"area001_max{str(max_active_patch_frac).replace('.', 'p')}"
    rows = load_json(cfg["input_json"])
    row_by_qid = {int(row["question_id"]): row for row in rows}
    if len(row_by_qid) != len(rows):
        raise ValueError(f"{dataset}: duplicate question_id")

    analysis = np.load(cfg["input_npz"], allow_pickle=True)
    question_ids = analysis["question_ids"].astype(np.int64)
    has_mask = analysis["has_mask"].astype(np.bool_).reshape(question_ids.shape[0], -1)
    active_frac = has_mask.mean(axis=1)
    active_frac_by_qid = {int(qid): float(active_frac[idx]) for idx, qid in enumerate(question_ids.tolist())}

    keep_npz_mask = (active_frac > 0.0) & (active_frac <= max_active_patch_frac)
    kept_mask_qids = {int(qid) for qid in question_ids[keep_npz_mask].tolist()}

    output_rows = []
    for row in rows:
        qid = int(row["question_id"])
        out = dict(row)
        if out.get("mask_supervision") == "sam3_patch_mask" and qid not in kept_mask_qids:
            out["mask_supervision"] = "none"
            frac = active_frac_by_qid.get(qid, 0.0)
            if frac == 0.0:
                out["data_source"] = f"{dataset}_qwenkeep_empty_mask_nomask"
            elif frac > max_active_patch_frac:
                out["data_source"] = f"{dataset}_qwenkeep_large_mask_nomask"
            else:
                out["data_source"] = f"{dataset}_qwenkeep_area_filtered_nomask"
        image_path = resolve_image_path(int(out["image_id"]), cfg["image_roots"])
        output_rows.append(to_llava_row(dataset, out, image_path))

    json_path = cfg["out_dir"] / f"{dataset}_filtered_sampled_10000_qwenkeep_sam3_nonumbermask_{tag}_llava.json"
    npz_path = Path(f"{cfg['npz_prefix']}_{tag}_compat.npz")
    summary_path = cfg["summary_dir"] / f"qwenkeep_sam3_nonumbermask_{tag}_train_package_summary.json"

    metadata = json.loads(str(analysis["metadata_json"].item()))
    metadata["area_filter_source_json"] = str(cfg["input_json"])
    metadata["area_filter_source_npz"] = str(cfg["input_npz"])
    metadata["area_filter_rule"] = f"keep mask rows with 0 < active_patch_frac <= {max_active_patch_frac}; keep all JSON samples"
    metadata["train_json"] = str(json_path)

    np.savez_compressed(
        npz_path,
        metadata_json=np.array(json.dumps(metadata, ensure_ascii=True)),
        image_names=analysis["image_names"][keep_npz_mask],
        question_ids=analysis["question_ids"][keep_npz_mask],
        image_ids=analysis["image_ids"][keep_npz_mask],
        original_widths=analysis["original_widths"][keep_npz_mask],
        original_heights=analysis["original_heights"][keep_npz_mask],
        padded_sides=analysis["padded_sides"][keep_npz_mask],
        pad_tops=analysis["pad_tops"][keep_npz_mask],
        pad_lefts=analysis["pad_lefts"][keep_npz_mask],
        mask_pixel_counts=analysis["mask_pixel_counts"][keep_npz_mask],
        matched_instance_counts=analysis["matched_instance_counts"][keep_npz_mask],
        coverage_ratio=analysis["coverage_ratio"][keep_npz_mask],
        has_mask=analysis["has_mask"][keep_npz_mask],
    )
    write_json(json_path, output_rows)

    json_masked_qids = {
        int(row["question_id"])
        for row in output_rows
        if row.get("mask_supervision") == "sam3_patch_mask"
    }
    if json_masked_qids != kept_mask_qids:
        raise ValueError(f"{dataset} {tag}: JSON/NPZ masked qids mismatch")

    summary = {
        "dataset": dataset,
        "tag": tag,
        "max_active_patch_frac": max_active_patch_frac,
        "input_json": str(cfg["input_json"]),
        "input_npz": str(cfg["input_npz"]),
        "output_json": str(json_path),
        "output_npz": str(npz_path),
        "json_total": len(output_rows),
        "input_npz_rows": int(question_ids.shape[0]),
        "output_npz_rows": int(keep_npz_mask.sum()),
        "empty_mask_removed": int((active_frac == 0.0).sum()),
        "large_mask_removed": int((active_frac > max_active_patch_frac).sum()),
        "mask_supervision_counts": dict(Counter(row.get("mask_supervision", "") for row in output_rows)),
        "data_source_counts": dict(Counter(row.get("data_source", "") for row in output_rows)),
        "answer_type_x_mask_counts": {
            f"{answer_type}|{mask_supervision}": count
            for (answer_type, mask_supervision), count in Counter(
                (str(row.get("answer_type", "")), str(row.get("mask_supervision", "")))
                for row in output_rows
            ).items()
        },
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    for dataset, cfg in DATASETS.items():
        for threshold in (0.5, 0.7):
            build_for_threshold(dataset, cfg, threshold)


if __name__ == "__main__":
    main()
