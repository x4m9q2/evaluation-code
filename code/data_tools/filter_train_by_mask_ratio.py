import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter a VQA-style training JSON by mask area ratio using patch-mask analysis output."
    )
    parser.add_argument("--input-json", required=True, help="Path to the source training JSON.")
    parser.add_argument(
        "--mask-analysis-path",
        required=True,
        help="Path to patch_mask_analysis_output_mask_llava_pad336_patch14.npz.",
    )
    parser.add_argument("--output-json", required=True, help="Path to save the filtered JSON.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Drop samples whose mask area ratio is strictly greater than this threshold.",
    )
    parser.add_argument(
        "--stats-json",
        default=None,
        help="Optional path to save filtering statistics as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_json = Path(args.input_json)
    mask_analysis_path = Path(args.mask_analysis_path)
    output_json = Path(args.output_json)

    with input_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    analysis = np.load(mask_analysis_path, allow_pickle=True)
    qids = analysis["question_ids"].astype(np.int64)
    mask_pixels = analysis["mask_pixel_counts"].astype(np.float64)
    areas = analysis["original_widths"].astype(np.float64) * analysis["original_heights"].astype(np.float64)
    ratios = mask_pixels / np.maximum(areas, 1.0)
    ratio_by_qid = {int(qid): float(ratio) for qid, ratio in zip(qids, ratios)}

    kept = []
    removed = []
    missing = []
    for item in data:
        qid = int(item["question_id"])
        ratio = ratio_by_qid.get(qid)
        if ratio is None:
            kept.append(item)
            missing.append(qid)
            continue
        if ratio > args.threshold:
            removed.append({"question_id": qid, "mask_ratio": ratio})
            continue
        kept.append(item)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    stats = {
        "input_json": str(input_json),
        "mask_analysis_path": str(mask_analysis_path),
        "threshold": args.threshold,
        "input_samples": len(data),
        "kept_samples": len(kept),
        "removed_samples": len(removed),
        "missing_in_analysis": len(missing),
        "removed_ratio": len(removed) / max(len(data), 1),
        "top_removed_examples": sorted(removed, key=lambda x: x["mask_ratio"], reverse=True)[:20],
    }

    if args.stats_json:
        stats_path = Path(args.stats_json)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
