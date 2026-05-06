#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge the mask-answerable train_raw subset with shortcut answers from merged_output_rule_mapping.json."
    )
    parser.add_argument(
        "--subset-json",
        type=Path,
        default=Path("/path/to/sage_repro_bundle/train_raw_filtered_drop_key_object_occluded_and_borderline.json"),
    )
    parser.add_argument(
        "--mapping-json",
        type=Path,
        default=Path("/path/to/sage_repro_bundle/merged_output_rule_mapping.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("/path/to/sage_repro_bundle/test_data/train_raw_mask_answerable_with_shortcut_answer.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subset_rows = json.loads(args.subset_json.read_text())
    mapping_blob = json.loads(args.mapping_json.read_text())
    mapping_rows = mapping_blob["results"] if isinstance(mapping_blob, dict) else mapping_blob
    shortcut_by_qid = {int(row["question_id"]): row["answer"] for row in mapping_rows}

    merged = []
    missing = []
    for row in subset_rows:
        qid = int(row["question_id"])
        shortcut_answer = shortcut_by_qid.get(qid)
        if shortcut_answer is None:
            missing.append(qid)
            continue
        item = dict(row)
        item["shortcut_answer"] = shortcut_answer
        merged.append(item)

    if missing:
        raise ValueError(f"Missing shortcut answers for {len(missing)} question_ids; first few: {missing[:10]}")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n")
    summary = {
        "subset_json": str(args.subset_json),
        "mapping_json": str(args.mapping_json),
        "output_json": str(args.output_json),
        "count": len(merged),
    }
    summary_path = args.output_json.with_suffix(args.output_json.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
