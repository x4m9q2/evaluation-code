import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter train samples whose question key objects are strongly occluded by mask."
    )
    parser.add_argument(
        "--train-json",
        default="/path/to/sage_repro_bundle/train_raw.json",
        help="Original train JSON.",
    )
    parser.add_argument(
        "--drop-json",
        default="/path/to/sage_repro_bundle/train_raw_llava_likely_unanswerable.json",
        help="Samples to drop from training.",
    )
    parser.add_argument(
        "--output-json",
        default="/path/to/sage_repro_bundle/train_raw_filtered_drop_key_object_occluded.json",
        help="Filtered train JSON path.",
    )
    parser.add_argument(
        "--dropped-json",
        default="/path/to/sage_repro_bundle/train_raw_dropped_key_object_occluded.json",
        help="Path to save the actually dropped rows.",
    )
    parser.add_argument(
        "--report-json",
        default="/path/to/sage_repro_bundle/train_raw_filtered_drop_key_object_occluded.report.json",
        help="Report JSON path.",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    train_path = Path(args.train_json)
    drop_path = Path(args.drop_json)
    output_path = Path(args.output_json)
    dropped_path = Path(args.dropped_json)
    report_path = Path(args.report_json)

    train = load_json(train_path)
    drop_rows = load_json(drop_path)
    drop_qids = {int(row["question_id"]) for row in drop_rows}

    filtered = []
    dropped = []
    for row in train:
        qid = int(row["question_id"])
        if qid in drop_qids:
            dropped.append(row)
        else:
            filtered.append(row)

    report = {
        "train_json": str(train_path),
        "drop_json": str(drop_path),
        "output_json": str(output_path),
        "dropped_json": str(dropped_path),
        "original_count": len(train),
        "drop_candidate_count": len(drop_rows),
        "actually_dropped_count": len(dropped),
        "filtered_count": len(filtered),
        "drop_rate": (len(dropped) / len(train)) if train else 0.0,
    }

    output_path.write_text(json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8")
    dropped_path.write_text(json.dumps(dropped, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
