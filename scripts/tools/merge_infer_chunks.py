#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Merge eval_test_raw_gemma3.py chunk JSON files in original source order.")
    parser.add_argument("--source-data", required=True, type=Path)
    parser.add_argument("--chunk-dir", required=True, type=Path)
    parser.add_argument("--num-chunks", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_json_array(path: Path):
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Expected JSON array: {path}")
    return rows


def main():
    args = parse_args()
    source_rows = load_json_array(args.source_data)
    source_stem = args.source_data.stem

    by_qid = {}
    for idx in range(args.num_chunks):
        chunk_path = args.chunk_dir / f"{source_stem}.chunk{idx}of{args.num_chunks}.json"
        rows = load_json_array(chunk_path)
        for row in rows:
            qid = int(row["question_id"])
            if qid in by_qid:
                raise ValueError(f"Duplicate question_id={qid} from {chunk_path}")
            by_qid[qid] = row

    merged = []
    missing = []
    for row in source_rows:
        qid = int(row["question_id"])
        item = by_qid.get(qid)
        if item is None:
            missing.append(qid)
        else:
            merged.append(item)

    if missing:
        raise RuntimeError(f"Missing {len(missing)} predictions; first few: {missing[:10]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()

