#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert test_raw JSON array to LLaVA-style JSONL for CausalMM evaluation.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            image_id = int(row["image_id"])
            item = {
                "question_id": int(row["question_id"]),
                "image": f"COCO_train2014_{image_id:012d}.jpg",
                "text": f"<image>\n{row['question']}",
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

