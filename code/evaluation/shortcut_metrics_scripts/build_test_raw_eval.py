#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert test_raw.json into the JSONL format expected by llava.eval.model_vqa_loader."
    )
    parser.add_argument(
        "--src",
        default=None,
        help="Source test_raw JSON path. Defaults to <bundle>/data/test_raw.json.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSONL path. Defaults to <bundle>/data/test_raw_eval.jsonl.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle_root = Path(__file__).resolve().parents[1]
    src = Path(args.src) if args.src else bundle_root / "data" / "test_raw.json"
    out = Path(args.out) if args.out else bundle_root / "data" / "test_raw_eval.jsonl"

    with src.open("r", encoding="utf-8") as f:
        data = json.load(f)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for item in data:
            rec = {
                "question_id": item["question_id"],
                "image": f"COCO_train2014_{int(item['image_id']):012d}.jpg",
                "text": item["question"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(data)} rows to {out}")


if __name__ == "__main__":
    main()
