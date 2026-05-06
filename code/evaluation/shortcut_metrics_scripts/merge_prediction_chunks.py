#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge llava.eval.model_vqa_loader chunk outputs into one JSONL file."
    )
    parser.add_argument("--chunk-dir", required=True, help="Directory containing chunk*.jsonl files.")
    parser.add_argument("--out", required=True, help="Merged output JSONL path.")
    parser.add_argument(
        "--pattern",
        default="chunk*.jsonl",
        help="Glob pattern for chunk files. Default: chunk*.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunk_dir = Path(args.chunk_dir)
    out = Path(args.out)

    rows = []
    chunk_files = sorted(chunk_dir.glob(args.pattern))
    if not chunk_files:
        raise FileNotFoundError(f"No files matched {args.pattern} in {chunk_dir}")

    for chunk in chunk_files:
        with chunk.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    rows.sort(key=lambda x: x["question_id"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Merged {len(rows)} rows from {len(chunk_files)} files into {out}")


if __name__ == "__main__":
    main()
