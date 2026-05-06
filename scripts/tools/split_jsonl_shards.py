#!/usr/bin/env python3
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Split a JSONL file into deterministic contiguous shards.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-shards", required=True, type=int)
    parser.add_argument("--stem", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")

    lines = args.input.read_text(encoding="utf-8").splitlines()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.stem or args.input.stem

    total = len(lines)
    base = total // args.num_shards
    rem = total % args.num_shards
    start = 0
    for shard_id in range(args.num_shards):
        size = base + (1 if shard_id < rem else 0)
        end = start + size
        output_path = args.output_dir / f"{stem}.shard{shard_id}of{args.num_shards}.jsonl"
        with output_path.open("w", encoding="utf-8") as f:
            for line in lines[start:end]:
                if line:
                    f.write(line + "\n")
        start = end
        print(output_path)


if __name__ == "__main__":
    main()
