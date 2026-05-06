#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from modelscope import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Qwen3.5-9B from ModelScope into a stable local path."
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--revision", default="master")
    parser.add_argument("--cache-dir", default="models/modelscope_cache")
    parser.add_argument("--local-dir", default="models/Qwen3.5-9B")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    Path(args.local_dir).mkdir(parents=True, exist_ok=True)

    model_dir = snapshot_download(
        model_id=args.model_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_dir=args.local_dir,
        local_files_only=args.local_files_only,
        max_workers=args.max_workers,
    )
    print(model_dir)


if __name__ == "__main__":
    main()
