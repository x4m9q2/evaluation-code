#!/usr/bin/env python3
"""One-shot driver for stage-2 batch request generation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code" / "shortcut_pipeline"
PIPE_ROOT = REPO_ROOT / "data" / "shortcut_pipeline"
DEFAULT_INPUT = PIPE_ROOT / "cross_modality_qa_input.json"
DEFAULT_MASK = PIPE_ROOT / "output_mask"
DEFAULT_OUTPUT = PIPE_ROOT / "batch_inputs" / "cross_modality_qa_requests.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", default=str(DEFAULT_INPUT))
    parser.add_argument("--mask-root", default=str(DEFAULT_MASK))
    parser.add_argument("--output-jsonl", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--max-output-tokens", type=int, default=400)
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    input_json = Path(args.input_json)
    mask_root = Path(args.mask_root).resolve()
    output_jsonl = Path(args.output_jsonl)

    input_json = input_json.resolve()
    output_jsonl = output_jsonl.resolve()

    if not input_json.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_json}")
    if not mask_root.exists():
        raise FileNotFoundError(f"Mask root not found: {mask_root}")

    cmd = [
        sys.executable,
        str(CODE_ROOT / "prepare_gqa_batch_requests.py"),
        "--input-json",
        str(input_json),
        "--mask-root",
        str(mask_root),
        "--output-jsonl",
        str(output_jsonl),
        "--limit",
        str(args.limit),
        "--model",
        args.model,
        "--max-output-tokens",
        str(args.max_output_tokens),
    ]
    run(cmd)
    print(f"[done] batch requests: {output_jsonl}", flush=True)


if __name__ == "__main__":
    main()
