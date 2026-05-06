#!/usr/bin/env python3
"""One-shot driver for stage-2 shortcut mask building and request generation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code" / "shortcut_pipeline"
PIPE_ROOT = REPO_ROOT / "data" / "shortcut_pipeline"

DEFAULT_MERGED = PIPE_ROOT / "gqa_merged_output_with_answer_type.json"
DEFAULT_QUESTIONS = (
    REPO_ROOT
    / "data"
    / "detect-shortcuts"
    / "data"
    / "vqa2"
    / "v2_OpenEnded_mscoco_train2014_questions.json"
)
DEFAULT_STAGE2_INPUT = PIPE_ROOT / "cross_modality_qa_input.json"
DEFAULT_QA_JSONL = PIPE_ROOT / "cross_modality_qa_questions.jsonl"
DEFAULT_MAPPING = PIPE_ROOT / "cross_modality_qa_mapping.json"
DEFAULT_UNION_MASK_ROOT = PIPE_ROOT / "union_mask"
DEFAULT_OUTPUT_MASK_ROOT = PIPE_ROOT / "output_mask"
DEFAULT_OUTPUT_JSONL = PIPE_ROOT / "batch_inputs" / "cross_modality_qa_requests.jsonl"
DEFAULT_IMAGE_ROOT = REPO_ROOT / "data" / "images" / "coco" / "train2014"
DEFAULT_SAM3_CHECKPOINT = REPO_ROOT / "models" / "sam3_ckpt" / "sam3.pt"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-json", default=str(DEFAULT_MERGED))
    parser.add_argument("--questions-json", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--input-json", default=str(DEFAULT_STAGE2_INPUT))
    parser.add_argument("--qa-jsonl", default=str(DEFAULT_QA_JSONL))
    parser.add_argument("--mapping-json", default=str(DEFAULT_MAPPING))
    parser.add_argument("--union-mask-root", default=str(DEFAULT_UNION_MASK_ROOT))
    parser.add_argument("--mask-root", default=str(DEFAULT_OUTPUT_MASK_ROOT))
    parser.add_argument("--output-jsonl", default=str(DEFAULT_OUTPUT_JSONL))
    parser.add_argument("--image-root", dest="image_roots", action="append", default=None)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--max-output-tokens", type=int, default=400)
    parser.add_argument("--sam3-checkpoint-path", default=str(DEFAULT_SAM3_CHECKPOINT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--prepare-masks-only",
        action="store_true",
        help="Prepare filtered stage-2 inputs and masked images only; skip request generation.",
    )
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    image_roots = [Path(p) for p in (args.image_roots or [DEFAULT_IMAGE_ROOT])]

    merged_json = Path(args.merged_json)
    questions_json = Path(args.questions_json)
    input_json = Path(args.input_json)
    qa_jsonl = Path(args.qa_jsonl)
    mapping_json = Path(args.mapping_json)
    union_mask_root = Path(args.union_mask_root)
    mask_root = Path(args.mask_root)
    output_jsonl = Path(args.output_jsonl)

    merged_json = merged_json.resolve()
    questions_json = questions_json.resolve()
    input_json = input_json.resolve()
    qa_jsonl = qa_jsonl.resolve()
    mapping_json = mapping_json.resolve()
    union_mask_root = union_mask_root.resolve()
    mask_root = mask_root.resolve()
    output_jsonl = output_jsonl.resolve()
    image_roots = [path.resolve() for path in image_roots]

    run(
        [
            sys.executable,
            str(CODE_ROOT / "prepare_stage2_inputs.py"),
            "--merged-json",
            str(merged_json),
            "--questions-json",
            str(questions_json),
            "--output-json",
            str(input_json),
            "--qa-jsonl",
            str(qa_jsonl),
            "--mapping-json",
            str(mapping_json),
            "--limit",
            str(args.limit),
        ]
    )

    mask_cmd = [
        sys.executable,
        str(REPO_ROOT / "code" / "sam3" / "scripts" / "generate_union_masks_from_mapping.py"),
        "--qa-jsonl",
        str(qa_jsonl),
        "--mapping-json",
        str(mapping_json),
        "--output-dir",
        str(union_mask_root),
        "--batch-size",
        str(args.batch_size),
        "--resolution",
        str(args.resolution),
        "--score-thresh",
        str(args.score_thresh),
        "--checkpoint-path",
        str(Path(args.sam3_checkpoint_path).resolve()),
        "--device",
        args.device,
        "--no-load-from-hf",
        "--num-shards",
        str(args.num_shards),
        "--shard-index",
        str(args.shard_index),
    ]
    for image_root in image_roots:
        mask_cmd.extend(["--image-root", str(image_root)])
    run(mask_cmd)

    apply_cmd = [
        sys.executable,
        str(CODE_ROOT / "apply_union_masks_to_images.py"),
        "--qa-jsonl",
        str(qa_jsonl),
        "--mask-dir",
        str(union_mask_root / "masks"),
        "--output-dir",
        str(mask_root),
    ]
    for image_root in image_roots:
        apply_cmd.extend(["--image-root", str(image_root)])
    run(apply_cmd)

    if args.prepare_masks_only:
        print(f"[done] stage2 input: {input_json}", flush=True)
        print(f"[done] SAM3 union masks: {union_mask_root}", flush=True)
        print(f"[done] masked RGB images: {mask_root}", flush=True)
        return

    request_cmd = [
        sys.executable,
        str(CODE_ROOT / "run_cross_modality_generation.py"),
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
    run(request_cmd)

    print(f"[done] stage2 input: {input_json}", flush=True)
    print(f"[done] SAM3 union masks: {union_mask_root}", flush=True)
    print(f"[done] masked RGB images: {mask_root}", flush=True)
    print(f"[done] batch requests: {output_jsonl}", flush=True)


if __name__ == "__main__":
    main()
