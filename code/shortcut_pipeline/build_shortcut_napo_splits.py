#!/usr/bin/env python3
"""Build 8:1:1 Gemma NaPO preference splits from shortcut stage-2 outputs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from build_vqa_v2_cmsv_splits import load_generated_rows, load_stage2_inputs


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPE_ROOT = REPO_ROOT / "data" / "shortcut_pipeline"
DEFAULT_INPUT_JSON = PIPE_ROOT / "cross_modality_qa_input.json"
DEFAULT_OUTPUT_JSONL = PIPE_ROOT / "batch_outputs" / "cross_modality_qa_responses.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "napo" / "shortcut_generated_vqa"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-json",
        default=str(DEFAULT_INPUT_JSON),
        help="Stage-2 candidate JSON produced by prepare_stage2_inputs.py.",
    )
    parser.add_argument(
        "--batch-output-jsonl",
        default=str(DEFAULT_OUTPUT_JSONL),
        help="Responses JSONL produced by submit_batch_requests.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write train.json / val.json / test.json.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed used before the deterministic 8:1:1 split.",
    )
    parser.add_argument(
        "--question-source",
        choices=("generated", "original"),
        default="generated",
        help="Question text used for the preference prompt.",
    )
    parser.add_argument(
        "--allow-missing-generated-answer",
        action="store_true",
        help="Keep rows even if the parsed model answer is empty.",
    )
    return parser.parse_args()


def compute_split_sizes(total: int, weights: Tuple[int, int, int] = (8, 1, 1)) -> Tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0

    denominator = sum(weights)
    raw_counts = [total * weight / denominator for weight in weights]
    split_sizes = [int(value) for value in raw_counts]
    remainder = total - sum(split_sizes)

    fractional_order = sorted(
        range(len(weights)),
        key=lambda idx: (raw_counts[idx] - split_sizes[idx], weights[idx]),
        reverse=True,
    )
    for idx in fractional_order[:remainder]:
        split_sizes[idx] += 1

    return split_sizes[0], split_sizes[1], split_sizes[2]


def split_rows(rows: List[Dict[str, Any]], seed: int) -> Dict[str, List[Dict[str, Any]]]:
    shuffled = list(rows)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    train_size, val_size, test_size = compute_split_sizes(len(shuffled))
    train_end = train_size
    val_end = train_end + val_size
    test_end = val_end + test_size

    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:test_end],
    }


def build_preference_rows(rows: List[Dict[str, Any]], question_source: str) -> List[Dict[str, Any]]:
    built: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        question = str(row.get(f"{question_source}_question", "")).strip()
        generated_answer = str(row.get("generated_answer", "")).strip()
        original_answer = str(row.get("original_answer", "")).strip()
        if not question or not generated_answer or not original_answer:
            continue
        if generated_answer.lower() == original_answer.lower():
            continue

        image_name = row.get("source_image")
        if not image_name and row.get("image_id") is not None:
            image_name = f"COCO_train2014_{int(row['image_id']):012d}.jpg"

        built.append(
            {
                "idx": idx,
                "question_id": int(row["question_id"]),
                "image_id": int(row["image_id"]),
                "image": image_name,
                "question": question,
                "answer_type": row.get("answer_type"),
                "answer": generated_answer,
                "shortcut_answer": original_answer,
                "positive": {
                    "answer": generated_answer,
                    "source": "generated_answer",
                },
                "negative": {
                    "answer": original_answer,
                    "source": "original_answer",
                },
                "chosen": generated_answer,
                "rejected": original_answer,
                "original_question": row.get("original_question"),
                "original_answer": original_answer,
                "generated_question": row.get("generated_question"),
                "generated_answer": generated_answer,
                "text_keywords": row.get("text_keywords", []),
                "visual_cues": row.get("visual_cues", []),
                "source_image": row.get("source_image"),
                "source_model": row.get("source_model"),
                "source_custom_id": row.get("source_custom_id"),
                "source_response_id": row.get("source_response_id"),
                "origin_dataset": "shortcut_stage2_generated_vqa",
                "question_source": question_source,
            }
        )
    return built


def attach_origin_split(splits: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    updated: Dict[str, List[Dict[str, Any]]] = {}
    for split_name, rows in splits.items():
        split_rows_list: List[Dict[str, Any]] = []
        for row in rows:
            updated_row = dict(row)
            updated_row["origin_split"] = split_name
            split_rows_list.append(updated_row)
        updated[split_name] = split_rows_list
    return updated


def write_splits(output_dir: Path, splits: Dict[str, List[Dict[str, Any]]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in splits.items():
        out_path = output_dir / f"{split_name}.json"
        out_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    args = parse_args()
    input_json = Path(args.input_json).resolve()
    batch_output_jsonl = Path(args.batch_output_jsonl).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_json.exists():
        raise FileNotFoundError(f"Stage-2 input JSON not found: {input_json}")
    if not batch_output_jsonl.exists():
        raise FileNotFoundError(f"Batch output JSONL not found: {batch_output_jsonl}")

    source_rows = load_stage2_inputs(input_json)
    generated_rows, stats = load_generated_rows(
        path=batch_output_jsonl,
        source_rows=source_rows,
        require_status_200=True,
        allow_missing_generated_answer=args.allow_missing_generated_answer,
    )
    same_answer_rows = sum(
        1
        for row in generated_rows
        if str(row.get("generated_answer", "")).strip().lower()
        == str(row.get("original_answer", "")).strip().lower()
    )
    preference_rows = build_preference_rows(generated_rows, question_source=args.question_source)
    splits = attach_origin_split(split_rows(preference_rows, seed=args.seed))
    write_splits(output_dir, splits)

    summary = {
        "input_json": str(input_json),
        "batch_output_jsonl": str(batch_output_jsonl),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "question_source": args.question_source,
        **stats,
        "same_answer_rows_filtered": same_answer_rows,
        "preference_rows": len(preference_rows),
        "train_size": len(splits["train"]),
        "val_size": len(splits["val"]),
        "test_size": len(splits["test"]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
