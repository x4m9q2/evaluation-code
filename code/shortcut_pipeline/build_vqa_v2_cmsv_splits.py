#!/usr/bin/env python3
"""Build VQA v2-CMSV-style train/val/test JSON files from stage-2 outputs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPE_ROOT = REPO_ROOT / "data" / "shortcut_pipeline"
SAGE_AS_ROOT = REPO_ROOT / "data" / "sage_as"
DEFAULT_INPUT_JSON = PIPE_ROOT / "cross_modality_qa_input.json"
DEFAULT_OUTPUT_JSONL = PIPE_ROOT / "batch_outputs" / "cross_modality_qa_responses.jsonl"
DEFAULT_OUTPUT_DIR = SAGE_AS_ROOT / "data" / "vqa_v2_cmsv"


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
        help="Random seed used before the deterministic 9:1:1 split.",
    )
    parser.add_argument(
        "--require-status-200",
        action="store_true",
        default=True,
        help="Keep only response rows with status_code == 200.",
    )
    parser.add_argument(
        "--allow-missing-generated-answer",
        action="store_true",
        help="Keep rows even if the parsed model answer is empty.",
    )
    return parser.parse_args()


def load_stage2_inputs(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("results", [])
    if not isinstance(rows, list):
        raise RuntimeError(f"{path} does not contain a 'results' list.")

    records: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        question_id = str(row.get("question_id", "")).strip()
        if not question_id:
            continue
        records[question_id] = row
    return records


def parse_generated_payload(text: str) -> Dict[str, Any]:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise RuntimeError("generated payload is not a JSON object")
    return payload


def extract_question_id(custom_id: str) -> str:
    token = str(custom_id or "").rsplit("-", 1)[-1].strip()
    if not token:
        raise RuntimeError(f"cannot parse question_id from custom_id={custom_id!r}")
    return token


def load_generated_rows(
    path: Path,
    source_rows: Dict[str, Dict[str, Any]],
    require_status_200: bool,
    allow_missing_generated_answer: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    built_rows: List[Dict[str, Any]] = []
    stats = {
        "total_response_rows": 0,
        "kept_rows": 0,
        "skipped_non_200": 0,
        "skipped_bad_custom_id": 0,
        "skipped_missing_source": 0,
        "skipped_bad_output_json": 0,
        "skipped_missing_generated_question": 0,
        "skipped_missing_generated_answer": 0,
    }

    seen_question_ids = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            stats["total_response_rows"] += 1
            row = json.loads(line)

            if require_status_200 and row.get("status_code") != 200:
                stats["skipped_non_200"] += 1
                continue

            try:
                question_id = extract_question_id(row.get("custom_id"))
            except Exception:
                stats["skipped_bad_custom_id"] += 1
                continue

            source = source_rows.get(question_id)
            if source is None:
                stats["skipped_missing_source"] += 1
                continue

            output_text = str(row.get("output_text", "")).strip()
            try:
                generated = parse_generated_payload(output_text)
            except Exception:
                stats["skipped_bad_output_json"] += 1
                continue

            generated_question = str(generated.get("question", "")).strip()
            generated_answer = str(generated.get("answer", "")).strip()
            if not generated_question:
                stats["skipped_missing_generated_question"] += 1
                continue
            if not generated_answer and not allow_missing_generated_answer:
                stats["skipped_missing_generated_answer"] += 1
                continue

            if question_id in seen_question_ids:
                continue
            seen_question_ids.add(question_id)

            built_rows.append(
                {
                    "question_id": int(source["question_id"]),
                    "question": generated_question,
                    "image_id": int(source["image_id"]),
                    "answer": generated_answer,
                    "answer_type": source.get("answer_type"),
                    "original_question": source.get("question"),
                    "original_answer": source.get("answer"),
                    "generated_question": generated_question,
                    "generated_answer": generated_answer,
                    "text_keywords": source.get("text_keywords", []),
                    "visual_cues": source.get("visual_cues", []),
                    "source_image": source.get("image"),
                    "source_model": row.get("model"),
                    "source_custom_id": row.get("custom_id"),
                    "source_response_id": (row.get("response") or {}).get("id"),
                }
            )
            stats["kept_rows"] += 1

    return built_rows, stats


def compute_split_sizes(total: int) -> Tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0

    train = total * 9 // 11
    val = total // 11
    test = total // 11
    remainder = total - train - val - test

    for name in ("train", "val", "test"):
        if remainder <= 0:
            break
        if name == "train":
            train += 1
        elif name == "val":
            val += 1
        else:
            test += 1
        remainder -= 1

    return train, val, test


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
    built_rows, stats = load_generated_rows(
        path=batch_output_jsonl,
        source_rows=source_rows,
        require_status_200=args.require_status_200,
        allow_missing_generated_answer=args.allow_missing_generated_answer,
    )
    splits = split_rows(built_rows, seed=args.seed)
    write_splits(output_dir, splits)

    summary = {
        "input_json": str(input_json),
        "batch_output_jsonl": str(batch_output_jsonl),
        "output_dir": str(output_dir),
        "seed": args.seed,
        **stats,
        "train_size": len(splits["train"]),
        "val_size": len(splits["val"]),
        "test_size": len(splits["test"]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
