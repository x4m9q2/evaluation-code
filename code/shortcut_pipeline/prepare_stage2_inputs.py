#!/usr/bin/env python3
"""Prepare stage-2 shortcut candidates and SAM3 inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPE_ROOT = REPO_ROOT / "data" / "shortcut_pipeline"
DEFAULT_QUESTIONS = (
    REPO_ROOT
    / "data"
    / "detect-shortcuts"
    / "data"
    / "vqa2"
    / "v2_OpenEnded_mscoco_train2014_questions.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--merged-json",
        default=str(PIPE_ROOT / "gqa_merged_output_with_answer_type.json"),
        help="Merged stage-1 shortcut output.",
    )
    parser.add_argument(
        "--questions-json",
        default=str(DEFAULT_QUESTIONS),
        help="Original VQA questions JSON used to recover question text.",
    )
    parser.add_argument(
        "--output-json",
        default=str(PIPE_ROOT / "cross_modality_qa_input.json"),
        help="Filtered stage-2 candidate JSON.",
    )
    parser.add_argument(
        "--qa-jsonl",
        default=str(PIPE_ROOT / "cross_modality_qa_questions.jsonl"),
        help="SAM3 QA JSONL with question text and image filename.",
    )
    parser.add_argument(
        "--mapping-json",
        default=str(PIPE_ROOT / "cross_modality_qa_mapping.json"),
        help="SAM3 mapping JSON containing visual_cues per question.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Maximum number of eligible samples to keep; <= 0 keeps all.",
    )
    return parser.parse_args()


def load_results(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("results", [])
    if not isinstance(rows, list):
        raise RuntimeError(f"{path} does not contain a 'results' list.")
    return rows


def load_questions(path: Path) -> Dict[int, Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("questions", [])
    if not isinstance(rows, list):
        raise RuntimeError(f"{path} does not contain a 'questions' list.")
    return {int(row["question_id"]): row for row in rows}


def normalize_tokens(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    normalized: List[str] = []
    for value in values:
        text = str(value).strip()
        if text:
            normalized.append(text)
    return normalized


def image_name_for_id(image_id: int) -> str:
    return f"COCO_train2014_{image_id:012d}.jpg"


def main() -> None:
    args = parse_args()
    merged_path = Path(args.merged_json).resolve()
    questions_path = Path(args.questions_json).resolve()
    output_json = Path(args.output_json).resolve()
    qa_jsonl = Path(args.qa_jsonl).resolve()
    mapping_json = Path(args.mapping_json).resolve()

    if not merged_path.exists():
        raise FileNotFoundError(f"Merged JSON not found: {merged_path}")
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions JSON not found: {questions_path}")

    merged_rows = load_results(merged_path)
    question_map = load_questions(questions_path)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    qa_jsonl.parent.mkdir(parents=True, exist_ok=True)
    mapping_json.parent.mkdir(parents=True, exist_ok=True)

    selected_rows: List[Dict[str, Any]] = []
    qa_rows: List[Dict[str, Any]] = []
    mapping_rows: List[Dict[str, Any]] = []

    skipped_ineligible = 0
    skipped_missing_text = 0
    skipped_missing_visual = 0
    skipped_missing_question = 0

    for row in merged_rows:
        qid = int(row["question_id"])
        image_id = int(row["image_id"])
        text_keywords = normalize_tokens(row.get("text_keywords", []))
        visual_cues = normalize_tokens(row.get("visual_cues", []))

        has_text_keywords = bool(text_keywords)
        has_visual_cues = bool(visual_cues)
        if not has_text_keywords or not has_visual_cues:
            skipped_ineligible += 1
            if not has_text_keywords:
                skipped_missing_text += 1
            if not has_visual_cues:
                skipped_missing_visual += 1
            continue

        question_row = question_map.get(qid)
        if question_row is None:
            skipped_missing_question += 1
            continue

        question = str(question_row.get("question", "")).strip()
        image_name = image_name_for_id(image_id)
        answer = str(row.get("answer", "")).strip()
        answer_type = row.get("answer_type")

        item = {
            "question_id": qid,
            "image_id": image_id,
            "question": question,
            "image": image_name,
            "text_keywords": text_keywords,
            "visual_cues": visual_cues,
            "answer": answer,
            "answer_type": answer_type,
        }
        selected_rows.append(item)
        qa_rows.append(
            {
                "question_id": qid,
                "image_id": image_id,
                "image": image_name,
                "text": question,
            }
        )
        mapping_rows.append(item)

        if args.limit > 0 and len(selected_rows) >= args.limit:
            break

    output_json.write_text(
        json.dumps({"results": selected_rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with qa_jsonl.open("w", encoding="utf-8") as fh:
        for row in qa_rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")
    mapping_json.write_text(
        json.dumps({"results": mapping_rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {
        "merged_json": str(merged_path),
        "questions_json": str(questions_path),
        "output_json": str(output_json),
        "qa_jsonl": str(qa_jsonl),
        "mapping_json": str(mapping_json),
        "total_merged_rows": len(merged_rows),
        "selected_rows": len(selected_rows),
        "skipped_ineligible": skipped_ineligible,
        "skipped_missing_text_keywords": skipped_missing_text,
        "skipped_missing_visual_cues": skipped_missing_visual,
        "skipped_missing_question_text": skipped_missing_question,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
