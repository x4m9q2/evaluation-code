#!/usr/bin/env python3
"""Merge shortcut matches and rules into a GQA-style output JSON.

This keeps the `gqa_merged_output.json` shape:

    {
      "results": [
        {
          "question_id": ...,
          "image_id": ...,
          "text_keywords": [...],
          "visual_cues": [...],
          "answer": "...",
          "answer_type": "..."
        }
      ]
    }

The `answer_type` is taken from the question record if it already exists.
Otherwise the script falls back to the paired VQA annotations file, keyed by
`question_id`. This makes the script work both for the standard VQA files and
for any custom question JSON that already carries `answer_type`.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPE_ROOT = REPO_ROOT / "data" / "shortcut_pipeline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rules-json",
        default=str(PIPE_ROOT / "rules" / "rules.json"),
        help="Path to rules.json produced by the shortcut mining stage.",
    )
    parser.add_argument(
        "--matches-json",
        default=str(PIPE_ROOT / "shortcuts_matches.json"),
        help="Path to shortcuts_matches.json produced by the CUDA matcher.",
    )
    parser.add_argument(
        "--questions-json",
        default=str(REPO_ROOT / "data" / "detect-shortcuts" / "data" / "vqa2" / "v2_OpenEnded_mscoco_train2014_questions.json"),
        help="Path to the VQA questions file or the merged question JSON.",
    )
    parser.add_argument(
        "--annotations-json",
        default=str(REPO_ROOT / "data" / "detect-shortcuts" / "data" / "vqa2" / "v2_mscoco_train2014_annotations.json"),
        help="Path to the paired VQA annotations file used to recover answer_type.",
    )
    parser.add_argument(
        "--output-json",
        default=str(PIPE_ROOT / "gqa_merged_output_with_answer_type.json"),
        help="Destination JSON file.",
    )
    parser.add_argument(
        "--include-unmatched",
        action="store_true",
        help="Keep unmatched rows with empty rule fields instead of skipping them.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def as_list(payload: Any, key: str) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        items = payload.get(key, [])
    else:
        items = payload
    if not isinstance(items, list):
        raise RuntimeError(f"Expected a list under '{key}'.")
    return items


def normalize_id(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def build_question_index(questions: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for question in questions:
        qid = question.get("question_id", question.get("id"))
        if qid is None:
            continue
        indexed[normalize_id(qid)] = dict(question)
    return indexed


def build_annotation_index(annotations: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for annotation in annotations:
        qid = annotation.get("question_id")
        if qid is None:
            continue
        indexed[normalize_id(qid)] = dict(annotation)
    return indexed


def build_rules_index(rules_payload: Any) -> Dict[str, Dict[str, Any]]:
    rules = as_list(rules_payload, "rules")
    indexed: Dict[str, Dict[str, Any]] = {}
    for idx, rule in enumerate(rules, start=1):
        record = dict(rule)
        rule_id = record.get("rule_id", idx)
        indexed[normalize_id(rule_id)] = record
    return indexed


def build_matches_list(matches_payload: Any) -> List[Dict[str, Any]]:
    return as_list(matches_payload, "results")


def extract_answer_type(
    question: Optional[Mapping[str, Any]],
    annotation: Optional[Mapping[str, Any]],
) -> str:
    if question:
        answer_type = str(question.get("answer_type", "")).strip().lower()
        if answer_type:
            return answer_type
    if annotation:
        answer_type = str(annotation.get("answer_type", "")).strip().lower()
        if answer_type:
            return answer_type
    return "unknown"


def merge_records(
    matches: List[Dict[str, Any]],
    rules_by_id: Dict[str, Dict[str, Any]],
    questions_by_id: Dict[str, Dict[str, Any]],
    annotations_by_id: Dict[str, Dict[str, Any]],
    include_unmatched: bool,
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []

    for match in matches:
        rule_id = normalize_id(match.get("rule_id", "0"))
        if rule_id == "0" and not include_unmatched:
            continue

        question_id = match.get("question_id")
        question_key = normalize_id(question_id)
        question = questions_by_id.get(question_key)
        annotation = annotations_by_id.get(question_key)
        rule = rules_by_id.get(rule_id)

        if not rule and not include_unmatched:
            continue

        image_id = match.get("image_id")
        if image_id is None and question:
            image_id = question.get("image_id")

        merged.append(
            {
                "question_id": question_id,
                "image_id": image_id,
                "text_keywords": list(rule.get("text_keywords", [])) if rule else [],
                "visual_cues": list(rule.get("visual_cues", [])) if rule else [],
                "answer": rule.get("answer", "") if rule else "",
                "answer_type": extract_answer_type(question, annotation),
            }
        )

    return merged


def main() -> None:
    args = parse_args()

    rules_path = Path(args.rules_json).resolve()
    matches_path = Path(args.matches_json).resolve()
    questions_path = Path(args.questions_json).resolve()
    annotations_path = Path(args.annotations_json).resolve()
    output_path = Path(args.output_json).resolve()

    for path, label in [
        (rules_path, "rules"),
        (matches_path, "matches"),
        (questions_path, "questions"),
        (annotations_path, "annotations"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} file not found: {path}")

    rules_payload = load_json(rules_path)
    matches_payload = load_json(matches_path)
    questions_payload = load_json(questions_path)
    annotations_payload = load_json(annotations_path)

    rules_by_id = build_rules_index(rules_payload)
    matches = build_matches_list(matches_payload)
    if isinstance(questions_payload, dict) and "questions" in questions_payload:
        questions_source = questions_payload["questions"]
    else:
        questions_source = questions_payload
    questions_by_id = build_question_index(as_list(questions_source, "questions")) if isinstance(questions_source, dict) else build_question_index(questions_source)
    annotations_by_id = build_annotation_index(as_list(annotations_payload, "annotations"))

    merged = merge_records(
        matches=matches,
        rules_by_id=rules_by_id,
        questions_by_id=questions_by_id,
        annotations_by_id=annotations_by_id,
        include_unmatched=args.include_unmatched,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump({"results": merged}, fh, ensure_ascii=False)

    print(f"Wrote {len(merged):,} merged rows to {output_path}")


if __name__ == "__main__":
    main()
