#!/usr/bin/env python3
"""Export a compact VQAv2 train2014 JSON used as no-mask stage-2 data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUESTIONS = (
    REPO_ROOT / "data/detect-shortcuts/data/vqa2/v2_OpenEnded_mscoco_train2014_questions.json"
)
DEFAULT_ANNOTATIONS = (
    REPO_ROOT / "data/detect-shortcuts/data/vqa2/v2_mscoco_train2014_annotations.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "data/stage2/vqa_train2014.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    questions_payload = load_json(args.questions)
    annotations_payload = load_json(args.annotations)

    questions = questions_payload.get("questions", [])
    annotations = annotations_payload.get("annotations", [])
    ann_by_qid = {int(row["question_id"]): row for row in annotations}
    if len(ann_by_qid) != len(annotations):
        raise ValueError("Duplicate question_id in annotations.")

    rows = []
    missing = []
    for question_row in questions:
        qid = int(question_row["question_id"])
        ann = ann_by_qid.get(qid)
        if ann is None:
            missing.append(qid)
            continue
        rows.append(
            {
                "question_id": qid,
                "image_id": int(question_row["image_id"]),
                "question": str(question_row["question"]).strip(),
                "answer": str(ann.get("multiple_choice_answer", "")).strip(),
                "answer_type": str(ann.get("answer_type", "other")).strip() or "other",
            }
        )

    if missing:
        raise ValueError(f"Missing annotations for {len(missing)} questions; first ids: {missing[:20]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)

    print(
        json.dumps(
            {
                "questions": str(args.questions),
                "annotations": str(args.annotations),
                "output": str(args.output),
                "rows": len(rows),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
