#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_GQA_PATH = Path("/path/to/sage_repro_bundle/data2/GQA/test.jsonl")
DEFAULT_VG_PATH = Path("/path/to/sage_repro_bundle/data2/vg/test.jsonl")
DEFAULT_GQA_IMAGE_ROOT = Path("/path/to/sage_repro_bundle/playground/data/gqa/images")
DEFAULT_VG_IMAGE_ROOTS = (
    Path("/path/to/sage_repro_bundle/playground/data/vg/VG_100K"),
    Path("/path/to/sage_repro_bundle/playground/data/vg/VG_100K_2"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build merged data2 evaluation inputs for LLaVA/Qwen inference and xVerify."
    )
    parser.add_argument("--gqa-path", type=Path, default=DEFAULT_GQA_PATH)
    parser.add_argument("--vg-path", type=Path, default=DEFAULT_VG_PATH)
    parser.add_argument(
        "--question-out",
        type=Path,
        required=True,
        help="Output JSONL with question_id/image/text fields for model inference.",
    )
    parser.add_argument(
        "--ref-out",
        type=Path,
        required=True,
        help="Output JSON file with question_id/answer/answer_type for xVerify conversion.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_image_path(dataset: str, row: dict) -> tuple[Path, str]:
    masked_path = Path(str(row.get("image_path", "")))
    if masked_path.is_file():
        return masked_path.resolve(), "masked"

    image_id = int(row["image_id"])
    if dataset == "GQA":
        original_path = DEFAULT_GQA_IMAGE_ROOT / f"{image_id}.jpg"
        if original_path.is_file():
            return original_path.resolve(), "original"
    elif dataset == "vg":
        for root in DEFAULT_VG_IMAGE_ROOTS:
            original_path = root / f"{image_id}.jpg"
            if original_path.is_file():
                return original_path.resolve(), "original"

    raise FileNotFoundError(
        f"Could not resolve image for dataset={dataset}, question_id={row.get('question_id')}, image_id={image_id}"
    )


def convert_rows(dataset: str, rows: list[dict]) -> tuple[list[dict], list[dict], dict[str, int]]:
    questions = []
    refs = []
    stats = {"masked": 0, "original": 0}

    for row in rows:
        question_id = int(row["question_id"])
        question = str(row["generated_question"]).strip()
        answer = str(row["generated_answer"]).strip()
        image_path, image_source = resolve_image_path(dataset, row)
        stats[image_source] += 1

        questions.append(
            {
                "question_id": question_id,
                "image": str(image_path),
                "text": question,
            }
        )
        refs.append(
            {
                "question_id": question_id,
                "answer": answer,
                "answer_type": row.get("answer_type", ""),
                "dataset": dataset,
                "image": str(image_path),
                "question": question,
            }
        )

    return questions, refs, stats


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    gqa_rows = load_jsonl(args.gqa_path)
    vg_rows = load_jsonl(args.vg_path)

    questions = []
    refs = []
    seen_question_ids = set()
    summary = {}

    for dataset, rows in (("GQA", gqa_rows), ("vg", vg_rows)):
        dataset_questions, dataset_refs, dataset_stats = convert_rows(dataset, rows)
        duplicate_ids = {row["question_id"] for row in dataset_questions if row["question_id"] in seen_question_ids}
        if duplicate_ids:
            raise ValueError(f"Found duplicated question_id values in merged data: {sorted(list(duplicate_ids))[:10]}")
        seen_question_ids.update(row["question_id"] for row in dataset_questions)
        questions.extend(dataset_questions)
        refs.extend(dataset_refs)
        summary[dataset] = {
            "count": len(rows),
            "masked_images": dataset_stats["masked"],
            "original_images": dataset_stats["original"],
        }

    questions.sort(key=lambda x: x["question_id"])
    refs.sort(key=lambda x: x["question_id"])

    save_jsonl(questions, args.question_out)
    save_json(refs, args.ref_out)

    print(
        json.dumps(
            {
                "question_out": str(args.question_out),
                "ref_out": str(args.ref_out),
                "total": len(questions),
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
