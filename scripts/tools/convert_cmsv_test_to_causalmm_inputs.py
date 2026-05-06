#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a VQA/GQA/VG-CMSV test split into the question JSONL and "
            "answer JSON files used by CausalMM evaluation wrappers."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--dataset", default="auto", choices=["auto", "vqa", "vqa_v2_cmsv", "gqa", "gqa_cmsv", "vg", "vg_cmsv"])
    parser.add_argument("--question-output", required=True, type=Path)
    parser.add_argument("--answer-output", required=True, type=Path)
    parser.add_argument("--image-policy", choices=["original", "image_path"], default="original")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_rows(path: Path) -> List[dict]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected JSON object at {path}:{line_no}")
                rows.append(row)
        return rows

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return data


def canonical_dataset_name(dataset: str) -> str:
    if dataset in {"vqa", "vqa_v2_cmsv"}:
        return "vqa"
    if dataset in {"gqa", "gqa_cmsv"}:
        return "gqa"
    if dataset in {"vg", "vg_cmsv"}:
        return "vg"
    return dataset


def infer_dataset(row: dict) -> Optional[str]:
    image_path = str(row.get("image_path") or row.get("image") or "").lower()
    if "masked_images/gqa" in image_path or "/gqa/" in image_path:
        return "gqa"
    if "masked_images/vg" in image_path or "/vg/" in image_path or "visual_genome" in image_path:
        return "vg"
    if "question" in row and "answer" in row:
        return "vqa"
    return None


def resolve_dataset(dataset: str, row: dict) -> str:
    dataset = canonical_dataset_name(dataset)
    if dataset != "auto":
        return dataset
    inferred = infer_dataset(row)
    if inferred is None:
        raise ValueError(
            "Could not infer dataset. Pass --dataset vqa, gqa, or vg. "
            f"question_id={row.get('question_id')}"
        )
    return inferred


def first_nonempty(row: dict, keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def original_image_name(row: dict, dataset: str) -> str:
    image_value = first_nonempty(row, ["image"])
    if image_value:
        return image_value

    image_id = int(row["image_id"])
    if dataset == "vqa":
        return f"COCO_train2014_{image_id:012d}.jpg"
    if dataset == "gqa":
        return f"{image_id}.jpg"
    if dataset == "vg":
        # Visual Genome users can point IMAGE_FOLDER at a root containing this
        # subdirectory, or override by pre-populating an `image` field.
        return f"VG_100K/{image_id}.jpg"
    raise ValueError(f"Unsupported dataset: {dataset}")


def image_name(row: dict, dataset: str, image_policy: str) -> str:
    if image_policy == "image_path":
        value = first_nonempty(row, ["image_path", "image"])
        if value:
            return value
    return original_image_name(row, dataset)


def normalize_row(row: dict, dataset_arg: str, image_policy: str) -> dict:
    dataset = resolve_dataset(dataset_arg, row)
    question = first_nonempty(row, ["question", "generated_question", "text"])
    answer = first_nonempty(row, ["answer", "generated_answer"])
    shortcut_answer = first_nonempty(row, ["shortcut_answer", "original_answer"])
    if not question:
        raise ValueError(f"Missing question/generated_question for question_id={row.get('question_id')}")
    if not answer:
        raise ValueError(f"Missing answer/generated_answer for question_id={row.get('question_id')}")

    out = dict(row)
    out["question_id"] = int(row["question_id"])
    out["question"] = question
    out["answer"] = answer
    out["correct_answer"] = answer
    out["shortcut_answer"] = shortcut_answer
    out["answer_type"] = str(row.get("answer_type", ""))
    out["image"] = image_name(row, dataset, image_policy)
    out["eval_dataset"] = dataset
    return out


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]

    normalized = [normalize_row(row, args.dataset, args.image_policy) for row in rows]

    args.question_output.parent.mkdir(parents=True, exist_ok=True)
    args.answer_output.parent.mkdir(parents=True, exist_ok=True)

    with args.question_output.open("w", encoding="utf-8") as f:
        for row in normalized:
            item = {
                "question_id": row["question_id"],
                "image": row["image"],
                "text": row["question"],
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with args.answer_output.open("w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(args.question_output)
    print(args.answer_output)


if __name__ == "__main__":
    main()
