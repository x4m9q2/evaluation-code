#!/usr/bin/env python3
"""Convert shortcut-generated NaPO JSON into a HF dataset directory for LLaVA NaPO."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset, DatasetDict, Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_JSON = REPO_ROOT / "data" / "napo" / "shortcut_generated_vqa" / "train.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "napo_llava" / "train_raw_pos_neg_shortcut_hf"
DEFAULT_IMAGE_ROOT = REPO_ROOT / "data" / "images" / "coco" / "train2014"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-json",
        default=str(DEFAULT_INPUT_JSON),
        help="Preference JSON split produced by build_shortcut_napo_splits.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="HF save_to_disk output directory for LLaVA NaPO.",
    )
    parser.add_argument(
        "--image-root",
        default=str(DEFAULT_IMAGE_ROOT),
        help="Directory containing COCO train2014 images.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output-dir before rebuilding.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise RuntimeError(f"{path} does not contain a JSON list.")
    return rows


def resolve_image_path(image_root: Path, row: Dict[str, Any]) -> Path:
    image_name = str(row.get("image") or "").strip()
    if image_name:
        image_path = image_root / image_name
        if image_path.exists():
            return image_path

    image_id = row.get("image_id")
    if image_id is None:
        raise FileNotFoundError(f"missing image and image_id for question_id={row.get('question_id')}")
    fallback = image_root / f"COCO_train2014_{int(image_id):012d}.jpg"
    if not fallback.exists():
        raise FileNotFoundError(f"missing image file for question_id={row.get('question_id')}: {fallback}")
    return fallback


def build_records(rows: List[Dict[str, Any]], image_root: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for row in rows:
        chosen = str(row.get("chosen", "")).strip()
        rejected = str(row.get("rejected", "")).strip()
        question = str(row.get("question", "")).strip()
        if not question or not chosen or not rejected:
            continue
        if chosen == rejected:
            continue

        image_path = resolve_image_path(image_root, row)
        records.append(
            {
                "idx": len(records),
                "question_id": int(row["question_id"]),
                "image_id": int(row["image_id"]),
                "question": question,
                "chosen": chosen,
                "rejected": rejected,
                "answer_type": row.get("answer_type"),
                "image": str(image_path),
                "image_path": str(image_path),
                "origin_dataset": row.get("origin_dataset", "shortcut_stage2_generated_vqa"),
                "origin_split": row.get("origin_split", "train"),
                "original_question": row.get("original_question"),
                "original_answer": row.get("original_answer"),
                "generated_question": row.get("generated_question"),
                "generated_answer": row.get("generated_answer"),
            }
        )
    return records


def main() -> None:
    args = parse_args()
    input_json = Path(args.input_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    image_root = Path(args.image_root).resolve()

    if not input_json.exists():
        raise FileNotFoundError(f"input JSON not found: {input_json}")
    if not image_root.exists():
        raise FileNotFoundError(f"image root not found: {image_root}")

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output dir already exists: {output_dir}; pass --overwrite to rebuild")
        shutil.rmtree(output_dir)

    rows = load_rows(input_json)
    records = build_records(rows, image_root)
    dataset = Dataset.from_list(records).cast_column("image", Image(decode=False))
    dataset_dict = DatasetDict({"train": dataset})
    dataset_dict.save_to_disk(str(output_dir))

    summary = {
        "input_json": str(input_json),
        "output_dir": str(output_dir),
        "image_root": str(image_root),
        "num_rows": len(rows),
        "num_records": len(records),
        "example": records[0] if records else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
