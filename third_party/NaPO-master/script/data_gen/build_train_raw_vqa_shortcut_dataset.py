#!/usr/bin/env python
import argparse
import json
import os
from collections import Counter

from datasets import Dataset, DatasetDict, Image


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-raw-llava",
        default="data/stage2/train_raw_llava.jsonl",
    )
    parser.add_argument(
        "--train-raw-answers",
        default="data/stage2/train_raw.json",
    )
    parser.add_argument(
        "--shortcut-answers",
        default="data/stage2/vqa_train2014.json",
    )
    parser.add_argument(
        "--image-root",
        default="data/images/coco/train2014",
    )
    parser.add_argument(
        "--output-dir",
        default="third_party/NaPO-master/datasets/train_raw_vqa_shortcut_hf",
    )
    parser.add_argument(
        "--stats-path",
        default=None,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {args.output_dir}. Pass --overwrite to rebuild."
            )
        import shutil

        shutil.rmtree(args.output_dir)

    llava_rows = load_jsonl(args.train_raw_llava)
    train_raw = {row["question_id"]: row for row in load_json(args.train_raw_answers)}
    shortcut = {row["question_id"]: row for row in load_json(args.shortcut_answers)}

    records = []
    skipped = Counter()
    question_mismatch = 0
    answer_type_counter = Counter()

    for idx, row in enumerate(llava_rows):
        question_id = row["question_id"]
        train_item = train_raw.get(question_id)
        shortcut_item = shortcut.get(question_id)

        if train_item is None:
            skipped["missing_train_raw_answer"] += 1
            continue
        if shortcut_item is None:
            skipped["missing_shortcut_answer"] += 1
            continue

        image_name = row["image"]
        image_path = os.path.join(args.image_root, image_name)
        if not os.path.exists(image_path):
            skipped["missing_image"] += 1
            continue

        llava_question = row["text"].strip()
        train_question = train_item["question"].strip()
        shortcut_question = shortcut_item["question"].strip()
        if llava_question != train_question or llava_question != shortcut_question:
            question_mismatch += 1

        answer_type = train_item.get("answer_type", "")
        answer_type_counter[answer_type] += 1

        records.append(
            {
                "idx": len(records),
                "question_id": question_id,
                "question": llava_question,
                "chosen": train_item["answer"],
                "rejected": shortcut_item["answer"],
                "answer_type": answer_type,
                "image": image_path,
                "image_path": image_path,
                "origin_dataset": "train_raw_vqa_shortcut",
                "origin_split": "train",
            }
        )

    dataset = Dataset.from_list(records).cast_column("image", Image(decode=False))
    dataset_dict = DatasetDict({"train": dataset})
    dataset_dict.save_to_disk(args.output_dir)

    stats = {
        "output_dir": args.output_dir,
        "num_examples": len(records),
        "skipped": dict(skipped),
        "question_mismatch_count": question_mismatch,
        "answer_type_counts": dict(answer_type_counter),
        "source_paths": {
            "train_raw_llava": args.train_raw_llava,
            "train_raw_answers": args.train_raw_answers,
            "shortcut_answers": args.shortcut_answers,
            "image_root": args.image_root,
        },
        "example": records[0] if records else None,
    }

    stats_path = (
        args.stats_path
        if args.stats_path is not None
        else args.output_dir.rstrip("/") + ".stats.json"
    )
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
