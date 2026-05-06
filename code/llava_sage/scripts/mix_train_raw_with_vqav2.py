#!/usr/bin/env python3
import argparse
import json
import random
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Mix the current train_raw dataset with VQAv2 train2014, "
            "optionally offsetting VQAv2 question_id values and emitting "
            "both raw-VQA and LLaVA-conversation outputs."
        )
    )
    parser.add_argument(
        "--train-raw",
        default="/path/to/sage_repro_bundle/train_raw.json",
        help="Path to the current train_raw.json file.",
    )
    parser.add_argument(
        "--vqav2-train",
        default="/path/to/sage_repro_bundle/vqa_train2014.json",
        help="Path to the VQAv2 train2014 json file.",
    )
    parser.add_argument(
        "--output-raw",
        default="/path/to/sage_repro_bundle/train_raw_plus_vqav2.json",
        help="Path to the mixed raw-VQA json output.",
    )
    parser.add_argument(
        "--output-llava",
        default="/path/to/sage_repro_bundle/playground/data/train_raw_plus_vqav2_llava_train2017.json",
        help="Path to the mixed LLaVA-conversation json output.",
    )
    parser.add_argument(
        "--stats-output",
        default="",
        help="Optional stats json path. Defaults to <output-raw>.stats.json.",
    )
    parser.add_argument(
        "--vqav2-mode",
        choices=["all", "overlap_only", "non_overlap_only"],
        default="all",
        help=(
            "Select whether to include all VQAv2 samples, only those whose "
            "question_id overlaps with train_raw, or only the non-overlap part."
        ),
    )
    parser.add_argument(
        "--vqav2-question-id-offset",
        type=int,
        default=None,
        help=(
            "Offset added to selected VQAv2 question_id values. "
            "Defaults to max(train_raw.question_id) + 1."
        ),
    )
    parser.add_argument(
        "--image-subdir",
        default="train2017",
        help="Subdirectory stored in the converted LLaVA image field.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=42,
        help="Deterministic shuffle seed used unless --no-shuffle is set.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Keep output order as train_raw first, then selected VQAv2 rows.",
    )
    return parser.parse_args()


def load_json_list(path: Path):
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON list.")
    return data


def normalize_raw_row(row, path: Path, index: int):
    required_keys = {"question_id", "question", "image_id", "answer"}
    missing = required_keys - set(row.keys())
    if missing:
        raise ValueError(f"{path} row {index} missing keys: {sorted(missing)}")

    question = str(row["question"]).strip()
    answer = str(row["answer"]).strip()
    if not question or not answer:
        raise ValueError(f"{path} row {index} has empty question or answer")

    return {
        "question_id": int(row["question_id"]),
        "question": question,
        "image_id": int(row["image_id"]),
        "answer": answer,
        "answer_type": row.get("answer_type", "other"),
    }


def to_llava_item(row, source_name: str, image_subdir: str):
    return {
        "id": f"{source_name}_{row['question_id']}",
        "image": f"{image_subdir}/{int(row['image_id']):012d}.jpg",
        "conversations": [
            {"from": "human", "value": f"<image>\n{row['question']}"},
            {"from": "gpt", "value": row["answer"]},
        ],
    }


def main():
    args = parse_args()

    train_raw_path = Path(args.train_raw)
    vqav2_path = Path(args.vqav2_train)
    output_raw_path = Path(args.output_raw)
    output_llava_path = Path(args.output_llava)
    stats_path = Path(args.stats_output) if args.stats_output else output_raw_path.with_suffix(output_raw_path.suffix + ".stats.json")

    train_raw_rows = load_json_list(train_raw_path)
    vqav2_rows = load_json_list(vqav2_path)

    train_raw_normalized = [
        normalize_raw_row(row, train_raw_path, idx) for idx, row in enumerate(train_raw_rows)
    ]
    vqav2_normalized = [
        normalize_raw_row(row, vqav2_path, idx) for idx, row in enumerate(vqav2_rows)
    ]

    train_raw_qids = {row["question_id"] for row in train_raw_normalized}
    vqav2_qids = {row["question_id"] for row in vqav2_normalized}
    overlap_qids = train_raw_qids & vqav2_qids

    if args.vqav2_mode == "all":
        selected_vqav2 = vqav2_normalized
    elif args.vqav2_mode == "overlap_only":
        selected_vqav2 = [row for row in vqav2_normalized if row["question_id"] in overlap_qids]
    else:
        selected_vqav2 = [row for row in vqav2_normalized if row["question_id"] not in train_raw_qids]

    qid_offset = args.vqav2_question_id_offset
    if qid_offset is None:
        qid_offset = max(train_raw_qids) + 1 if train_raw_qids else 1

    mixed_rows = []
    mixed_rows.extend(("train_raw", row) for row in train_raw_normalized)
    for row in selected_vqav2:
        remapped = dict(row)
        remapped["question_id"] = remapped["question_id"] + qid_offset
        mixed_rows.append(("vqav2_train2014", remapped))

    if not args.no_shuffle:
        random.Random(args.shuffle_seed).shuffle(mixed_rows)

    raw_output = [row for _, row in mixed_rows]
    llava_output = [
        to_llava_item(row, source_name=source, image_subdir=args.image_subdir)
        for source, row in mixed_rows
    ]

    unique_qids = {row["question_id"] for row in raw_output}
    if len(unique_qids) != len(raw_output):
        raise ValueError(
            f"Mixed output still has duplicate question_id values: "
            f"{len(raw_output) - len(unique_qids)} duplicates"
        )

    output_raw_path.parent.mkdir(parents=True, exist_ok=True)
    with output_raw_path.open("w") as f:
        json.dump(raw_output, f, ensure_ascii=False)

    output_llava_path.parent.mkdir(parents=True, exist_ok=True)
    with output_llava_path.open("w") as f:
        json.dump(llava_output, f, ensure_ascii=False)

    source_counter = Counter(source for source, _ in mixed_rows)
    answer_type_counter = Counter(row.get("answer_type", "other") for row in raw_output)

    stats = {
        "train_raw_path": str(train_raw_path),
        "vqav2_train_path": str(vqav2_path),
        "output_raw_path": str(output_raw_path),
        "output_llava_path": str(output_llava_path),
        "train_raw_count": len(train_raw_normalized),
        "vqav2_count": len(vqav2_normalized),
        "overlap_question_id_count": len(overlap_qids),
        "vqav2_mode": args.vqav2_mode,
        "selected_vqav2_count": len(selected_vqav2),
        "vqav2_question_id_offset": qid_offset,
        "output_count": len(raw_output),
        "unique_output_question_id_count": len(unique_qids),
        "source_counts": dict(source_counter),
        "answer_type_counts": dict(answer_type_counter),
        "shuffled": not args.no_shuffle,
        "shuffle_seed": None if args.no_shuffle else args.shuffle_seed,
        "image_subdir": args.image_subdir,
    }

    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
