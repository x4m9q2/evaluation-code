#!/usr/bin/env python3
"""One-shot stage-1 shortcut pipeline for VQA2/COCO inputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code" / "shortcut_pipeline"
DEFAULT_WORK_DIR = REPO_ROOT / "data" / "shortcut_pipeline"
DEFAULT_INSTANCES = REPO_ROOT / "annotations" / "instances_train2014.json"
DEFAULT_QUESTIONS = REPO_ROOT / "data" / "detect-shortcuts" / "data" / "vqa2" / "v2_OpenEnded_mscoco_train2014_questions.json"
DEFAULT_ANNOTATIONS = REPO_ROOT / "data" / "detect-shortcuts" / "data" / "vqa2" / "v2_mscoco_train2014_annotations.json"
DEFAULT_GMINER = CODE_ROOT / "bin" / "GMiner"
DEFAULT_MATCHER = CODE_ROOT / "bin" / "cuda"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances-json", default=str(DEFAULT_INSTANCES))
    parser.add_argument("--questions-json", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--annotations-json", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--gminer-path", default=str(DEFAULT_GMINER))
    parser.add_argument("--matcher-binary", default=str(DEFAULT_MATCHER))
    parser.add_argument("--support", type=float, default=0.02)
    parser.add_argument("--max-length", type=int, default=4)
    parser.add_argument("--min-conf", type=float, default=0.3)
    parser.add_argument("--most-common-answers", type=int, default=200)
    parser.add_argument("--matcher-gpus", default="0,1,2,3")
    parser.add_argument("--matcher-batch-size", type=int, default=262144)
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    parser.add_argument("--image-to-detection-json", default="")
    parser.add_argument("--rules-dir", default="")
    parser.add_argument("--matches-json", default="")
    parser.add_argument("--merged-json", default="")
    parser.add_argument("--subset-questions-json", default="")
    parser.add_argument("--subset-annotations-json", default="")
    parser.add_argument("--image-ids-json", default="")
    parser.add_argument("--limit", type=int, default=0, help="Limit stage-1 inputs; <= 0 keeps all.")
    return parser.parse_args()


def run(cmd: List[str]) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def unwrap_questions(path: Path) -> List[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "questions" in payload:
        return list(payload["questions"])
    if isinstance(payload, list):
        return list(payload)
    raise RuntimeError(f"Unsupported questions format: {path}")


def unwrap_annotations(path: Path) -> List[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "annotations" in payload:
        return list(payload["annotations"])
    if isinstance(payload, list):
        return list(payload)
    raise RuntimeError(f"Unsupported annotations format: {path}")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def write_subset(
    questions_path: Path,
    annotations_path: Path,
    subset_questions_path: Path,
    subset_annotations_path: Path,
    image_ids_path: Path,
    limit: int,
) -> Path:
    questions = unwrap_questions(questions_path)[:limit]
    annotations = unwrap_annotations(annotations_path)[:limit]
    subset_questions_path.parent.mkdir(parents=True, exist_ok=True)
    subset_annotations_path.parent.mkdir(parents=True, exist_ok=True)
    image_ids_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(subset_questions_path, {"questions": questions})
    write_json(subset_annotations_path, {"annotations": annotations})
    image_ids = sorted({str(item.get("image_id")) for item in questions if item.get("image_id") is not None})
    write_json(image_ids_path, image_ids)
    print(
        f"[subset] wrote {len(questions)} questions, {len(annotations)} annotations, {len(image_ids)} image ids",
        flush=True,
    )
    return image_ids_path


def main() -> None:
    args = parse_args()

    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    code_dir = CODE_ROOT.resolve()

    questions_path = Path(args.questions_json).resolve()
    annotations_path = Path(args.annotations_json).resolve()
    instances_path = Path(args.instances_json).resolve()
    gminer_path = str(Path(args.gminer_path).resolve())
    matcher_binary = str(Path(args.matcher_binary).resolve())
    image_to_detection_arg = Path(args.image_to_detection_json).resolve() if args.image_to_detection_json else None
    rules_dir_arg = Path(args.rules_dir).resolve() if args.rules_dir else None
    matches_json_arg = Path(args.matches_json).resolve() if args.matches_json else None
    merged_json_arg = Path(args.merged_json).resolve() if args.merged_json else None
    subset_questions_arg = Path(args.subset_questions_json).resolve() if args.subset_questions_json else None
    subset_annotations_arg = Path(args.subset_annotations_json).resolve() if args.subset_annotations_json else None
    image_ids_arg = Path(args.image_ids_json).resolve() if args.image_ids_json else None

    subset_questions_path = subset_questions_arg or (work_dir / "train_questions.json")
    subset_annotations_path = subset_annotations_arg or (work_dir / "train_annotations.json")
    image_ids_path = image_ids_arg or (work_dir / "image_ids.json")

    if args.limit > 0:
        image_to_detection = image_to_detection_arg or (work_dir / "image_to_detection.json")
        rules_dir = rules_dir_arg or (work_dir / "rules")
        matches_json = matches_json_arg or (work_dir / "shortcuts_matches.json")
        merged_json = merged_json_arg or (work_dir / "gqa_merged_output_with_answer_type.json")
        image_ids_path = write_subset(
            questions_path,
            annotations_path,
            subset_questions_path,
            subset_annotations_path,
            image_ids_path,
            args.limit,
        )
        questions_path = subset_questions_path
        annotations_path = subset_annotations_path
        max_images = "0"
    else:
        image_ids_path = None
        image_to_detection = image_to_detection_arg or (work_dir / "image_to_detection.json")
        rules_dir = rules_dir_arg or (work_dir / "rules")
        matches_json = matches_json_arg or (work_dir / "shortcuts_matches.json")
        merged_json = merged_json_arg or (work_dir / "gqa_merged_output_with_answer_type.json")
        max_images = "0"

    rules_dir.mkdir(parents=True, exist_ok=True)

    transfer_cmd = [
        sys.executable,
        str(code_dir / "transfer_detection.py"),
        "--instances-json",
        str(instances_path),
        "--output-json",
        str(image_to_detection),
    ]
    if image_ids_path is not None:
        transfer_cmd.extend(["--image-ids-json", str(image_ids_path), "--max-images", max_images])
    run(transfer_cmd)

    run(
        [
            sys.executable,
            str(code_dir / "generate_rules_json.py"),
            "--dataset",
            "vqa",
            "--train_questions_path",
            str(questions_path),
            "--train_annotations_path",
            str(annotations_path),
            "--visual_words",
            str(image_to_detection),
            "--gminer_path",
            gminer_path,
            "--save_dir",
            str(rules_dir),
            "--support",
            str(args.support),
            "--max_length",
            str(args.max_length),
            "--min_conf",
            str(args.min_conf),
            "--most_common_answers",
            str(args.most_common_answers),
        ]
    )

    run(
        [
            matcher_binary,
            "--rules_json",
            str(rules_dir / "rules.json"),
            "--questions_json",
            str(questions_path),
            "--annotations_json",
            str(annotations_path),
            "--image_classes_json",
            str(image_to_detection),
            "--output_json",
            str(matches_json),
            "--gpus",
            args.matcher_gpus,
            "--batch_size",
            str(args.matcher_batch_size),
        ]
    )

    merge_cmd = [
        sys.executable,
        str(code_dir / "merge_shortcuts_output.py"),
        "--rules-json",
        str(rules_dir / "rules.json"),
        "--matches-json",
        str(matches_json),
        "--questions-json",
        str(questions_path),
        "--annotations-json",
        str(annotations_path),
        "--output-json",
        str(merged_json),
    ]
    if args.limit > 0:
        merge_cmd.append("--include-unmatched")
    run(merge_cmd)

    print(f"[done] merged output: {merged_json}", flush=True)


if __name__ == "__main__":
    main()
