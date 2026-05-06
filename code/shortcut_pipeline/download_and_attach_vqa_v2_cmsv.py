#!/usr/bin/env python3
"""Download official HF VQA v2-CMSV splits, optionally attaching shortcut_answer."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "sage_as" / "data" / "vqa_v2_cmsv"
DEFAULT_ANNOTATIONS_JSON = (
    REPO_ROOT
    / "data"
    / "detect-shortcuts"
    / "data"
    / "vqa2"
    / "v2_mscoco_train2014_annotations.json"
)
DEFAULT_BASE_URL = (
    "https://huggingface.co/datasets/as-benchmark-artifacts/"
    "vqa-cmsv-benchmark/resolve/main/data/vqa_v2_cmsv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write official train/val/test JSON files.",
    )
    parser.add_argument(
        "--annotations-json",
        default=str(DEFAULT_ANNOTATIONS_JSON),
        help="Official VQA v2 train annotations used when attaching shortcut_answer.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base HF URL containing train.json / val.json / test.json.",
    )
    parser.add_argument(
        "--http-proxy",
        default="",
        help="HTTP/HTTPS proxy used for HF download. Set empty string to disable.",
    )
    parser.add_argument(
        "--attach-shortcut-answer",
        action="store_true",
        help="Attach shortcut_answer to train.json using VQA official annotations.",
    )
    return parser.parse_args()


def build_opener(proxy_url: str) -> urllib.request.OpenerDirector:
    if proxy_url:
        return urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url})
        )
    return urllib.request.build_opener()


def download_json(opener: urllib.request.OpenerDirector, url: str) -> List[Dict]:
    with opener.open(url, timeout=300) as response:
        payload = json.load(response)
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected JSON array from {url}")
    return payload


def load_shortcut_answers(path: Path) -> Dict[int, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    annotations = payload.get("annotations", [])
    if not isinstance(annotations, list):
        raise RuntimeError(f"{path} does not contain an 'annotations' list.")
    answer_by_qid: Dict[int, str] = {}
    for row in annotations:
        try:
            qid = int(row["question_id"])
        except (KeyError, TypeError, ValueError):
            continue
        answer_by_qid[qid] = str(row.get("multiple_choice_answer", "")).strip()
    return answer_by_qid


def attach_shortcut_answer(rows: List[Dict], shortcut_by_qid: Dict[int, str]) -> int:
    attached = 0
    for row in rows:
        try:
            qid = int(row["question_id"])
        except (KeyError, TypeError, ValueError):
            continue
        shortcut_answer = shortcut_by_qid.get(qid, "")
        if shortcut_answer:
            row["shortcut_answer"] = shortcut_answer
            attached += 1
    return attached


def write_json(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    annotations_json = Path(args.annotations_json).resolve()

    opener = build_opener(args.http_proxy.strip())
    shortcut_by_qid: Dict[int, str] = {}
    if args.attach_shortcut_answer:
        if not annotations_json.exists():
            raise FileNotFoundError(f"Annotations JSON not found: {annotations_json}")
        shortcut_by_qid = load_shortcut_answers(annotations_json)

    summary = {
        "output_dir": str(output_dir),
        "annotations_json": str(annotations_json),
        "base_url": args.base_url,
        "http_proxy": args.http_proxy,
        "attach_shortcut_answer": args.attach_shortcut_answer,
        "splits": {},
    }

    for split in ("train", "val", "test"):
        url = f"{args.base_url}/{split}.json"
        rows = download_json(opener, url)
        attached = 0
        if split == "train" and args.attach_shortcut_answer:
            attached = attach_shortcut_answer(rows, shortcut_by_qid)
        out_path = output_dir / f"{split}.json"
        write_json(out_path, rows)
        summary["splits"][split] = {
            "path": str(out_path),
            "rows": len(rows),
            "shortcut_answer_attached": attached,
            "keys": sorted(rows[0].keys()) if rows else [],
        }

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
