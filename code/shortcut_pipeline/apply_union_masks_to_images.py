#!/usr/bin/env python3
"""Apply SAM3 union masks to original images and build masked RGB outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPE_ROOT = REPO_ROOT / "data" / "shortcut_pipeline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qa-jsonl",
        default=str(PIPE_ROOT / "cross_modality_qa_questions.jsonl"),
        help="SAM3 QA JSONL containing question_id, image_id and image filename.",
    )
    parser.add_argument(
        "--mask-dir",
        default=str(PIPE_ROOT / "union_mask" / "masks"),
        help="Directory containing SAM3 union masks named by question_id.",
    )
    parser.add_argument(
        "--image-root",
        dest="image_roots",
        action="append",
        default=None,
        help="Image root to search for row['image']. Repeat to search multiple roots.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PIPE_ROOT / "output_mask"),
        help="Directory to write masked RGB images.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Maximum number of QA rows to process; <= 0 keeps all.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip masked images that already exist.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def resolve_image_path(image_name: str, image_roots: List[Path]) -> Path:
    image_path = Path(image_name)
    if image_path.is_absolute() and image_path.exists():
        return image_path

    checked: List[str] = []
    for root in image_roots:
        candidate = root / image_name
        checked.append(str(candidate))
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not resolve image {image_name!r}. Checked: {checked[:8]}"
    )


def resolve_mask_path(mask_dir: Path, question_id: int) -> Optional[Path]:
    for suffix in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = mask_dir / f"{question_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def apply_mask(image_path: Path, mask_path: Path, output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)

    image_np = np.asarray(image).copy()
    mask_np = np.asarray(mask) > 0
    image_np[mask_np] = 0
    Image.fromarray(image_np).save(output_path)


def main() -> None:
    args = parse_args()
    qa_jsonl = Path(args.qa_jsonl).resolve()
    mask_dir = Path(args.mask_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    image_roots = [Path(p).resolve() for p in (args.image_roots or [])]

    if not qa_jsonl.exists():
        raise FileNotFoundError(f"QA JSONL not found: {qa_jsonl}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask dir not found: {mask_dir}")
    if not image_roots:
        raise ValueError("At least one --image-root is required.")

    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    written = 0
    skipped_existing = 0
    skipped_missing_mask = 0
    skipped_missing_image = 0

    for row in load_jsonl(qa_jsonl):
        if args.limit > 0 and processed >= args.limit:
            break

        processed += 1
        qid = int(row["question_id"])
        image_id = int(row["image_id"])
        output_path = output_dir / f"{qid}_{image_id}.png"

        if args.skip_existing and output_path.exists():
            skipped_existing += 1
            continue

        mask_path = resolve_mask_path(mask_dir, qid)
        if mask_path is None:
            skipped_missing_mask += 1
            continue

        try:
            image_path = resolve_image_path(str(row["image"]), image_roots)
        except FileNotFoundError:
            skipped_missing_image += 1
            continue

        apply_mask(image_path=image_path, mask_path=mask_path, output_path=output_path)
        written += 1

    summary = {
        "qa_jsonl": str(qa_jsonl),
        "mask_dir": str(mask_dir),
        "image_roots": [str(p) for p in image_roots],
        "output_dir": str(output_dir),
        "processed_rows": processed,
        "written": written,
        "skipped_existing": skipped_existing,
        "skipped_missing_mask": skipped_missing_mask,
        "skipped_missing_image": skipped_missing_image,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
