#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_DATA_PATH = Path("/path/to/sage_repro_bundle/test_data/test_raw_with_shortcut_answer.json")
DEFAULT_IMAGE_DIR = Path("data/images/coco/train2014")
DEFAULT_MASK_PATH = Path("/path/to/sage_repro_bundle/patch_mask_analysis_output_mask_coco_seg_direct_llava_pad336_patch14.npz")
DEFAULT_OUTPUT_DIR = Path("/path/to/sage_repro_bundle/analysis/qwen_patch_suppressed_test_raw_r1p0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build question-specific images with patch-level suppression applied."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--patch-mask-analysis-path", type=Path, default=DEFAULT_MASK_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--suppress-ratio",
        type=float,
        default=1.0,
        help="0 keeps original pixels; 1 zeros masked patches completely.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def ensure_exists(path: Path, kind: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{kind} not found: {path}")


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Expected JSON array in {path}")
    return rows


def expand2square(image: Image.Image, background_color=(122, 116, 104)) -> Image.Image:
    width, height = image.size
    if width == height:
        return image
    if width > height:
        result = Image.new(image.mode, (width, width), background_color)
        result.paste(image, (0, (width - height) // 2))
        return result
    result = Image.new(image.mode, (height, height), background_color)
    result.paste(image, ((height - width) // 2, 0))
    return result


def apply_patch_suppression(image: Image.Image, coverage: np.ndarray, suppress_ratio: float) -> Image.Image:
    image = expand2square(image.convert("RGB")).resize((336, 336), Image.Resampling.BICUBIC)
    arr = np.asarray(image).astype(np.float32)
    cell = 14
    for r in range(24):
        for c in range(24):
            ratio = float(coverage[r, c])
            if ratio <= 0.0:
                continue
            scale = max(0.0, 1.0 - suppress_ratio * ratio)
            y0, y1 = r * cell, (r + 1) * cell
            x0, x1 = c * cell, (c + 1) * cell
            arr[y0:y1, x0:x1] *= scale
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def main() -> None:
    args = parse_args()
    ensure_exists(args.data_path, "data")
    ensure_exists(args.image_dir, "image dir")
    ensure_exists(args.patch_mask_analysis_path, "patch mask npz")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.data_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    npz = np.load(args.patch_mask_analysis_path, allow_pickle=True)
    qids = [int(x) for x in npz["question_ids"].tolist()]
    coverage_ratio = npz["coverage_ratio"]
    qid_to_idx = {qid: idx for idx, qid in enumerate(qids)}

    written = 0
    skipped = 0
    missing_mask = 0
    for idx, row in enumerate(rows, start=1):
        qid = int(row["question_id"])
        image_id = int(row["image_id"])
        out_path = args.output_dir / f"{qid}_{image_id}.jpg"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        mask_idx = qid_to_idx.get(qid)
        if mask_idx is None:
            missing_mask += 1
            continue
        src_path = args.image_dir / f"COCO_train2014_{image_id:012d}.jpg"
        ensure_exists(src_path, "source image")
        image = Image.open(src_path).convert("RGB")
        suppressed = apply_patch_suppression(image, coverage_ratio[mask_idx], args.suppress_ratio)
        suppressed.save(out_path, quality=args.jpeg_quality)
        written += 1
        if idx % 500 == 0:
            print(
                f"[progress] {idx}/{len(rows)} written={written} skipped={skipped} missing_mask={missing_mask}",
                flush=True,
            )

    summary = {
        "data_path": str(args.data_path),
        "patch_mask_analysis_path": str(args.patch_mask_analysis_path),
        "output_dir": str(args.output_dir),
        "suppress_ratio": args.suppress_ratio,
        "total_rows": len(rows),
        "written": written,
        "skipped_existing": skipped,
        "missing_mask": missing_mask,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
