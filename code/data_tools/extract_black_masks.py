import argparse
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_MASK_DIR = Path("/path/to/sage_repro_bundle/output_mask")
DEFAULT_ORIGINAL_DIR = Path("data/images/coco/train2014")
DEFAULT_OUTPUT_DIR = Path("/path/to/sage_repro_bundle/pure_black_mask")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare output_mask with the original image and extract a pure black binary mask."
    )
    parser.add_argument("--mask-dir", type=Path, default=DEFAULT_MASK_DIR)
    parser.add_argument("--original-dir", type=Path, default=DEFAULT_ORIGINAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--drop-threshold",
        type=int,
        default=25,
        help="Minimum per-pixel brightness drop from original to output_mask.",
    )
    parser.add_argument(
        "--black-threshold",
        type=int,
        default=30,
        help="Maximum channel value to still treat output_mask pixels as black.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing mask files in the output directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N images, useful for quick checks.",
    )
    return parser.parse_args()


def iter_mask_files(mask_dir: Path):
    for path in sorted(mask_dir.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        yield path


def resolve_original_path(mask_path: Path, original_dir: Path):
    stem_suffix = mask_path.stem.rsplit("_", 1)[-1]
    candidates = [original_dir / f"{stem_suffix}{mask_path.suffix.lower()}"]

    if stem_suffix.isdigit():
        image_id = int(stem_suffix)
        candidates.extend(
            [
                original_dir / f"COCO_train2014_{image_id:012d}.jpg",
                original_dir / f"COCO_train2014_{image_id:012d}.png",
                original_dir / f"COCO_val2014_{image_id:012d}.jpg",
                original_dir / f"COCO_val2014_{image_id:012d}.png",
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Cannot find original image for {mask_path.name}")


def extract_binary_black_mask(
    original_path: Path,
    output_mask_path: Path,
    drop_threshold: int,
    black_threshold: int,
):
    original = Image.open(original_path).convert("RGB")
    output_mask = Image.open(output_mask_path).convert("RGB")

    if original.size != output_mask.size:
        original = original.resize(output_mask.size, Image.Resampling.BILINEAR)

    original_np = np.asarray(original, dtype=np.int16)
    output_np = np.asarray(output_mask, dtype=np.int16)

    # We only care about pixels darkened by the mask, not symmetric JPEG noise.
    brightness_drop = np.max(np.clip(original_np - output_np, 0, None), axis=2)
    output_darkness = np.max(output_np, axis=2)

    masked_region = (brightness_drop >= drop_threshold) & (output_darkness <= black_threshold)
    binary_mask = np.where(masked_region, 0, 255).astype(np.uint8)
    return Image.fromarray(binary_mask, mode="L")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    missing = 0
    errors = 0

    mask_files = iter_mask_files(args.mask_dir)
    if args.limit is not None:
        mask_files = list(mask_files)[: args.limit]

    for index, mask_path in enumerate(mask_files, start=1):
        output_path = args.output_dir / f"{mask_path.stem}.png"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            original_path = resolve_original_path(mask_path, args.original_dir)
        except FileNotFoundError:
            missing += 1
            print(f"[missing] {mask_path.name}")
            continue

        try:
            binary_mask = extract_binary_black_mask(
                original_path=original_path,
                output_mask_path=mask_path,
                drop_threshold=args.drop_threshold,
                black_threshold=args.black_threshold,
            )
            binary_mask.save(output_path)
            processed += 1
        except Exception as exc:
            errors += 1
            print(f"[error] {mask_path.name}: {exc}")
            continue

        if index % 1000 == 0:
            print(
                f"processed={processed} skipped={skipped} missing={missing} errors={errors} last={mask_path.name}"
            )

    print(
        f"done: processed={processed} skipped={skipped} missing={missing} errors={errors} output_dir={args.output_dir}"
    )


if __name__ == "__main__":
    main()
