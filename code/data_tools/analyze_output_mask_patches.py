import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


DEFAULT_OUTPUT_MASK_DIR = Path("/path/to/sage_repro_bundle/output_mask")
DEFAULT_ORIGINAL_DIR = Path("data/images/coco/train2014")
DEFAULT_MODEL_CONFIG = Path("/path/to/sage_repro_bundle/llava-v1.5-7b/config.json")
DEFAULT_VISION_CONFIG = Path("/path/to/sage_repro_bundle/clip-vit-large-patch14-336/config.json")
DEFAULT_PREPROCESSOR_CONFIG = Path("/path/to/sage_repro_bundle/clip-vit-large-patch14-336/preprocessor_config.json")
DEFAULT_OUTPUT = Path("/path/to/sage_repro_bundle/patch_mask_analysis_output_mask_llava_pad336_patch14.npz")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze output_mask images under LLaVA's pad-to-square 336px CLIP preprocessing "
            "and record which visual patches contain mask pixels plus each patch's mask ratio."
        )
    )
    parser.add_argument("--output-mask-dir", type=Path, default=DEFAULT_OUTPUT_MASK_DIR)
    parser.add_argument("--original-dir", type=Path, default=DEFAULT_ORIGINAL_DIR)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--vision-config", type=Path, default=DEFAULT_VISION_CONFIG)
    parser.add_argument("--preprocessor-config", type=Path, default=DEFAULT_PREPROCESSOR_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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
        "--workers",
        type=int,
        default=1,
        help="Worker count for per-image analysis. 1 is the most stable choice here.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only analyze the first N images.")
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Fail immediately if an original image cannot be resolved.",
    )
    parser.add_argument(
        "--include-train-json",
        type=Path,
        default=None,
        help="Only analyze mask files whose question_id appears in this train JSON.",
    )
    return parser.parse_args()


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def iter_mask_files(mask_dir: Path):
    for path in sorted(mask_dir.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        yield path


def load_allowed_question_ids(train_json: Path) -> set[int]:
    data = load_json(train_json)
    allowed = set()
    for row in data:
        qid = row.get("question_id")
        if qid is None:
            continue
        try:
            allowed.add(int(qid))
        except (TypeError, ValueError):
            continue
    return allowed


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

    brightness_drop = np.max(np.clip(original_np - output_np, 0, None), axis=2)
    output_darkness = np.max(output_np, axis=2)
    masked_region = (brightness_drop >= drop_threshold) & (output_darkness <= black_threshold)
    return masked_region.astype(np.float32)


def pad_to_square(mask: np.ndarray):
    height, width = mask.shape
    side = max(height, width)
    pad_top = (side - height) // 2
    pad_left = (side - width) // 2
    square = np.zeros((side, side), dtype=np.float32)
    square[pad_top:pad_top + height, pad_left:pad_left + width] = mask
    return square, pad_top, pad_left, side


def compute_patch_coverage(square_mask: np.ndarray, grid_size: int):
    mask_tensor = torch.from_numpy(square_mask).unsqueeze(0).unsqueeze(0)
    coverage = F.adaptive_avg_pool2d(mask_tensor, (grid_size, grid_size))
    return coverage.squeeze(0).squeeze(0).numpy().astype(np.float32)


def analyze_single_mask(task):
    mask_path, original_dir, drop_threshold, black_threshold, grid_size, strict_missing = task
    try:
        original_path = resolve_original_path(mask_path, original_dir)
    except FileNotFoundError:
        if strict_missing:
            raise
        return {
            "status": "missing",
            "image_name": mask_path.name,
        }

    mask = extract_binary_black_mask(
        original_path=original_path,
        output_mask_path=mask_path,
        drop_threshold=drop_threshold,
        black_threshold=black_threshold,
    )
    square_mask, pad_top, pad_left, side = pad_to_square(mask)
    coverage = compute_patch_coverage(square_mask, grid_size)
    contains_mask = coverage > 0.0

    image_name = mask_path.name
    stem_parts = mask_path.stem.rsplit("_", 1)
    question_id = int(stem_parts[0]) if stem_parts[0].isdigit() else -1
    image_id = int(stem_parts[1]) if len(stem_parts) == 2 and stem_parts[1].isdigit() else -1

    return {
        "status": "ok",
        "image_name": image_name,
        "question_id": question_id,
        "image_id": image_id,
        "original_height": int(mask.shape[0]),
        "original_width": int(mask.shape[1]),
        "padded_side": int(side),
        "pad_top": int(pad_top),
        "pad_left": int(pad_left),
        "coverage": coverage,
        "has_mask": contains_mask,
        "mask_pixel_count": int(mask.sum()),
    }


def main():
    args = parse_args()

    model_cfg = load_json(args.model_config)
    vision_cfg = load_json(args.vision_config)["vision_config"]
    preprocessor_cfg = load_json(args.preprocessor_config)

    image_aspect_ratio = model_cfg.get("image_aspect_ratio", "square")
    if image_aspect_ratio != "pad":
        raise ValueError(
            f"This script currently matches the repo's pad workflow only, but got image_aspect_ratio={image_aspect_ratio!r}."
        )

    image_size = int(vision_cfg["image_size"])
    patch_size = int(vision_cfg["patch_size"])
    grid_size = image_size // patch_size

    processor_crop_size = preprocessor_cfg["crop_size"]
    if isinstance(processor_crop_size, dict):
        crop_height = int(processor_crop_size["height"])
    else:
        crop_height = int(processor_crop_size)
    if crop_height != image_size:
        raise ValueError(f"Unexpected crop size {crop_height}; expected {image_size}.")

    mask_paths = list(iter_mask_files(args.output_mask_dir))
    if args.include_train_json is not None:
        allowed_qids = load_allowed_question_ids(args.include_train_json)
        filtered_paths = []
        for path in mask_paths:
            stem_parts = path.stem.rsplit("_", 1)
            if not stem_parts or not stem_parts[0].isdigit():
                continue
            if int(stem_parts[0]) in allowed_qids:
                filtered_paths.append(path)
        mask_paths = filtered_paths
    if args.limit is not None:
        mask_paths = mask_paths[: args.limit]

    requested_images = len(mask_paths)
    image_names = np.empty(requested_images, dtype=object)
    question_ids = np.empty(requested_images, dtype=np.int64)
    image_ids = np.empty(requested_images, dtype=np.int64)
    original_widths = np.empty(requested_images, dtype=np.int32)
    original_heights = np.empty(requested_images, dtype=np.int32)
    padded_sides = np.empty(requested_images, dtype=np.int32)
    pad_tops = np.empty(requested_images, dtype=np.int32)
    pad_lefts = np.empty(requested_images, dtype=np.int32)
    mask_pixel_counts = np.empty(requested_images, dtype=np.int64)
    coverage_ratio = np.empty((requested_images, grid_size, grid_size), dtype=np.float32)
    has_mask = np.empty((requested_images, grid_size, grid_size), dtype=np.bool_)

    total_masked_patches = 0
    total_mask_coverage = 0.0
    total_mask_pixels = 0
    missing_count = 0

    tasks = [
        (
            mask_path,
            args.original_dir,
            args.drop_threshold,
            args.black_threshold,
            grid_size,
            args.strict_missing,
        )
        for mask_path in mask_paths
    ]

    torch.set_num_threads(1)

    if args.workers <= 1:
        result_iter = map(analyze_single_mask, tasks)
    else:
        executor = ThreadPoolExecutor(max_workers=args.workers)
        result_iter = executor.map(analyze_single_mask, tasks)

    try:
        write_idx = 0
        for read_idx, result in enumerate(result_iter):
            if result["status"] == "missing":
                missing_count += 1
                print(f"[missing] {result['image_name']}")
                continue

            image_names[write_idx] = result["image_name"]
            question_ids[write_idx] = result["question_id"]
            image_ids[write_idx] = result["image_id"]
            original_widths[write_idx] = result["original_width"]
            original_heights[write_idx] = result["original_height"]
            padded_sides[write_idx] = result["padded_side"]
            pad_tops[write_idx] = result["pad_top"]
            pad_lefts[write_idx] = result["pad_left"]
            mask_pixel_counts[write_idx] = result["mask_pixel_count"]
            coverage_ratio[write_idx] = result["coverage"]
            has_mask[write_idx] = result["has_mask"]
            write_idx += 1

            total_masked_patches += int(result["has_mask"].sum())
            total_mask_coverage += float(result["coverage"].sum())
            total_mask_pixels += int(result["mask_pixel_count"])

            if (read_idx + 1) % 1000 == 0:
                print(f"processed={read_idx + 1}/{requested_images} kept={write_idx} last={result['image_name']}")
    finally:
        if args.workers > 1:
            executor.shutdown(wait=True)

    num_images = write_idx
    image_names = image_names[:num_images]
    question_ids = question_ids[:num_images]
    image_ids = image_ids[:num_images]
    original_widths = original_widths[:num_images]
    original_heights = original_heights[:num_images]
    padded_sides = padded_sides[:num_images]
    pad_tops = pad_tops[:num_images]
    pad_lefts = pad_lefts[:num_images]
    mask_pixel_counts = mask_pixel_counts[:num_images]
    coverage_ratio = coverage_ratio[:num_images]
    has_mask = has_mask[:num_images]

    metadata = {
        "model_config": str(args.model_config),
        "vision_config": str(args.vision_config),
        "preprocessor_config": str(args.preprocessor_config),
        "output_mask_dir": str(args.output_mask_dir),
        "original_dir": str(args.original_dir),
        "image_aspect_ratio": image_aspect_ratio,
        "image_size": image_size,
        "patch_size": patch_size,
        "grid_size": grid_size,
        "num_patches_per_image": grid_size * grid_size,
        "patch_order": "row-major (coverage_ratio[i, row, col])",
        "coverage_ratio_definition": (
            "fraction of each visual patch covered by the extracted binary mask after "
            "LLaVA's pad-to-square preprocessing"
        ),
        "contains_mask_definition": "coverage_ratio > 0",
        "mask_extraction": {
            "source": "output_mask versus original image",
            "drop_threshold": args.drop_threshold,
            "black_threshold": args.black_threshold,
        },
        "include_train_json": str(args.include_train_json) if args.include_train_json else None,
        "requested_images": requested_images,
        "analyzed_images": num_images,
        "missing_original_images": missing_count,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        metadata_json=np.array(json.dumps(metadata, ensure_ascii=True)),
        image_names=image_names,
        question_ids=question_ids,
        image_ids=image_ids,
        original_widths=original_widths,
        original_heights=original_heights,
        padded_sides=padded_sides,
        pad_tops=pad_tops,
        pad_lefts=pad_lefts,
        mask_pixel_counts=mask_pixel_counts,
        coverage_ratio=coverage_ratio,
        has_mask=has_mask,
    )

    mean_masked_patches = total_masked_patches / max(num_images, 1)
    mean_patch_coverage = total_mask_coverage / max(num_images * grid_size * grid_size, 1)
    mean_mask_pixels = total_mask_pixels / max(num_images, 1)

    print(f"done: images={num_images} output={args.output}")
    print(f"grid={grid_size}x{grid_size} patch_size={patch_size} image_size={image_size}")
    print(f"mean_masked_patches_per_image={mean_masked_patches:.4f}")
    print(f"mean_patch_coverage_over_all_patches={mean_patch_coverage:.6f}")
    print(f"mean_mask_pixels_per_image={mean_mask_pixels:.2f}")


if __name__ == "__main__":
    main()
