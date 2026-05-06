import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


DEFAULT_MASK_DIR = Path("/path/to/sage_repro_bundle/pure_black_mask")
DEFAULT_MODEL_CONFIG = Path("/path/to/sage_repro_bundle/llava-v1.5-7b/config.json")
DEFAULT_VISION_CONFIG = Path("/path/to/sage_repro_bundle/clip-vit-large-patch14-336/config.json")
DEFAULT_PREPROCESSOR_CONFIG = Path("/path/to/sage_repro_bundle/clip-vit-large-patch14-336/preprocessor_config.json")
DEFAULT_OUTPUT = Path("/path/to/sage_repro_bundle/patch_mask_analysis_llava_pad336_patch14.npz")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze which LLaVA visual patches contain mask pixels and how much each patch is covered."
    )
    parser.add_argument("--mask-dir", type=Path, default=DEFAULT_MASK_DIR)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--vision-config", type=Path, default=DEFAULT_VISION_CONFIG)
    parser.add_argument("--preprocessor-config", type=Path, default=DEFAULT_PREPROCESSOR_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None, help="Only analyze the first N masks.")
    return parser.parse_args()


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def load_mask(mask_path: Path):
    mask = np.array(Image.open(mask_path).convert("L"))
    return (mask < 128).astype(np.float32)


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

    mask_paths = sorted(
        path for path in args.mask_dir.iterdir() if path.is_file() and not path.name.startswith(".")
    )
    if args.limit is not None:
        mask_paths = mask_paths[: args.limit]

    num_images = len(mask_paths)
    image_names = np.empty(num_images, dtype=object)
    image_ids = np.empty(num_images, dtype=np.int64)
    original_widths = np.empty(num_images, dtype=np.int32)
    original_heights = np.empty(num_images, dtype=np.int32)
    padded_sides = np.empty(num_images, dtype=np.int32)
    pad_tops = np.empty(num_images, dtype=np.int32)
    pad_lefts = np.empty(num_images, dtype=np.int32)
    coverage_ratio = np.empty((num_images, grid_size, grid_size), dtype=np.float32)
    has_mask = np.empty((num_images, grid_size, grid_size), dtype=np.bool_)

    total_masked_patches = 0
    total_mask_coverage = 0.0

    for idx, mask_path in enumerate(mask_paths):
        mask = load_mask(mask_path)
        square_mask, pad_top, pad_left, side = pad_to_square(mask)
        coverage = compute_patch_coverage(square_mask, grid_size)
        contains_mask = coverage > 0.0

        image_names[idx] = mask_path.name
        image_ids[idx] = int(mask_path.stem.rsplit("_", 1)[-1])
        original_heights[idx], original_widths[idx] = mask.shape
        padded_sides[idx] = side
        pad_tops[idx] = pad_top
        pad_lefts[idx] = pad_left
        coverage_ratio[idx] = coverage
        has_mask[idx] = contains_mask

        total_masked_patches += int(contains_mask.sum())
        total_mask_coverage += float(coverage.sum())

        if (idx + 1) % 1000 == 0:
            print(f"processed={idx + 1}/{num_images} last={mask_path.name}")

    metadata = {
        "model_config": str(args.model_config),
        "vision_config": str(args.vision_config),
        "preprocessor_config": str(args.preprocessor_config),
        "mask_dir": str(args.mask_dir),
        "image_aspect_ratio": image_aspect_ratio,
        "image_size": image_size,
        "patch_size": patch_size,
        "grid_size": grid_size,
        "num_patches_per_image": grid_size * grid_size,
        "patch_order": "row-major (coverage_ratio[i, row, col])",
        "coverage_ratio_definition": "fraction of each visual patch covered by the binary mask after LLaVA's pad-to-square preprocessing",
        "contains_mask_definition": "coverage_ratio > 0",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        metadata_json=np.array(json.dumps(metadata, ensure_ascii=True)),
        image_names=image_names,
        image_ids=image_ids,
        original_widths=original_widths,
        original_heights=original_heights,
        padded_sides=padded_sides,
        pad_tops=pad_tops,
        pad_lefts=pad_lefts,
        coverage_ratio=coverage_ratio,
        has_mask=has_mask,
    )

    mean_masked_patches = total_masked_patches / max(num_images, 1)
    mean_patch_coverage = total_mask_coverage / max(num_images * grid_size * grid_size, 1)
    print(f"done: images={num_images} output={args.output}")
    print(f"grid={grid_size}x{grid_size} patch_size={patch_size} image_size={image_size}")
    print(f"mean_masked_patches_per_image={mean_masked_patches:.4f}")
    print(f"mean_patch_coverage_over_all_patches={mean_patch_coverage:.6f}")


if __name__ == "__main__":
    main()
