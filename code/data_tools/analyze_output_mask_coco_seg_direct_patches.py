import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


DEFAULT_MAPPING_JSON = Path("/path/to/sage_repro_bundle/merged_output_rule_mapping.json")
DEFAULT_COCO_INSTANCES = Path("/path/to/sage_repro_bundle/object_annotation_bundle/coco/instances_train2017.json")
DEFAULT_IMAGE_DIR = Path("data/images/coco/train2014")
DEFAULT_MODEL_CONFIG = Path("/path/to/sage_repro_bundle/llava-v1.5-7b/config.json")
DEFAULT_VISION_CONFIG = Path("/path/to/sage_repro_bundle/clip-vit-large-patch14-336/config.json")
DEFAULT_PREPROCESSOR_CONFIG = Path("/path/to/sage_repro_bundle/clip-vit-large-patch14-336/preprocessor_config.json")
DEFAULT_OUTPUT = Path("/path/to/sage_repro_bundle/patch_mask_analysis_output_mask_coco_seg_direct_llava_pad336_patch14.npz")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate patch-level mask coverage directly from COCO segmentation masks used by "
            "output_mask_coco_seg, without reverse-engineering rendered JPG masks."
        )
    )
    parser.add_argument("--mapping-json", type=Path, default=DEFAULT_MAPPING_JSON)
    parser.add_argument("--coco-instances", type=Path, default=DEFAULT_COCO_INSTANCES)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--vision-config", type=Path, default=DEFAULT_VISION_CONFIG)
    parser.add_argument("--preprocessor-config", type=Path, default=DEFAULT_PREPROCESSOR_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-train-json", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--only-with-cues", action="store_true")
    parser.add_argument(
        "--shrink-patch-rings",
        type=int,
        default=0,
        help="Shrink the binary patch mask inward by N patch rings after computing patch coverage.",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_mapping(path: Path) -> list[dict]:
    data = load_json(path)
    return data["results"]


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


def load_coco(path: Path):
    data = load_json(path)
    cat_name_by_id = {int(x["id"]): str(x["name"]).lower() for x in data["categories"]}
    anns_by_image = defaultdict(list)
    for ann in data["annotations"]:
        anns_by_image[int(ann["image_id"])].append(
            {
                "category_name": cat_name_by_id[int(ann["category_id"])],
                "segmentation": ann["segmentation"],
            }
        )
    return anns_by_image


def image_path(image_dir: Path, image_id: int) -> Path:
    return image_dir / f"COCO_train2014_{image_id:012d}.jpg"


def decode_uncompressed_rle(segmentation: dict, width: int, height: int) -> np.ndarray:
    counts = segmentation["counts"]
    rle_h, rle_w = map(int, segmentation["size"])
    flat = np.zeros(rle_h * rle_w, dtype=np.uint8)
    idx = 0
    val = 0
    for run in counts:
        run = int(run)
        end = min(idx + max(run, 0), flat.size)
        if val == 1 and end > idx:
            flat[idx:end] = 1
        idx = end
        val = 1 - val
        if idx >= flat.size:
            break
    mask = flat.reshape((rle_w, rle_h)).T
    if (rle_w, rle_h) != (width, height):
        mask_img = Image.fromarray(mask * 255)
        mask_img = mask_img.resize((width, height), Image.Resampling.NEAREST)
        mask = (np.asarray(mask_img) > 0).astype(np.uint8)
    return mask.astype(bool)


def segmentation_to_mask(segmentation, width: int, height: int) -> np.ndarray:
    if isinstance(segmentation, list):
        canvas = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(canvas)
        for poly in segmentation:
            if len(poly) < 6:
                continue
            pts = [(float(poly[i]), float(poly[i + 1])) for i in range(0, len(poly), 2)]
            draw.polygon(pts, fill=1, outline=1)
        return np.asarray(canvas, dtype=np.uint8).astype(bool)
    if isinstance(segmentation, dict):
        return decode_uncompressed_rle(segmentation, width=width, height=height)
    raise TypeError(f"Unsupported segmentation type: {type(segmentation).__name__}")


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


def shrink_patch_mask_rings(coverage: np.ndarray, rings: int) -> tuple[np.ndarray, np.ndarray]:
    if rings <= 0:
        has_mask = coverage > 0.0
        return coverage.astype(np.float32), has_mask

    active = (coverage > 0.0).astype(np.float32)
    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32)
    active_tensor = torch.from_numpy(active).unsqueeze(0).unsqueeze(0)
    for _ in range(rings):
        neighbor_count = F.conv2d(active_tensor, kernel, padding=1)
        active_tensor = (neighbor_count == 9).to(dtype=torch.float32)

    shrunken = active_tensor.squeeze(0).squeeze(0).numpy().astype(bool)
    coverage = np.where(shrunken, coverage, 0.0).astype(np.float32)
    return coverage, shrunken


def analyze_single(task):
    entry, anns_by_image, image_dir, grid_size, only_with_cues, shrink_patch_rings = task
    question_id = int(entry["question_id"])
    image_id = int(entry["image_id"])
    cues = {str(x).lower() for x in entry.get("visual_cues", []) if str(x).strip()}

    src_path = image_path(image_dir, image_id)
    if not src_path.exists():
        return {"status": "missing_image", "question_id": question_id, "image_id": image_id}

    image = Image.open(src_path).convert("RGB")
    width, height = image.size

    matched_anns = []
    for ann in anns_by_image.get(image_id, []):
        if ann["category_name"] in cues:
            matched_anns.append(ann)

    if not matched_anns and only_with_cues:
        return {"status": "no_matching_cue", "question_id": question_id, "image_id": image_id}

    union_mask = np.zeros((height, width), dtype=bool)
    for ann in matched_anns:
        union_mask |= segmentation_to_mask(ann["segmentation"], width=width, height=height)

    square_mask, pad_top, pad_left, side = pad_to_square(union_mask.astype(np.float32))
    coverage = compute_patch_coverage(square_mask, grid_size)
    coverage, has_mask = shrink_patch_mask_rings(coverage, shrink_patch_rings)

    return {
        "status": "ok",
        "image_name": f"{question_id}_{image_id}.jpg",
        "question_id": question_id,
        "image_id": image_id,
        "original_width": width,
        "original_height": height,
        "padded_side": int(side),
        "pad_top": int(pad_top),
        "pad_left": int(pad_left),
        "mask_pixel_count": int(union_mask.sum()),
        "matched_instance_count": len(matched_anns),
        "coverage": coverage,
        "has_mask": has_mask,
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

    mapping = load_mapping(args.mapping_json)
    if args.include_train_json is not None:
        allowed_qids = load_allowed_question_ids(args.include_train_json)
        mapping = [row for row in mapping if int(row["question_id"]) in allowed_qids]
    if args.limit is not None:
        mapping = mapping[: args.limit]

    anns_by_image = load_coco(args.coco_instances)

    requested_images = len(mapping)
    image_names = np.empty(requested_images, dtype=object)
    question_ids = np.empty(requested_images, dtype=np.int64)
    image_ids = np.empty(requested_images, dtype=np.int64)
    original_widths = np.empty(requested_images, dtype=np.int32)
    original_heights = np.empty(requested_images, dtype=np.int32)
    padded_sides = np.empty(requested_images, dtype=np.int32)
    pad_tops = np.empty(requested_images, dtype=np.int32)
    pad_lefts = np.empty(requested_images, dtype=np.int32)
    mask_pixel_counts = np.empty(requested_images, dtype=np.int64)
    matched_instance_counts = np.empty(requested_images, dtype=np.int32)
    coverage_ratio = np.empty((requested_images, grid_size, grid_size), dtype=np.float32)
    has_mask = np.empty((requested_images, grid_size, grid_size), dtype=np.bool_)

    tasks = [
        (entry, anns_by_image, args.image_dir, grid_size, args.only_with_cues, args.shrink_patch_rings)
        for entry in mapping
    ]

    if args.workers <= 1:
        result_iter = map(analyze_single, tasks)
    else:
        executor = ThreadPoolExecutor(max_workers=args.workers)
        result_iter = executor.map(analyze_single, tasks)

    total_masked_patches = 0
    total_mask_coverage = 0.0
    total_mask_pixels = 0
    missing_count = 0
    no_matching_cue_count = 0
    write_idx = 0
    try:
        for read_idx, result in enumerate(result_iter):
            status = result["status"]
            if status == "missing_image":
                missing_count += 1
                continue
            if status == "no_matching_cue":
                no_matching_cue_count += 1
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
            matched_instance_counts[write_idx] = result["matched_instance_count"]
            coverage_ratio[write_idx] = result["coverage"]
            has_mask[write_idx] = result["has_mask"]
            write_idx += 1

            total_masked_patches += int(result["has_mask"].sum())
            total_mask_coverage += float(result["coverage"].sum())
            total_mask_pixels += int(result["mask_pixel_count"])

            if (read_idx + 1) % 1000 == 0:
                print(f"processed={read_idx + 1}/{requested_images} kept={write_idx}")
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
    matched_instance_counts = matched_instance_counts[:num_images]
    coverage_ratio = coverage_ratio[:num_images]
    has_mask = has_mask[:num_images]

    metadata = {
        "mapping_json": str(args.mapping_json),
        "coco_instances": str(args.coco_instances),
        "image_dir": str(args.image_dir),
        "model_config": str(args.model_config),
        "vision_config": str(args.vision_config),
        "preprocessor_config": str(args.preprocessor_config),
        "image_aspect_ratio": image_aspect_ratio,
        "image_size": image_size,
        "patch_size": patch_size,
        "grid_size": grid_size,
        "num_patches_per_image": grid_size * grid_size,
        "patch_order": "row-major (coverage_ratio[i, row, col])",
        "coverage_ratio_definition": (
            "fraction of each visual patch covered by the union of matched COCO segmentation masks "
            "after LLaVA's pad-to-square preprocessing"
        ),
        "contains_mask_definition": "coverage_ratio > 0",
        "shrink_patch_rings": int(args.shrink_patch_rings),
        "mask_source": "direct_coco_segmentation_union_for_visual_cues",
        "include_train_json": str(args.include_train_json) if args.include_train_json else None,
        "requested_images": requested_images,
        "analyzed_images": num_images,
        "missing_original_images": missing_count,
        "skipped_no_matching_cue": no_matching_cue_count,
        "only_with_cues": bool(args.only_with_cues),
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
        matched_instance_counts=matched_instance_counts,
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
