import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate output-mask style images by blacking out COCO instance segmentations for visual cues."
    )
    parser.add_argument(
        "--mapping-json",
        default="/path/to/sage_repro_bundle/merged_output_rule_mapping.json",
        help="Path to merged_output_rule_mapping.json.",
    )
    parser.add_argument(
        "--coco-instances",
        default="/path/to/sage_repro_bundle/object_annotation_bundle/coco/instances_train2017.json",
        help="Path to COCO instances annotation JSON.",
    )
    parser.add_argument(
        "--image-dir",
        default="data/images/coco/train2014",
        help="Directory containing original COCO train2014 images.",
    )
    parser.add_argument(
        "--output-dir",
        default="/path/to/sage_repro_bundle/output_mask_coco_seg",
        help="Directory to write generated masked images.",
    )
    parser.add_argument(
        "--stats-json",
        default="/path/to/sage_repro_bundle/output_mask_coco_seg.stats.json",
        help="Path to write generation stats.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N eligible samples.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip outputs that already exist.",
    )
    parser.add_argument(
        "--only-with-cues",
        action="store_true",
        help="Only emit images when at least one COCO instance matches visual_cues. Otherwise skip.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for written images.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress every N processed items.",
    )
    return parser.parse_args()


def load_mapping(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)["results"]


def load_coco(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
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
    size = segmentation["size"]
    rle_h, rle_w = int(size[0]), int(size[1])
    flat = np.zeros(rle_h * rle_w, dtype=np.uint8)
    idx = 0
    val = 0
    for run in counts:
        run = int(run)
        if idx >= flat.size:
            break
        end = min(idx + max(run, 0), flat.size)
        if val == 1 and end > idx:
            flat[idx:end] = 1
        idx = end
        val = 1 - val
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


def main() -> None:
    args = parse_args()
    mapping = load_mapping(Path(args.mapping_json))
    anns_by_image = load_coco(Path(args.coco_instances))
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counters = Counter()
    counters["mapping_items"] = len(mapping)

    for idx, entry in enumerate(mapping, start=1):
        if args.limit is not None and counters["processed"] >= args.limit:
            break

        question_id = int(entry["question_id"])
        image_id = int(entry["image_id"])
        cues = {str(x).lower() for x in entry.get("visual_cues", []) if str(x).strip()}
        src_path = image_path(image_dir, image_id)
        out_path = output_dir / f"{question_id}_{image_id}.jpg"

        if args.skip_existing and out_path.exists():
            counters["processed"] += 1
            counters["skipped_existing"] += 1
            continue
        if not src_path.exists():
            counters["missing_source_image"] += 1
            continue

        matched_anns = []
        for ann in anns_by_image.get(image_id, []):
            if ann["category_name"] in cues:
                matched_anns.append(ann)

        if not matched_anns:
            if args.only_with_cues:
                counters["skipped_no_matching_cue"] += 1
                continue
            image = Image.open(src_path).convert("RGB")
            image.save(out_path, quality=args.jpeg_quality)
            counters["processed"] += 1
            counters["written_passthrough"] += 1
            if args.progress_every > 0 and counters["processed"] % args.progress_every == 0:
                print(
                    f"processed={counters['processed']} written={counters['written_masked'] + counters['written_passthrough']} "
                    f"matched={counters['written_masked']}"
                )
            continue

        image = Image.open(src_path).convert("RGB")
        image_np = np.asarray(image).copy()
        height, width = image_np.shape[:2]
        union_mask = np.zeros((height, width), dtype=bool)
        for ann in matched_anns:
            union_mask |= segmentation_to_mask(ann["segmentation"], width=width, height=height)

        image_np[union_mask] = 0
        Image.fromarray(image_np).save(out_path, quality=args.jpeg_quality)

        counters["processed"] += 1
        counters["written_masked"] += 1
        counters["matched_instances_total"] += len(matched_anns)
        counters["matched_pixels_total"] += int(union_mask.sum())
        if args.progress_every > 0 and counters["processed"] % args.progress_every == 0:
            print(
                f"processed={counters['processed']} written={counters['written_masked'] + counters['written_passthrough']} "
                f"matched={counters['written_masked']}"
            )

    stats = {
        "mapping_json": args.mapping_json,
        "coco_instances": args.coco_instances,
        "image_dir": args.image_dir,
        "output_dir": args.output_dir,
        "limit": args.limit,
        "skip_existing": args.skip_existing,
        "only_with_cues": args.only_with_cues,
        "jpeg_quality": args.jpeg_quality,
        "counts": dict(counters),
    }
    Path(args.stats_json).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
