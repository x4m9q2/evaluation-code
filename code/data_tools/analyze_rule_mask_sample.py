import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample rule-mask pairs and measure mask overlap against COCO boxes for visual cues."
    )
    parser.add_argument(
        "--mapping-json",
        default="/path/to/sage_repro_bundle/merged_output_rule_mapping.json",
        help="Path to merged_output_rule_mapping.json.",
    )
    parser.add_argument(
        "--mask-dir",
        default="/path/to/sage_repro_bundle/output_mask",
        help="Directory containing output mask images.",
    )
    parser.add_argument(
        "--image-dir",
        default="data/images/coco/train2014",
        help="Directory containing original COCO train2014 images.",
    )
    parser.add_argument(
        "--coco-instances",
        default="/path/to/sage_repro_bundle/object_annotation_bundle/coco/instances_train2017.json",
        help="COCO instances annotation file.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of items to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260414,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--output-json",
        default="/path/to/sage_repro_bundle/rule_mask_sample100_summary.json",
        help="Path to write machine-readable summary.",
    )
    parser.add_argument(
        "--output-md",
        default="/path/to/sage_repro_bundle/rule_mask_sample100_summary.md",
        help="Path to write Markdown summary.",
    )
    return parser.parse_args()


def load_mapping(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["results"]


def load_coco(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    category_name_by_id = {int(cat["id"]): cat["name"].lower() for cat in data["categories"]}
    image_ids = {int(img["id"]) for img in data["images"]}
    anns_by_image = defaultdict(list)
    for ann in data["annotations"]:
        anns_by_image[int(ann["image_id"])].append(
            {
                "bbox": ann["bbox"],
                "segmentation": ann["segmentation"],
                "category_name": category_name_by_id[int(ann["category_id"])],
                "iscrowd": int(ann.get("iscrowd", 0)),
                "area": float(ann.get("area", 0.0)),
            }
        )
    return image_ids, anns_by_image


def image_path(image_dir: Path, image_id: int) -> Path:
    return image_dir / f"COCO_train2014_{image_id:012d}.jpg"


def build_mask_binary(original_path: Path, mask_path: Path) -> np.ndarray:
    original = Image.open(original_path).convert("RGB")
    masked = Image.open(mask_path).convert("RGB")
    if original.size != masked.size:
        original = original.resize(masked.size, Image.Resampling.BILINEAR)
    original_np = np.asarray(original, dtype=np.int16)
    masked_np = np.asarray(masked, dtype=np.int16)
    brightness_drop = original_np.mean(axis=2) - masked_np.mean(axis=2)
    binary = (brightness_drop > 40.0) | (masked_np.max(axis=2) < 30)
    return binary


def bbox_to_slices(bbox: list[float], width: int, height: int):
    x, y, w, h = bbox
    x0 = max(0, int(np.floor(x)))
    y0 = max(0, int(np.floor(y)))
    x1 = min(width, int(np.ceil(x + w)))
    y1 = min(height, int(np.ceil(y + h)))
    if x1 <= x0 or y1 <= y0:
        return None
    return y0, y1, x0, x1


def decode_uncompressed_rle(segmentation: dict, width: int, height: int) -> np.ndarray:
    counts = segmentation["counts"]
    size = segmentation["size"]
    if len(size) != 2:
        raise ValueError(f"Unexpected RLE size: {size}")
    rle_h, rle_w = int(size[0]), int(size[1])
    flat = np.zeros(rle_h * rle_w, dtype=np.uint8)
    idx = 0
    val = 0
    for run in counts:
        run = int(run)
        if run < 0:
            raise ValueError(f"Negative run length in RLE: {run}")
        if idx + run > flat.size:
            run = flat.size - idx
        if val == 1 and run > 0:
            flat[idx : idx + run] = 1
        idx += run
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


def evaluate_sample(entry: dict, mask_dir: Path, image_dir: Path, anns_by_image: dict[int, list[dict]]) -> dict:
    question_id = int(entry["question_id"])
    image_id = int(entry["image_id"])
    cues = [str(x).lower() for x in entry.get("visual_cues", [])]
    original_path = image_path(image_dir, image_id)
    mask_path = mask_dir / f"{question_id}_{image_id}.jpg"

    result = {
        "question_id": question_id,
        "image_id": image_id,
        "answer": entry.get("answer"),
        "text_keywords": entry.get("text_keywords", []),
        "visual_cues": entry.get("visual_cues", []),
        "matched_rule": entry.get("matched_rule"),
        "original_path": str(original_path),
        "mask_path": str(mask_path),
        "status": "ok",
    }

    if not original_path.exists() or not mask_path.exists():
        result["status"] = "missing_file"
        return result

    mask_binary = build_mask_binary(original_path, mask_path)
    height, width = mask_binary.shape
    mask_area = int(mask_binary.sum())
    result["mask_area_pixels"] = mask_area
    result["mask_area_ratio"] = float(mask_area / max(height * width, 1))

    anns = anns_by_image.get(image_id, [])
    cue_anns = [ann for ann in anns if ann["category_name"] in cues]
    result["coco_boxes_total"] = len(anns)
    result["coco_boxes_matching_visual_cues"] = len(cue_anns)
    result["matching_categories_found"] = sorted({ann["category_name"] for ann in cue_anns})

    if not cue_anns:
        result["status"] = "no_matching_coco_box"
        return result

    best = None
    for ann in cue_anns:
        seg_mask = segmentation_to_mask(ann["segmentation"], width=width, height=height)
        seg_area = int(seg_mask.sum())
        if seg_area <= 0:
            continue
        overlap = int(np.logical_and(mask_binary, seg_mask).sum())
        union = int(np.logical_or(mask_binary, seg_mask).sum())
        seg_coverage = float(overlap / max(seg_area, 1))
        mask_share = float(overlap / max(mask_area, 1))
        iou = float(overlap / max(union, 1))
        slices = bbox_to_slices(ann["bbox"], width=width, height=height)
        bbox_area = None
        if slices is not None:
            y0, y1, x0, x1 = slices
            bbox_area = int((y1 - y0) * (x1 - x0))
        candidate = {
            "category_name": ann["category_name"],
            "bbox": [float(v) for v in ann["bbox"]],
            "bbox_area_pixels": bbox_area,
            "segmentation_area_pixels": seg_area,
            "annotation_area": ann["area"],
            "overlap_pixels": overlap,
            "segmentation_mask_coverage": seg_coverage,
            "mask_inside_bbox_share": mask_share,
            "iou": iou,
        }
        if best is None or (
            candidate["segmentation_mask_coverage"],
            candidate["iou"],
            candidate["overlap_pixels"],
        ) > (
            best["segmentation_mask_coverage"],
            best["iou"],
            best["overlap_pixels"],
        ):
            best = candidate

    if best is None:
        result["status"] = "invalid_bbox"
        return result

    result["best_match"] = best
    return result


def label_quality(sample: dict) -> str:
    if sample["status"] != "ok":
        return sample["status"]
    coverage = sample["best_match"]["segmentation_mask_coverage"]
    mask_share = sample["best_match"]["mask_inside_bbox_share"]
    mask_area_ratio = sample["mask_area_ratio"]
    if coverage >= 0.5 and mask_share >= 0.15 and mask_area_ratio <= 0.6:
        return "strong"
    if coverage >= 0.2 and mask_share >= 0.03 and mask_area_ratio <= 0.75:
        return "partial"
    return "weak"


def summarize(samples: list[dict], args: argparse.Namespace) -> dict:
    quality = Counter()
    trigger_counter = Counter()
    cue_counter = Counter()
    status_counter = Counter(sample["status"] for sample in samples)
    mask_ratios = []
    segmentation_coverages = []
    mask_shares = []
    ious = []

    for sample in samples:
        if sample["status"] == "ok":
            q = label_quality(sample)
            quality[q] += 1
            mask_ratios.append(sample["mask_area_ratio"])
            segmentation_coverages.append(sample["best_match"]["segmentation_mask_coverage"])
            mask_shares.append(sample["best_match"]["mask_inside_bbox_share"])
            ious.append(sample["best_match"]["iou"])
        else:
            quality[sample["status"]] += 1
        rule = sample.get("matched_rule") or {}
        trigger = str(rule.get("trigger", ""))
        if trigger:
            trigger_counter[trigger] += 1
        for cue in sample.get("visual_cues", []):
            cue_counter[str(cue).lower()] += 1

    def stats(values: list[float]) -> dict:
        if not values:
            return {}
        arr = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    sortable = []
    for sample in samples:
        if sample["status"] != "ok":
            continue
        sortable.append(
            {
                "question_id": sample["question_id"],
                "image_id": sample["image_id"],
                "quality": label_quality(sample),
                "trigger": (sample.get("matched_rule") or {}).get("trigger"),
                "visual_cues": sample.get("visual_cues", []),
                "answer": sample.get("answer"),
                "mask_area_ratio": sample.get("mask_area_ratio"),
                "segmentation_mask_coverage": sample["best_match"]["segmentation_mask_coverage"],
                "mask_inside_bbox_share": sample["best_match"]["mask_inside_bbox_share"],
                "iou": sample["best_match"]["iou"],
            }
        )
    strong_examples = [x for x in sortable if x["quality"] == "strong"][:10]
    weak_examples = [x for x in sortable if x["quality"] == "weak"][:10]
    best_overlap_examples = sorted(
        sortable,
        key=lambda x: (x["segmentation_mask_coverage"], x["iou"], x["mask_inside_bbox_share"]),
        reverse=True,
    )[:10]
    worst_overlap_examples = sorted(
        sortable,
        key=lambda x: (x["segmentation_mask_coverage"], x["iou"], x["mask_inside_bbox_share"]),
    )[:10]

    return {
        "config": {
            "mapping_json": args.mapping_json,
            "mask_dir": args.mask_dir,
            "image_dir": args.image_dir,
            "coco_instances": args.coco_instances,
            "sample_size": args.sample_size,
            "seed": args.seed,
        },
        "summary": {
            "sampled_items": len(samples),
            "status_counts": dict(status_counter),
            "quality_counts": dict(quality),
            "quality_rate_over_valid": {
                k: float(v / max(status_counter.get("ok", 0), 1))
                for k, v in quality.items()
                if k in {"strong", "partial", "weak"}
            },
            "mask_area_ratio_stats": stats(mask_ratios),
            "segmentation_mask_coverage_stats": stats(segmentation_coverages),
            "mask_inside_bbox_share_stats": stats(mask_shares),
            "iou_stats": stats(ious),
            "top_triggers": trigger_counter.most_common(15),
            "top_visual_cues": cue_counter.most_common(15),
            "strong_examples": strong_examples,
            "weak_examples": weak_examples,
            "best_overlap_examples": best_overlap_examples,
            "worst_overlap_examples": worst_overlap_examples,
        },
        "samples": samples,
    }


def render_markdown(report: dict) -> str:
    summary = report["summary"]
    lines = []
    lines.append("# Rule-Mask Sample Summary")
    lines.append("")
    lines.append("## Config")
    for key, value in report["config"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Aggregate")
    lines.append(f"- Sampled items: `{summary['sampled_items']}`")
    lines.append(f"- Status counts: `{json.dumps(summary['status_counts'], ensure_ascii=False)}`")
    lines.append(f"- Quality counts: `{json.dumps(summary['quality_counts'], ensure_ascii=False)}`")
    lines.append(f"- Quality rate over valid: `{json.dumps(summary['quality_rate_over_valid'], ensure_ascii=False)}`")
    lines.append(f"- Mask area ratio stats: `{json.dumps(summary['mask_area_ratio_stats'], ensure_ascii=False)}`")
    lines.append(f"- Segmentation mask coverage stats: `{json.dumps(summary['segmentation_mask_coverage_stats'], ensure_ascii=False)}`")
    lines.append(f"- Mask-inside-bbox share stats: `{json.dumps(summary['mask_inside_bbox_share_stats'], ensure_ascii=False)}`")
    lines.append(f"- IoU stats: `{json.dumps(summary['iou_stats'], ensure_ascii=False)}`")
    lines.append("")
    lines.append("## Top Triggers")
    for trigger, count in summary["top_triggers"]:
        lines.append(f"- `{trigger}`: `{count}`")
    lines.append("")
    lines.append("## Top Visual Cues")
    for cue, count in summary["top_visual_cues"]:
        lines.append(f"- `{cue}`: `{count}`")
    lines.append("")
    lines.append("## Strong Examples")
    for item in summary["strong_examples"]:
        lines.append(
            f"- qid=`{item['question_id']}` img=`{item['image_id']}` cue=`{item['visual_cues']}` "
            f"trigger=`{item['trigger']}` answer=`{item['answer']}` "
            f"seg_cov=`{item['segmentation_mask_coverage']:.4f}` mask_share=`{item['mask_inside_bbox_share']:.4f}` "
            f"iou=`{item['iou']:.4f}` "
            f"mask_ratio=`{item['mask_area_ratio']:.4f}`"
        )
    lines.append("")
    lines.append("## Weak Examples")
    for item in summary["weak_examples"]:
        lines.append(
            f"- qid=`{item['question_id']}` img=`{item['image_id']}` cue=`{item['visual_cues']}` "
            f"trigger=`{item['trigger']}` answer=`{item['answer']}` "
            f"seg_cov=`{item['segmentation_mask_coverage']:.4f}` mask_share=`{item['mask_inside_bbox_share']:.4f}` "
            f"iou=`{item['iou']:.4f}` "
            f"mask_ratio=`{item['mask_area_ratio']:.4f}`"
        )
    lines.append("")
    lines.append("## Best Overlap Examples")
    for item in summary["best_overlap_examples"]:
        lines.append(
            f"- qid=`{item['question_id']}` img=`{item['image_id']}` cue=`{item['visual_cues']}` "
            f"trigger=`{item['trigger']}` answer=`{item['answer']}` "
            f"quality=`{item['quality']}` "
            f"seg_cov=`{item['segmentation_mask_coverage']:.4f}` mask_share=`{item['mask_inside_bbox_share']:.4f}` "
            f"iou=`{item['iou']:.4f}` "
            f"mask_ratio=`{item['mask_area_ratio']:.4f}`"
        )
    lines.append("")
    lines.append("## Worst Overlap Examples")
    for item in summary["worst_overlap_examples"]:
        lines.append(
            f"- qid=`{item['question_id']}` img=`{item['image_id']}` cue=`{item['visual_cues']}` "
            f"trigger=`{item['trigger']}` answer=`{item['answer']}` "
            f"quality=`{item['quality']}` "
            f"seg_cov=`{item['segmentation_mask_coverage']:.4f}` mask_share=`{item['mask_inside_bbox_share']:.4f}` "
            f"iou=`{item['iou']:.4f}` "
            f"mask_ratio=`{item['mask_area_ratio']:.4f}`"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    mapping_path = Path(args.mapping_json)
    mask_dir = Path(args.mask_dir)
    image_dir = Path(args.image_dir)
    coco_path = Path(args.coco_instances)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)

    results = load_mapping(mapping_path)
    image_ids, anns_by_image = load_coco(coco_path)
    rng = random.Random(args.seed)

    eligible = []
    for entry in results:
        image_id = int(entry["image_id"])
        question_id = int(entry["question_id"])
        cues = [str(x).lower() for x in entry.get("visual_cues", [])]
        if not cues:
            continue
        if image_id not in image_ids:
            continue
        if not anns_by_image.get(image_id):
            continue
        if not image_path(image_dir, image_id).exists():
            continue
        if not (mask_dir / f"{question_id}_{image_id}.jpg").exists():
            continue
        eligible.append(entry)

    if len(eligible) < args.sample_size:
        raise ValueError(f"Eligible samples {len(eligible)} < requested sample size {args.sample_size}")

    sampled = rng.sample(eligible, args.sample_size)
    evaluated = [evaluate_sample(entry, mask_dir, image_dir, anns_by_image) for entry in sampled]
    report = summarize(evaluated, args)

    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(render_markdown(report), encoding="utf-8")

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
