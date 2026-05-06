import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract likely-unanswerable train_raw_llava questions under a masked-image setting."
    )
    parser.add_argument(
        "--train-json",
        default="/path/to/sage_repro_bundle/train_raw.json",
        help="Path to train_raw-style JSON.",
    )
    parser.add_argument(
        "--mask-dir",
        default="/path/to/sage_repro_bundle/output_mask_coco_seg",
        help="Directory containing masked images named question_id_image_id.jpg.",
    )
    parser.add_argument(
        "--image-dir",
        default="data/images/coco/train2014",
        help="Directory containing original COCO train2014 images.",
    )
    parser.add_argument(
        "--coco-instances",
        default="/path/to/sage_repro_bundle/object_annotation_bundle/coco/instances_train2017.json",
        help="Path to COCO instances annotation JSON.",
    )
    parser.add_argument(
        "--output-json",
        default="/path/to/sage_repro_bundle/train_raw_llava_likely_unanswerable.json",
        help="Path to save extracted samples.",
    )
    parser.add_argument(
        "--report-json",
        default="/path/to/sage_repro_bundle/train_raw_llava_likely_unanswerable.report.json",
        help="Path to save extraction report.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N samples.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress every N samples.",
    )
    return parser.parse_args()


def load_train(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_coco(path: Path):
    with path.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    cat_name_by_id = {int(c["id"]): c["name"].lower() for c in coco["categories"]}
    all_cats = sorted(set(cat_name_by_id.values()), key=len, reverse=True)
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[int(ann["image_id"])].append(
            {
                "category_name": cat_name_by_id[int(ann["category_id"])],
                "segmentation": ann["segmentation"],
            }
        )
    return all_cats, anns_by_image


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


def seg_to_mask(segmentation, width: int, height: int) -> np.ndarray:
    if isinstance(segmentation, list):
        canvas = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(canvas)
        for poly in segmentation:
            if len(poly) >= 6:
                pts = [(float(poly[i]), float(poly[i + 1])) for i in range(0, len(poly), 2)]
                draw.polygon(pts, fill=1, outline=1)
        return np.asarray(canvas).astype(bool)
    return decode_uncompressed_rle(segmentation, width, height)


def build_mask_binary(orig_path: Path, masked_path: Path) -> np.ndarray:
    original = Image.open(orig_path).convert("RGB")
    masked = Image.open(masked_path).convert("RGB")
    if masked.size != original.size:
        masked = masked.resize(original.size)
    original_np = np.asarray(original, dtype=np.int16)
    masked_np = np.asarray(masked, dtype=np.int16)
    brightness_drop = original_np.mean(axis=2) - masked_np.mean(axis=2)
    return (brightness_drop > 40.0) | (masked_np.max(axis=2) < 30)


def mention_categories(question: str, all_cats: list[str]) -> list[str]:
    q = question.lower()
    hits = []
    for cat in all_cats:
        pattern = r"\b" + re.escape(cat) + r"(?:s|es)?\b"
        if re.search(pattern, q):
            hits.append(cat)
    return sorted(set(hits))


def classify_sample(
    item: dict,
    all_cats: list[str],
    anns_by_image: dict[int, list[dict]],
    mask_dir: Path,
    image_dir: Path,
) -> dict | None:
    qid = int(item["question_id"])
    iid = int(item["image_id"])
    orig_path = image_dir / f"COCO_train2014_{iid:012d}.jpg"
    masked_path = mask_dir / f"{qid}_{iid}.jpg"
    if not orig_path.exists() or not masked_path.exists():
        return None

    q = item["question"]
    ql = q.lower()
    mentions = mention_categories(q, all_cats)
    mask = build_mask_binary(orig_path, masked_path)
    mask_ratio = float(mask.mean())

    mention_stats = []
    max_cov = 0.0
    max_iou = 0.0
    for cat in mentions:
        best = None
        for ann in anns_by_image.get(iid, []):
            if ann["category_name"] != cat:
                continue
            seg = seg_to_mask(ann["segmentation"], width=mask.shape[1], height=mask.shape[0])
            seg_area = int(seg.sum())
            if seg_area <= 0:
                continue
            overlap = int(np.logical_and(mask, seg).sum())
            union = int(np.logical_or(mask, seg).sum())
            cov = float(overlap / max(seg_area, 1))
            share = float(overlap / max(int(mask.sum()), 1))
            iou = float(overlap / max(union, 1))
            candidate = {
                "category": cat,
                "seg_coverage": cov,
                "mask_share": share,
                "iou": iou,
            }
            if best is None or (cov, iou, share) > (
                best["seg_coverage"],
                best["iou"],
                best["mask_share"],
            ):
                best = candidate
        if best is not None:
            mention_stats.append(best)
            max_cov = max(max_cov, best["seg_coverage"])
            max_iou = max(max_iou, best["iou"])

    reasons = []
    if not mentions:
        reasons.append("no_explicit_coco_object_in_question")
    if mask_ratio > 0.45:
        reasons.append("large_mask")
    if mention_stats:
        if max_cov >= 0.7 and max_iou >= 0.15:
            reasons.append("question_object_strongly_masked")
        elif max_cov >= 0.35 and max_iou >= 0.05:
            reasons.append("question_object_partially_masked")
        else:
            reasons.append("question_object_mostly_visible")

    if any(x in ql for x in ("how many", "number of", "many")):
        reasons.append("count_question")
    reasons.append("attribute_or_relation_question")

    label = "likely_answerable"
    if (
        "question_object_strongly_masked" in reasons
        and ("count_question" in reasons or "attribute_or_relation_question" in reasons)
    ):
        label = "likely_unanswerable"
    elif (
        "question_object_partially_masked" in reasons
        and ("count_question" in reasons or "attribute_or_relation_question" in reasons)
    ):
        label = "borderline"
    elif not mentions and mask_ratio < 0.25:
        label = "likely_answerable"
    elif "large_mask" in reasons and item.get("answer_type") != "yes/no":
        label = "borderline"

    if "question_object_mostly_visible" in reasons:
        label = "likely_answerable"

    return {
        "question_id": qid,
        "image_id": iid,
        "question": q,
        "answer": item["answer"],
        "answer_type": item["answer_type"],
        "mentions": mentions,
        "mention_mask_stats": mention_stats,
        "mask_ratio": mask_ratio,
        "label": label,
        "reasons": reasons,
    }


def main() -> None:
    args = parse_args()
    train = load_train(Path(args.train_json))
    all_cats, anns_by_image = load_coco(Path(args.coco_instances))
    mask_dir = Path(args.mask_dir)
    image_dir = Path(args.image_dir)

    extracted = []
    evaluated = 0
    missing_mask = 0
    label_counter = Counter()
    reason_counter = Counter()

    for idx, item in enumerate(train, start=1):
        if args.limit is not None and evaluated >= args.limit:
            break
        result = classify_sample(item, all_cats, anns_by_image, mask_dir, image_dir)
        if result is None:
            missing_mask += 1
            continue
        evaluated += 1
        label_counter[result["label"]] += 1
        for reason in result["reasons"]:
            reason_counter[reason] += 1
        if result["label"] == "likely_unanswerable":
            extracted.append(item)
        if args.progress_every > 0 and evaluated % args.progress_every == 0:
            print(
                f"evaluated={evaluated} likely_unanswerable={label_counter['likely_unanswerable']} "
                f"borderline={label_counter['borderline']} answerable={label_counter['likely_answerable']}"
            )

    Path(args.output_json).write_text(json.dumps(extracted, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "train_json": args.train_json,
        "mask_dir": args.mask_dir,
        "coco_instances": args.coco_instances,
        "evaluated": evaluated,
        "missing_mask_or_image": missing_mask,
        "label_counts": dict(label_counter),
        "reason_counts": dict(reason_counter),
        "output_json": args.output_json,
    }
    Path(args.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
