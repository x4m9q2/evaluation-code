import argparse
import json
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter train_raw by direct COCO segmentation mask overlap labels."
    )
    parser.add_argument("--train-json", default="/path/to/sage_repro_bundle/train_raw.json")
    parser.add_argument("--mapping-json", default="/path/to/sage_repro_bundle/merged_output_rule_mapping.json")
    parser.add_argument("--coco-instances", default="/path/to/sage_repro_bundle/object_annotation_bundle/coco/instances_train2017.json")
    parser.add_argument(
        "--drop-labels",
        nargs="+",
        default=["likely_unanswerable", "borderline"],
    )
    parser.add_argument(
        "--output-json",
        default="/path/to/sage_repro_bundle/train_raw_filtered_drop_key_object_occluded_and_borderline.json",
    )
    parser.add_argument(
        "--dropped-json",
        default="/path/to/sage_repro_bundle/train_raw_dropped_key_object_occluded_and_borderline.json",
    )
    parser.add_argument(
        "--classified-json",
        default="/path/to/sage_repro_bundle/train_raw_mask_occlusion_classified.json",
    )
    parser.add_argument(
        "--report-json",
        default="/path/to/sage_repro_bundle/train_raw_filtered_drop_key_object_occluded_and_borderline.report.json",
    )
    parser.add_argument("--progress-every", type=int, default=2000)
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_coco(path: Path):
    data = load_json(path)
    cat_name_by_id = {int(c["id"]): str(c["name"]).lower() for c in data["categories"]}
    anns_by_image = defaultdict(list)
    image_sizes = {}
    for image in data["images"]:
        image_sizes[int(image["id"])] = (int(image["width"]), int(image["height"]))
    for ann in data["annotations"]:
        anns_by_image[int(ann["image_id"])].append(
            {
                "category_name": cat_name_by_id[int(ann["category_id"])],
                "segmentation": ann["segmentation"],
            }
        )
    all_cats = sorted(set(cat_name_by_id.values()), key=len, reverse=True)
    return all_cats, anns_by_image, image_sizes


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


def mention_categories(question: str, all_cats: list[str]) -> list[str]:
    q = question.lower()
    hits = []
    for cat in all_cats:
        pattern = r"\b" + re.escape(cat) + r"(?:s|es)?\b"
        if re.search(pattern, q):
            hits.append(cat)
    return sorted(set(hits))


def main() -> None:
    args = parse_args()
    train = load_json(Path(args.train_json))
    mapping = load_json(Path(args.mapping_json))
    mapping_rows = mapping["results"] if isinstance(mapping, dict) and "results" in mapping else mapping
    cues_by_qid = {
        int(row["question_id"]): {str(x).lower() for x in row.get("visual_cues", []) if str(x).strip()}
        for row in mapping_rows
    }
    all_cats, anns_by_image, image_sizes = load_coco(Path(args.coco_instances))
    drop_labels = set(args.drop_labels)

    @lru_cache(maxsize=8192)
    def ann_masks_for_image(image_id: int):
        width, height = image_sizes[image_id]
        grouped = defaultdict(list)
        for ann in anns_by_image.get(image_id, []):
            grouped[ann["category_name"]].append(seg_to_mask(ann["segmentation"], width, height))
        return grouped, width, height

    classified = []
    filtered = []
    dropped = []
    label_counter = Counter()
    reason_counter = Counter()

    for idx, item in enumerate(train, start=1):
        qid = int(item["question_id"])
        iid = int(item["image_id"])
        mentions = mention_categories(item["question"], all_cats)
        cues = cues_by_qid.get(qid, set())
        grouped_masks, width, height = ann_masks_for_image(iid)

        mask = np.zeros((height, width), dtype=bool)
        matched_instance_count = 0
        for cue in cues:
            for ann_mask in grouped_masks.get(cue, []):
                mask |= ann_mask
                matched_instance_count += 1
        mask_ratio = float(mask.mean()) if mask.size else 0.0

        mention_stats = []
        max_cov = 0.0
        max_iou = 0.0
        mask_pixels = int(mask.sum())
        for cat in mentions:
            best = None
            for seg in grouped_masks.get(cat, []):
                seg_area = int(seg.sum())
                if seg_area <= 0:
                    continue
                overlap = int(np.logical_and(mask, seg).sum())
                union = int(np.logical_or(mask, seg).sum())
                cov = float(overlap / max(seg_area, 1))
                share = float(overlap / max(mask_pixels, 1))
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
        if any(x in item["question"].lower() for x in ("how many", "number of", "many")):
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

        result = {
            "question_id": qid,
            "image_id": iid,
            "question": item["question"],
            "answer": item["answer"],
            "answer_type": item.get("answer_type"),
            "visual_cues": sorted(cues),
            "mentions": mentions,
            "mention_mask_stats": mention_stats,
            "matched_instance_count": matched_instance_count,
            "mask_ratio": mask_ratio,
            "label": label,
            "reasons": reasons,
        }
        classified.append(result)
        label_counter[label] += 1
        for reason in reasons:
            reason_counter[reason] += 1
        if label in drop_labels:
            dropped.append(item)
        else:
            filtered.append(item)

        if args.progress_every > 0 and idx % args.progress_every == 0:
            print(
                f"processed={idx} "
                f"unanswerable={label_counter['likely_unanswerable']} "
                f"borderline={label_counter['borderline']} "
                f"answerable={label_counter['likely_answerable']}",
                flush=True,
            )

    Path(args.output_json).write_text(json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.dropped_json).write_text(json.dumps(dropped, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.classified_json).write_text(json.dumps(classified, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "train_json": args.train_json,
        "mapping_json": args.mapping_json,
        "coco_instances": args.coco_instances,
        "drop_labels": sorted(drop_labels),
        "evaluated": len(classified),
        "label_counts": dict(label_counter),
        "reason_counts": dict(reason_counter),
        "output_json": args.output_json,
        "dropped_json": args.dropped_json,
        "classified_json": args.classified_json,
        "dropped_count": len(dropped),
        "filtered_count": len(filtered),
        "drop_rate": (len(dropped) / len(train)) if train else 0.0,
    }
    Path(args.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
