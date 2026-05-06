import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


BUNDLE_DIR = Path("/path/to/sage_repro_bundle/mask_review_bundle")
EVAL100_REPORT = Path("/path/to/sage_repro_bundle/train_raw_llava_mask_eval100_report.json")
RULE100_REPORT = Path("/path/to/sage_repro_bundle/rule_mask_sample100_summary.json")
TRAIN_RAW = Path("/path/to/sage_repro_bundle/train_raw.json")
OLD_MASK_DIR = Path("/path/to/sage_repro_bundle/output_mask")
NEW_MASK_DIR = Path("/path/to/sage_repro_bundle/output_mask_coco_seg")
IMAGE_DIR = Path("data/images/coco/train2014")
MAX_OCCLUDED_EXAMPLES = 20
COCO_INSTANCES = Path("/path/to/sage_repro_bundle/object_annotation_bundle/coco/instances_train2017.json")
TRAIN_KEY_OBJECT_SCAN_LIMIT = 3000


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def safe_ascii(text: str) -> str:
    return str(text).encode("ascii", "replace").decode("ascii")


def copy_with_manifest(samples: list[dict], out_dir: Path, title: str) -> None:
    reset_dir(out_dir)
    manifest = []
    for item in samples:
        src = Path(item["panel_path"])
        if not src.exists():
            continue
        dst = out_dir / src.name
        shutil.copy2(src, dst)
        row = {
            "question_id": item["question_id"],
            "image_id": item["image_id"],
            "question": item["question"],
            "answer": item["answer"],
            "answer_type": item["answer_type"],
            "label": item["label"],
            "reasons": item.get("reasoning", []),
            "mentions": item.get("mentions", []),
            "mask_ratio": item.get("mask_ratio"),
            "panel_file": dst.name,
        }
        manifest.append(row)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [f"# {title}", "", f"- Count: `{len(manifest)}`", ""]
    for item in manifest:
        lines.append(
            f"- `{item['panel_file']}` qid=`{item['question_id']}` ans=`{item['answer']}` "
            f"type=`{item['answer_type']}` mentions=`{item['mentions']}` "
            f"mask_ratio=`{item['mask_ratio']}` q={item['question']}"
        )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def old_mask_reason(sample: dict) -> str:
    best = sample["best_match"]
    cov = float(best["segmentation_mask_coverage"])
    share = float(best["mask_inside_bbox_share"])
    mask_ratio = float(sample["mask_area_ratio"])
    parts = []
    if cov < 0.2:
        parts.append("miss_object")
    if share < 0.03:
        parts.append("not_focused")
    if mask_ratio > 0.6:
        parts.append("mask_too_large")
    if best["iou"] < 0.05:
        parts.append("tiny_overlap")
    if not parts:
        parts.append("threshold_edge")
    return "+".join(parts)


def quality_label(sample: dict) -> str:
    if sample["status"] != "ok":
        return sample["status"]
    cov = float(sample["best_match"]["segmentation_mask_coverage"])
    share = float(sample["best_match"]["mask_inside_bbox_share"])
    mask_ratio = float(sample["mask_area_ratio"])
    if cov >= 0.5 and share >= 0.15 and mask_ratio <= 0.6:
        return "strong"
    if cov >= 0.2 and share >= 0.03 and mask_ratio <= 0.75:
        return "partial"
    return "weak"


def build_old_mask_panels(samples: list[dict], out_dir: Path, train_by_qid: dict[int, dict]) -> None:
    reset_dir(out_dir)
    manifest = []
    for sample in samples:
        qid = int(sample["question_id"])
        iid = int(sample["image_id"])
        train_item = train_by_qid.get(qid, {})
        orig = IMAGE_DIR / f"COCO_train2014_{iid:012d}.jpg"
        old = OLD_MASK_DIR / f"{qid}_{iid}.jpg"
        if not orig.exists() or not old.exists():
            continue
        orig_im = Image.open(orig).convert("RGB")
        old_im = Image.open(old).convert("RGB")
        if old_im.size != orig_im.size:
            old_im = old_im.resize(orig_im.size)
        top = 94
        canvas = Image.new("RGB", (orig_im.width * 2, orig_im.height + top), "white")
        canvas.paste(orig_im, (0, top))
        canvas.paste(old_im, (orig_im.width, top))
        draw = ImageDraw.Draw(canvas)
        reason = old_mask_reason(sample)
        question_text = train_item.get("question") or "unavailable in local train sets"
        lines = [
            f"qid={qid} iid={iid}",
            f"question={safe_ascii(question_text)[:110]}",
            f"visual_cues={safe_ascii(str(sample.get('visual_cues', [])))[:110]}",
            f"ans={safe_ascii(train_item.get('answer', sample.get('answer', '')))}",
            f"reason={reason}",
        ]
        for i, line in enumerate(lines):
            draw.text((10, 8 + i * 20), line, fill="black")
        draw.text((10, top - 18), "original", fill="black")
        draw.text((orig_im.width + 10, top - 18), "old output_mask", fill="black")
        dst = out_dir / f"{qid}_{iid}.jpg"
        canvas.save(dst, quality=90)
        manifest.append(
            {
                "question_id": qid,
                "image_id": iid,
                "question": question_text,
                "answer": train_item.get("answer"),
                "answer_type": train_item.get("answer_type"),
                "visual_cues": sample.get("visual_cues", []),
                "trigger": (sample.get("matched_rule") or {}).get("trigger"),
                "mask_ratio": sample.get("mask_area_ratio"),
                "segmentation_mask_coverage": sample["best_match"]["segmentation_mask_coverage"],
                "mask_inside_bbox_share": sample["best_match"]["mask_inside_bbox_share"],
                "iou": sample["best_match"]["iou"],
                "reason": reason,
                "panel_file": dst.name,
            }
        )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Original output_mask Problem Cases", "", f"- Count: `{len(manifest)}`", ""]
    for item in manifest:
        lines.append(
            f"- `{item['panel_file']}` qid=`{item['question_id']}` reason=`{item['reason']}` "
            f"cue=`{item['visual_cues']}` trigger=`{item['trigger']}` "
            f"seg_cov=`{item['segmentation_mask_coverage']:.4f}` "
            f"mask_share=`{item['mask_inside_bbox_share']:.4f}` iou=`{item['iou']:.4f}` "
            f"q={item['question']}"
        )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def occlusion_score(sample: dict) -> float:
    best = sample["best_match"]
    return (
        float(best["segmentation_mask_coverage"]) * 0.55
        + float(best["mask_inside_bbox_share"]) * 0.30
        + float(best["iou"]) * 0.15
    )


def select_occluded_key_object_examples(samples: list[dict], limit: int = MAX_OCCLUDED_EXAMPLES) -> list[dict]:
    candidates = [
        x
        for x in samples
        if x["status"] == "ok"
        and float(x["best_match"]["segmentation_mask_coverage"]) >= 0.9
        and float(x["best_match"]["mask_inside_bbox_share"]) >= 0.15
    ]
    candidates.sort(key=occlusion_score, reverse=True)
    picked = []
    cue_counter: dict[str, int] = {}
    for sample in candidates:
        cues = sample.get("visual_cues", [])
        primary_cue = str(cues[0]) if cues else "unknown"
        if cue_counter.get(primary_cue, 0) >= 3:
            continue
        picked.append(sample)
        cue_counter[primary_cue] = cue_counter.get(primary_cue, 0) + 1
        if len(picked) >= limit:
            break
    return picked


def build_old_mask_occlusion_panels(samples: list[dict], out_dir: Path, train_by_qid: dict[int, dict]) -> None:
    reset_dir(out_dir)
    manifest = []
    for sample in samples:
        qid = int(sample["question_id"])
        iid = int(sample["image_id"])
        train_item = train_by_qid.get(qid, {})
        orig = IMAGE_DIR / f"COCO_train2014_{iid:012d}.jpg"
        old = OLD_MASK_DIR / f"{qid}_{iid}.jpg"
        if not orig.exists() or not old.exists():
            continue
        orig_im = Image.open(orig).convert("RGB")
        old_im = Image.open(old).convert("RGB")
        if old_im.size != orig_im.size:
            old_im = old_im.resize(orig_im.size)
        top = 112
        canvas = Image.new("RGB", (orig_im.width * 2, orig_im.height + top), "white")
        canvas.paste(orig_im, (0, top))
        canvas.paste(old_im, (orig_im.width, top))
        draw = ImageDraw.Draw(canvas)
        best = sample["best_match"]
        question_text = train_item.get("question") or "unavailable in local train sets"
        lines = [
            f"qid={qid} iid={iid}",
            f"question={safe_ascii(question_text)[:110]}",
            f"visual_cues={safe_ascii(str(sample.get('visual_cues', [])))[:110]}",
            f"ans={safe_ascii(train_item.get('answer', sample.get('answer', '')))}",
            (
                "effect=key_object_occluded "
                f"seg_cov={float(best['segmentation_mask_coverage']):.3f} "
                f"mask_share={float(best['mask_inside_bbox_share']):.3f} "
                f"iou={float(best['iou']):.3f}"
            ),
        ]
        for i, line in enumerate(lines):
            draw.text((10, 8 + i * 20), line, fill="black")
        draw.text((10, top - 18), "original", fill="black")
        draw.text((orig_im.width + 10, top - 18), "old output_mask", fill="black")
        dst = out_dir / f"{qid}_{iid}.jpg"
        canvas.save(dst, quality=90)
        manifest.append(
            {
                "question_id": qid,
                "image_id": iid,
                "question": question_text,
                "answer": train_item.get("answer"),
                "answer_type": train_item.get("answer_type"),
                "visual_cues": sample.get("visual_cues", []),
                "trigger": (sample.get("matched_rule") or {}).get("trigger"),
                "mask_ratio": sample.get("mask_area_ratio"),
                "segmentation_mask_coverage": best["segmentation_mask_coverage"],
                "mask_inside_bbox_share": best["mask_inside_bbox_share"],
                "iou": best["iou"],
                "effect": "question_key_object_occluded",
                "panel_file": dst.name,
            }
        )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Train Question Key Objects Occluded By Original output_mask", "", f"- Count: `{len(manifest)}`", ""]
    for item in manifest:
        lines.append(
            f"- `{item['panel_file']}` qid=`{item['question_id']}` cue=`{item['visual_cues']}` "
            f"trigger=`{item['trigger']}` seg_cov=`{item['segmentation_mask_coverage']:.4f}` "
            f"mask_share=`{item['mask_inside_bbox_share']:.4f}` iou=`{item['iou']:.4f}` "
            f"q={item['question']}"
        )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_coco_mentions(path: Path) -> tuple[list[str], dict[int, list[dict]]]:
    coco = json.loads(path.read_text(encoding="utf-8"))
    cat_name_by_id = {int(c["id"]): c["name"].lower() for c in coco["categories"]}
    all_cats = sorted(set(cat_name_by_id.values()), key=len, reverse=True)
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
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


def question_object_occlusion_score(sample: dict) -> float:
    return (
        float(sample["segmentation_mask_coverage"]) * 0.55
        + float(sample["iou"]) * 0.30
        + float(sample["mask_inside_bbox_share"]) * 0.15
    )


def select_question_key_object_occluded_examples(
    train: list[dict],
    all_cats: list[str],
    anns_by_image: dict[int, list[dict]],
    mask_dir: Path,
    limit: int = MAX_OCCLUDED_EXAMPLES,
    scan_limit: int = TRAIN_KEY_OBJECT_SCAN_LIMIT,
) -> list[dict]:
    candidates = []
    for item in train[:scan_limit]:
        qid = int(item["question_id"])
        iid = int(item["image_id"])
        orig_path = IMAGE_DIR / f"COCO_train2014_{iid:012d}.jpg"
        masked_path = mask_dir / f"{qid}_{iid}.jpg"
        if not orig_path.exists() or not masked_path.exists():
            continue
        mentions = mention_categories(item["question"], all_cats)
        if not mentions:
            continue
        mask = build_mask_binary(orig_path, masked_path)
        best = None
        for cat in mentions:
            for ann in anns_by_image.get(iid, []):
                if ann["category_name"] != cat:
                    continue
                seg = seg_to_mask(ann["segmentation"], width=mask.shape[1], height=mask.shape[0])
                seg_area = int(seg.sum())
                if seg_area <= 0:
                    continue
                overlap = int(np.logical_and(mask, seg).sum())
                union = int(np.logical_or(mask, seg).sum())
                candidate = {
                    "category": cat,
                    "segmentation_mask_coverage": float(overlap / max(seg_area, 1)),
                    "mask_inside_bbox_share": float(overlap / max(int(mask.sum()), 1)),
                    "iou": float(overlap / max(union, 1)),
                }
                if best is None or question_object_occlusion_score(candidate) > question_object_occlusion_score(best):
                    best = candidate
        if best is None:
            continue
        if best["segmentation_mask_coverage"] < 0.7 or best["iou"] < 0.15:
            continue
        candidates.append(
            {
                "question_id": qid,
                "image_id": iid,
                "question": item["question"],
                "answer": item["answer"],
                "answer_type": item.get("answer_type"),
                "mentions": mentions,
                "matched_question_object": best["category"],
                "mask_ratio": float(mask.mean()),
                "segmentation_mask_coverage": best["segmentation_mask_coverage"],
                "mask_inside_bbox_share": best["mask_inside_bbox_share"],
                "iou": best["iou"],
            }
        )
    candidates.sort(key=question_object_occlusion_score, reverse=True)
    picked = []
    mention_counter: dict[str, int] = {}
    for sample in candidates:
        obj = str(sample["matched_question_object"])
        if mention_counter.get(obj, 0) >= 3:
            continue
        picked.append(sample)
        mention_counter[obj] = mention_counter.get(obj, 0) + 1
        if len(picked) >= limit:
            break
    return picked


def build_train_question_object_occlusion_panels(
    samples: list[dict],
    out_dir: Path,
    mask_dir: Path,
    title: str,
    masked_label: str,
    effect_label: str,
) -> None:
    reset_dir(out_dir)
    manifest = []
    for sample in samples:
        qid = int(sample["question_id"])
        iid = int(sample["image_id"])
        orig = IMAGE_DIR / f"COCO_train2014_{iid:012d}.jpg"
        masked = mask_dir / f"{qid}_{iid}.jpg"
        if not orig.exists() or not masked.exists():
            continue
        orig_im = Image.open(orig).convert("RGB")
        masked_im = Image.open(masked).convert("RGB")
        if masked_im.size != orig_im.size:
            masked_im = masked_im.resize(orig_im.size)
        top = 112
        canvas = Image.new("RGB", (orig_im.width * 2, orig_im.height + top), "white")
        canvas.paste(orig_im, (0, top))
        canvas.paste(masked_im, (orig_im.width, top))
        draw = ImageDraw.Draw(canvas)
        lines = [
            f"qid={qid} iid={iid}",
            f"question={safe_ascii(sample['question'])[:110]}",
            f"question_objects={safe_ascii(str(sample['mentions']))[:110]}",
            f"matched_object={safe_ascii(sample['matched_question_object'])}",
            f"ans={safe_ascii(sample['answer'])}",
            (
                f"effect={effect_label} "
                f"seg_cov={float(sample['segmentation_mask_coverage']):.3f} "
                f"mask_share={float(sample['mask_inside_bbox_share']):.3f} "
                f"iou={float(sample['iou']):.3f}"
            ),
        ]
        for i, line in enumerate(lines):
            draw.text((10, 8 + i * 18), line, fill="black")
        draw.text((10, top - 18), "original", fill="black")
        draw.text((orig_im.width + 10, top - 18), masked_label, fill="black")
        dst = out_dir / f"{qid}_{iid}.jpg"
        canvas.save(dst, quality=90)
        row = dict(sample)
        row["effect"] = effect_label
        row["panel_file"] = dst.name
        manifest.append(row)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [f"# {title}", "", f"- Count: `{len(manifest)}`", ""]
    for item in manifest:
        lines.append(
            f"- `{item['panel_file']}` qid=`{item['question_id']}` matched_object=`{item['matched_question_object']}` "
            f"mentions=`{item['mentions']}` seg_cov=`{item['segmentation_mask_coverage']:.4f}` "
            f"mask_share=`{item['mask_inside_bbox_share']:.4f}` iou=`{item['iou']:.4f}` "
            f"q={item['question']}"
        )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dir(BUNDLE_DIR)
    eval100 = json.loads(EVAL100_REPORT.read_text(encoding="utf-8"))
    rule100 = json.loads(RULE100_REPORT.read_text(encoding="utf-8"))
    train = json.loads(TRAIN_RAW.read_text(encoding="utf-8"))
    all_cats, anns_by_image = load_coco_mentions(COCO_INSTANCES)
    train_by_qid = {int(x["question_id"]): x for x in train}

    unanswerable = [x for x in eval100["samples"] if x["label"] == "likely_unanswerable"]
    answerable = [x for x in eval100["samples"] if x["label"] == "likely_answerable"]
    weak_old = [x for x in rule100["samples"] if quality_label(x) == "weak"]
    occluded_old = select_question_key_object_occluded_examples(train, all_cats, anns_by_image, OLD_MASK_DIR)
    occluded_new = select_question_key_object_occluded_examples(train, all_cats, anns_by_image, NEW_MASK_DIR)

    copy_with_manifest(
        unanswerable,
        BUNDLE_DIR / "1_unanswerable_questions_with_mask",
        "Likely Unanswerable Questions With New Mask",
    )
    copy_with_manifest(
        answerable,
        BUNDLE_DIR / "2_likely_answerable_questions_with_mask",
        "Likely Answerable Questions With New Mask",
    )
    build_old_mask_panels(
        weak_old,
        BUNDLE_DIR / "3_problematic_original_output_mask_cases",
        train_by_qid,
    )
    build_train_question_object_occlusion_panels(
        occluded_old,
        BUNDLE_DIR / "4_key_objects_occluded_by_original_output_mask",
        OLD_MASK_DIR,
        "Train Question Key Objects Occluded By Original output_mask",
        "old output_mask",
        "train_question_key_object_occluded",
    )
    build_train_question_object_occlusion_panels(
        occluded_new,
        BUNDLE_DIR / "5_key_objects_occluded_by_new_coco_mask",
        NEW_MASK_DIR,
        "Train Question Key Objects Occluded By New COCO Mask",
        "new coco mask",
        "train_question_key_object_occluded_by_new_coco_mask",
    )

    summary = {
        "unanswerable_count": len(unanswerable),
        "answerable_count": len(answerable),
        "problematic_old_mask_count": len(weak_old),
        "old_mask_key_object_occluded_examples": len(occluded_old),
        "new_coco_mask_key_object_occluded_examples": len(occluded_new),
        "bundle_dir": str(BUNDLE_DIR),
    }
    (BUNDLE_DIR / "README.md").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
