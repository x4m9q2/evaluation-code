#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path("/path/to/sage_repro_bundle")
IMAGE_DIR = Path("/root/train2014")
MASK_PATH = ROOT / "patch_mask_analysis_output_mask_coco_seg_direct_llava_pad336_patch14.npz"
CLASSIFIED_PATH = ROOT / "test_data/test_raw_with_shortcut_answer.classified_same_rule.json"
RULE_PATH = ROOT / "merged_output_rule_mapping.json"
R0_FILTERED_PATH = ROOT / "analysis/xverify_filtered_gate_suppress_ckpt16470/test_raw_with_shortcut_answer.r0.filtered.json"
R1_FILTERED_PATH = ROOT / "analysis/xverify_filtered_gate_suppress_ckpt16470/test_raw_with_shortcut_answer.r1p0.filtered.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--r0-xverify-path",
        default=str(sorted((ROOT / "analysis/xverify_casebook_ckpt16470_filtered/r0_accuracy").glob("Eval_Judge_*.json"))[-1]),
    )
    parser.add_argument(
        "--r1-xverify-path",
        default=str(sorted((ROOT / "analysis/xverify_casebook_ckpt16470_filtered/r1p0_accuracy").glob("Eval_Judge_*.json"))[-1]),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "analysis/degraded_gate_suppress_ckpt16470_filtered_casebook_20"),
    )
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def expand2square(pil_img: Image.Image, background_color=(122, 116, 104)) -> Image.Image:
    width, height = pil_img.size
    if width == height:
        return pil_img
    if width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    result = Image.new(pil_img.mode, (height, height), background_color)
    result.paste(pil_img, ((height - width) // 2, 0))
    return result


def overlay_mask(image: Image.Image, coverage: np.ndarray) -> Image.Image:
    image = expand2square(image.convert("RGB")).resize((336, 336))
    base = np.asarray(image).astype(np.float32)
    overlay = base.copy()
    cell = 14
    for r in range(24):
        for c in range(24):
            ratio = float(coverage[r, c])
            if ratio <= 0:
                continue
            y0, y1 = r * cell, (r + 1) * cell
            x0, x1 = c * cell, (c + 1) * cell
            overlay[y0:y1, x0:x1] *= max(0.0, 1.0 - 0.85 * ratio)
            overlay[y0:y1, x0:x1, 0] += 140.0 * ratio
    out = np.clip(overlay, 0, 255).astype(np.uint8)
    canvas = Image.fromarray(out)
    draw = ImageDraw.Draw(canvas)
    for i in range(25):
        pos = i * cell
        draw.line((0, pos, 336, pos), fill=(255, 255, 255), width=1)
        draw.line((pos, 0, pos, 336), fill=(255, 255, 255), width=1)
    return canvas


def binary_mask_image(coverage: np.ndarray) -> Image.Image:
    mask = (coverage > 0).astype(np.uint8) * 255
    mask = np.kron(mask, np.ones((14, 14), dtype=np.uint8))
    return Image.fromarray(mask, mode="L")


def load_candidates(r0_xverify_path: Path, r1_xverify_path: Path):
    r0_rows = {int(x["question_id"]): x for x in load_json(r0_xverify_path)["results"]}
    r1_rows = {int(x["question_id"]): x for x in load_json(r1_xverify_path)["results"]}
    r0_meta = {int(x["question_id"]): x for x in load_json(R0_FILTERED_PATH)}
    r1_meta = {int(x["question_id"]): x for x in load_json(R1_FILTERED_PATH)}
    cls = {int(x["question_id"]): x for x in load_json(CLASSIFIED_PATH)}
    rule_results = load_json(RULE_PATH).get("results", [])
    rule = {int(x["question_id"]): x for x in rule_results}

    npz = np.load(MASK_PATH, allow_pickle=True)
    qid_to_idx = {int(q): i for i, q in enumerate(npz["question_ids"].tolist())}
    coverage_ratio = npz["coverage_ratio"]

    candidates = []
    for qid, row in r0_meta.items():
        r0 = r0_rows.get(qid)
        r1 = r1_rows.get(qid)
        if r0 is None or r1 is None:
            continue
        r0_ok = str(r0.get("judge_valid", "")).lower() == "true" and str(r0.get("xVerify-0.5B-I_judgment_result", "")).lower() == "correct"
        r1_ok = str(r1.get("judge_valid", "")).lower() == "true" and str(r1.get("xVerify-0.5B-I_judgment_result", "")).lower() == "correct"
        if not r0_ok or r1_ok:
            continue
        c = cls[qid]
        mention_stats = c.get("mention_mask_stats", [])
        max_iou = max((float(x.get("iou", 0.0)) for x in mention_stats), default=0.0)
        max_cov = max((float(x.get("seg_coverage", 0.0)) for x in mention_stats), default=0.0)
        candidates.append(
            {
                "question_id": qid,
                "image_id": int(row["image_id"]),
                "answer_type": row["answer_type"],
                "question": row["question"],
                "gt": row["answer"],
                "shortcut": row.get("shortcut_answer"),
                "r0_pred": row.get("model_pred"),
                "r1_pred": r1_meta[qid].get("model_pred"),
                "visual_cues": c.get("visual_cues", []),
                "mentions": c.get("mentions", []),
                "mention_mask_stats": mention_stats,
                "filter_label": c.get("label"),
                "filter_reasons": c.get("reasons", []),
                "mask_ratio": float(c.get("mask_ratio", 0.0)),
                "rule_trigger": (rule.get(qid, {}).get("matched_rule") or {}).get("trigger", ""),
                "rule_answer": rule.get(qid, {}).get("answer"),
                "max_iou": max_iou,
                "max_cov": max_cov,
                "coverage_ratio": coverage_ratio[qid_to_idx[qid]],
            }
        )
    return candidates


def select_cases(candidates, limit):
    groups = {"number": [], "other": [], "yes/no": []}
    for row in candidates:
        groups.setdefault(row["answer_type"], []).append(row)
    for rows in groups.values():
        rows.sort(key=lambda x: (-x["max_iou"], -x["max_cov"], -x["mask_ratio"], x["question_id"]))

    selected = []
    seen = set()
    cycle = ["number", "other", "yes/no"]
    while len(selected) < min(limit, len(candidates)):
        progressed = False
        for key in cycle:
            rows = groups.get(key, [])
            while rows and rows[0]["question_id"] in seen:
                rows.pop(0)
            if not rows:
                continue
            row = rows.pop(0)
            selected.append(row)
            seen.add(row["question_id"])
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
    if len(selected) < limit:
        rest = sorted(
            [x for x in candidates if x["question_id"] not in seen],
            key=lambda x: (-x["max_iou"], -x["max_cov"], -x["mask_ratio"], x["question_id"]),
        )
        selected.extend(rest[: limit - len(selected)])
    return selected


def infer_reason(row: dict) -> str:
    mentions = row.get("mentions", [])
    cues = row.get("visual_cues", [])
    answer_type = row.get("answer_type", "")
    r0_pred = row.get("r0_pred")
    r1_pred = row.get("r1_pred")
    if mentions and row.get("max_iou", 0.0) >= 0.01:
        target = mentions[0]
        return f"patch mask 与题目关键对象 `{target}` 明显重合，压制后直接削弱了作答证据，因此模型从 `{r0_pred}` 退化到 `{r1_pred}`。"
    if answer_type == "number":
        focus = mentions[0] if mentions else (cues[0] if cues else "待计数主体")
        return f"这是计数题，mask 主要覆盖 `{focus}` 或其邻域，压制后容易漏数/并数，所以从 `{r0_pred}` 变成 `{r1_pred}`。"
    if "attribute_or_relation_question" in row.get("filter_reasons", []):
        focus = mentions[0] if mentions else (cues[0] if cues else "局部目标")
        return f"这是属性/关系判断题，压制削弱了 `{focus}` 附近的颜色、方位或局部外观线索，导致判断从 `{r0_pred}` 退化到 `{r1_pred}`。"
    if not mentions:
        cue_text = ", ".join(cues) if cues else "视觉 cue"
        return f"题目没有显式 COCO 对象词，压制更像是在削弱场景上下文与 `{cue_text}` 附近线索，而不是单一物体，因此模型更容易出错。"
    focus = mentions[0]
    return f"压制区域与 `{focus}` 周围证据部分重合，导致支持答案的局部细节不够稳定，因此从 `{r0_pred}` 变成 `{r1_pred}`。"


def write_case(output_dir: Path, idx: int, row: dict):
    image_path = IMAGE_DIR / f"COCO_train2014_{int(row['image_id']):012d}.jpg"
    image = Image.open(image_path).convert("RGB")
    orig_out = output_dir / f"{idx:02d}_{row['question_id']}_orig.jpg"
    overlay_out = output_dir / f"{idx:02d}_{row['question_id']}_mask_overlay.jpg"
    binary_out = output_dir / f"{idx:02d}_{row['question_id']}_mask_binary.png"
    image.save(orig_out, quality=95)
    overlay_mask(image, row["coverage_ratio"]).save(overlay_out, quality=95)
    binary_mask_image(row["coverage_ratio"]).save(binary_out)

    meta = {
        "question_id": row["question_id"],
        "image_id": row["image_id"],
        "answer_type": row["answer_type"],
        "question": row["question"],
        "gt": row["gt"],
        "shortcut": row["shortcut"],
        "r0_pred": row["r0_pred"],
        "r1_pred": row["r1_pred"],
        "visual_cues": row["visual_cues"],
        "mentions": row["mentions"],
        "mention_mask_stats": row["mention_mask_stats"],
        "mask_ratio": row["mask_ratio"],
        "rule_trigger": row["rule_trigger"],
        "rule_answer": row["rule_answer"],
        "possible_error_reason": infer_reason(row),
        "original_image": orig_out.name,
        "mask_overlay_image": overlay_out.name,
        "mask_binary_image": binary_out.name,
    }
    with open(output_dir / f"{idx:02d}_{row['question_id']}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def write_summary(output_dir: Path, rows):
    csv_path = output_dir / "cases.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id", "question_id", "image_id", "answer_type", "question", "gt",
                "shortcut", "r0_pred", "r1_pred", "visual_cues", "mentions",
                "mask_ratio", "rule_trigger", "rule_answer", "possible_error_reason",
                "original_image", "mask_overlay_image", "mask_binary_image",
            ],
        )
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "case_id": idx,
                    "question_id": row["question_id"],
                    "image_id": row["image_id"],
                    "answer_type": row["answer_type"],
                    "question": row["question"],
                    "gt": row["gt"],
                    "shortcut": row["shortcut"],
                    "r0_pred": row["r0_pred"],
                    "r1_pred": row["r1_pred"],
                    "visual_cues": "|".join(row["visual_cues"]),
                    "mentions": "|".join(row["mentions"]),
                    "mask_ratio": row["mask_ratio"],
                    "rule_trigger": row["rule_trigger"],
                    "rule_answer": row["rule_answer"],
                    "possible_error_reason": infer_reason(row),
                    "original_image": f"{idx:02d}_{row['question_id']}_orig.jpg",
                    "mask_overlay_image": f"{idx:02d}_{row['question_id']}_mask_overlay.jpg",
                    "mask_binary_image": f"{idx:02d}_{row['question_id']}_mask_binary.png",
                }
            )

    md_path = output_dir / "README.md"
    lines = [
        "# Filtered Degradation Casebook For Checkpoint-16470",
        "",
        "样本来自过滤后测试集中的 `r0` 正确、`r1.0` 错误集合。",
        "这里的掩码图是按 24x24 patch coverage 叠加到 pad-336 视图上的可视化，不是原始实例分割 mask。",
        "",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"## Case {idx:02d} | qid={row['question_id']}",
                f"- image_id: {row['image_id']}",
                f"- answer_type: {row['answer_type']}",
                f"- question: {row['question']}",
                f"- gt / shortcut / r0 / r1: {row['gt']} / {row['shortcut']} / {row['r0_pred']} / {row['r1_pred']}",
                f"- visual_cues: {', '.join(row['visual_cues']) if row['visual_cues'] else '(none)'}",
                f"- mentions: {', '.join(row['mentions']) if row['mentions'] else '(none)'}",
                f"- rule trigger -> shortcut answer: {row['rule_trigger']} -> {row['rule_answer']}",
                f"- 可能错因: {infer_reason(row)}",
                f"- files: `{idx:02d}_{row['question_id']}_orig.jpg`, `{idx:02d}_{row['question_id']}_mask_overlay.jpg`, `{idx:02d}_{row['question_id']}_mask_binary.png`, `{idx:02d}_{row['question_id']}.json`",
                "",
            ]
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates(Path(args.r0_xverify_path), Path(args.r1_xverify_path))
    selected = select_cases(candidates, args.limit)
    for idx, row in enumerate(selected, start=1):
        write_case(output_dir, idx, row)
    write_summary(output_dir, selected)
    print(output_dir)


if __name__ == "__main__":
    main()
