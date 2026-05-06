#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path("/path/to/sage_repro_bundle")
R0_PATH = ROOT / "infer_result_test_raw_gate_suppress_ckpt16470_r0/checkpoint-16470/test_raw_with_shortcut_answer.json"
R1_PATH = ROOT / "infer_result_test_raw_gate_suppress_ckpt16470_r1p0/checkpoint-16470/test_raw_with_shortcut_answer.json"
RULE_PATH = ROOT / "merged_output_rule_mapping.json"
MASK_PATH = ROOT / "patch_mask_analysis_output_mask_coco_seg_direct_llava_pad336_patch14.npz"
IMAGE_DIR = Path("/root/train2014")


SELECTED_QIDS = [
    41700002,
    537825029,
    184490008,
    521947002,
    231705002,
    23695001,
    101195001,
    542000001,
    132002000,
    277127001,
    350077041,
    205769008,
    329638016,
    114728002,
    208723000,
    134597008,
    104310002,
    400204003,
    365175000,
    505895004,
]


MANUAL_NOTES = {
    41700002: "关键证据是右侧较小的红车和方位关系。被压制区域覆盖到街景主体后，模型更容易忽略右侧局部红车，退化成默认否定。",
    537825029: "问题依赖左侧咖啡杯托上的细小勺子。压制图主要影响桌面/甜点邻域，小物体边缘线索被一起削弱，导致漏检。",
    184490008: "需要同时分辨颜色、物体类别和左右位置。压制后沙发上多个遥控器/手柄的相对位置更难绑定，模型转向捷径式肯定回答。",
    521947002: "真正要看的是手、键盘和猫头的局部空间关系。压制了猫附近 patch 后，左侧手是否触到键盘的细节被破坏。",
    231705002: "这是人物属性组合题，关键是绿色上衣和蓝色裤子。压制人物相关区域后，颜色和服饰绑定变差，模型由对变错。",
    23695001: "题目核心是路牌文字，不是车。当前 visual cue 选到 car 明显偏题，压制车附近区域没有去掉捷径，反而干扰了整幅街景理解。",
    101195001: "要靠手柄外形和连线细节区分主机品牌。压制人物/手持区域后，控制器的细粒度形状信息受损。",
    542000001: "这里压制对象和答案相关，飞机主体被压制后，模型还能看出是飞机，但丢失更细的机型级别区分，只剩泛化回答 airplane。",
    132002000: "问题问蔬菜拼盘里树干由什么组成，真正关键是左侧食物造型。压制 bowl 这类错位 cue 会扰乱整盘食材布局，导致语义漂移。",
    277127001: "要看马具扣和最近栅栏柱的相对方向。压制马头/马具区域后，空间参照物之间的相对位置更难判断，直接翻成 left。",
    350077041: "题目只问右侧卡车颜色。压制图明显压到卡车主体，颜色证据直接被削弱，因此从 red 退化到更常见的 blue。",
    205769008: "这是细长局部颜色条纹问题，证据面积很小。压制人物/滑雪板邻域后，蓝色条纹可见度下降，模型更偏向默认白色。",
    329638016: "计数对象就是鸟。压制与 bird 对齐的 patch 后，相当于直接删除计数目标的一部分，因此从 2 掉到 1 很合理。",
    114728002: "题目要数左翼下方发动机数量，证据与 airplane 主体高度重合。压制机翼/机身邻域会直接影响发动机可见性。",
    208723000: "需要判断滑雪者脚下到底是一块还是两块板。压制人物附近 patch 后，板体边界更难分开，模型转成 two。",
    134597008: "这题看门边花盆数量，压制对象却是 dog/bench，属于场景级误伤。门廊局部被连带削弱后，两个花盆容易并成一个。",
    104310002: "计数依赖多个床的完整轮廓。压制 bed cue 后，远处床体被部分抹弱，最容易从 3 数成 2。",
    400204003: "要看水槽下右侧橱柜门数量。压制到厕所/卫浴相关 patch 后，柜门竖线和阴影边界变弱，模型直接漏数成 0。",
    365175000: "题目问领带上的数字数量，证据极细且靠近手。压制人物区域后，小数字纹样几乎不可读，计数退化明显。",
    505895004: "需要只数左半边带红色配料的披萨。压制 pizza 主体后，红色 topping 的分布变不清楚，模型由 1 退成 0。",
}


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_records():
    d0 = {x["question_id"]: x for x in load_json(R0_PATH)}
    d1 = {x["question_id"]: x for x in load_json(R1_PATH)}
    rules = {x["question_id"]: x for x in load_json(RULE_PATH)["results"]}
    npz = np.load(MASK_PATH, allow_pickle=True)
    qid_to_idx = {int(q): i for i, q in enumerate(npz["question_ids"].tolist())}
    coverage_ratio = npz["coverage_ratio"]
    rows = []
    for qid in SELECTED_QIDS:
        a = d0[qid]
        b = d1[qid]
        r = rules.get(qid, {})
        rows.append(
            {
                "question_id": qid,
                "image_id": a["image_id"],
                "answer_type": a["answer_type"],
                "question": a["question"],
                "gt": a["answer"],
                "shortcut": a["shortcut_answer"],
                "r0_pred": a["model_pred"],
                "r1_pred": b["model_pred"],
                "became_shortcut": b["model_pred"] == a["shortcut_answer"],
                "visual_cues": r.get("visual_cues", []),
                "trigger": (r.get("matched_rule") or {}).get("trigger", ""),
                "mask_idx": qid_to_idx[qid],
                "coverage_ratio": coverage_ratio[qid_to_idx[qid]],
                "manual_reason": MANUAL_NOTES.get(qid, ""),
            }
        )
    return rows


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
    # Match LLaVA's pad-to-square preprocessing before mapping 24x24 visual patches.
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


def write_case(output_dir: Path, idx: int, row: dict):
    image_path = IMAGE_DIR / f"COCO_train2014_{int(row['image_id']):012d}.jpg"
    image = Image.open(image_path)
    orig_out = output_dir / f"{idx:02d}_{row['question_id']}_orig.jpg"
    mask_out = output_dir / f"{idx:02d}_{row['question_id']}_suppressed_vis.jpg"
    expand2square(image.convert("RGB")).save(orig_out, quality=95)
    overlay_mask(image, row["coverage_ratio"]).save(mask_out, quality=95)
    meta = {
        k: v
        for k, v in row.items()
        if k not in {"coverage_ratio", "mask_idx"}
    }
    meta["original_image"] = orig_out.name
    meta["suppressed_vis"] = mask_out.name
    with open(output_dir / f"{idx:02d}_{row['question_id']}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def write_summary(output_dir: Path, rows: list[dict]):
    csv_path = output_dir / "cases.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "question_id",
                "image_id",
                "answer_type",
                "question",
                "gt",
                "shortcut",
                "r0_pred",
                "r1_pred",
                "became_shortcut",
                "visual_cues",
                "trigger",
                "manual_reason",
                "original_image",
                "suppressed_vis",
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
                    "became_shortcut": row["became_shortcut"],
                    "visual_cues": "|".join(row["visual_cues"]),
                    "trigger": row["trigger"],
                    "manual_reason": row["manual_reason"],
                    "original_image": f"{idx:02d}_{row['question_id']}_orig.jpg",
                    "suppressed_vis": f"{idx:02d}_{row['question_id']}_suppressed_vis.jpg",
                }
            )

    md_path = output_dir / "README.md"
    lines = [
        "# Gate Patch Suppression Degradation Casebook",
        "",
        "20 个样本均来自 `r=0` 正确、`r=1.0` 错误的退化集合。",
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
                f"- trigger: {row['trigger']}",
                f"- 可能退化原因: {row['manual_reason']}",
                f"- files: `{idx:02d}_{row['question_id']}_orig.jpg`, `{idx:02d}_{row['question_id']}_suppressed_vis.jpg`, `{idx:02d}_{row['question_id']}.json`",
                "",
            ]
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "analysis/degraded_gate_suppress_casebook_20"),
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_records()
    for idx, row in enumerate(rows, start=1):
        write_case(output_dir, idx, row)
    write_summary(output_dir, rows)
    print(output_dir)


if __name__ == "__main__":
    main()
