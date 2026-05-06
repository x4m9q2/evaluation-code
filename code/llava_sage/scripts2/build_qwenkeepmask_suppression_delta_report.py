#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager
from matplotlib.table import Table


ROOT = Path("/path/to/sage_repro_bundle")
SUMMARY_DIR = ROOT / "analysis" / "qwenkeepmask_epoch_metrics_summary"
COMPARE_DIR = SUMMARY_DIR / "xverify_compare_r0_r1"

R0_JSON = ROOT / "infer_result_train_filtered_masked_qwenratio_gate_suppress_assembled_pretrain_meanreg_r0" / "assembled_llava_v15_from_pretrain_meanreg_20260319_052641" / "train_raw_filtered_masked_qwenratio_oldbase_sam3_with_shortcut_answer.json"
R1_JSON = ROOT / "infer_result_train_filtered_masked_qwenratio_gate_suppress_assembled_pretrain_meanreg_r1p0" / "assembled_llava_v15_from_pretrain_meanreg_20260319_052641" / "train_raw_filtered_masked_qwenratio_oldbase_sam3_with_shortcut_answer.json"


def configure_chinese_font() -> str | None:
    candidates = [
        "/root/.local/share/fonts/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ]
    for path in candidates:
        p = Path(path)
        if p.exists():
            font_manager.fontManager.addfont(str(p))
            name = font_manager.FontProperties(fname=str(p)).get_name()
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["axes.unicode_minus"] = False
    return None


def find_latest_eval_json(base_dir: Path, tag: str) -> Path:
    matches = sorted((base_dir / f"xverify_{tag}").glob("Eval_Judge_*.json"), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No Eval_Judge_*.json under {base_dir / f'xverify_{tag}'}")
    return matches[-1]


def load_judgments(eval_json: Path) -> list[bool]:
    with eval_json.open() as f:
        data = json.load(f)
    out = []
    for row in data["results"]:
        valid = str(row.get("judge_valid", "")).lower() == "true"
        correct = str(row.get("xVerify-0.5B-I_judgment_result", "")).lower() == "correct"
        out.append(valid and correct)
    return out


def build_delta_rows() -> list[list[str]]:
    rows_r0 = json.load(open(R0_JSON))
    rows_r1 = json.load(open(R1_JSON))
    if len(rows_r0) != len(rows_r1):
        raise ValueError("r0 and r1 json lengths differ")

    r0_acc = load_judgments(find_latest_eval_json(COMPARE_DIR / "r0", "accuracy"))
    r1_acc = load_judgments(find_latest_eval_json(COMPARE_DIR / "r1p0", "accuracy"))
    if not (len(rows_r0) == len(r0_acc) == len(r1_acc)):
        raise ValueError("row/judgment lengths differ")

    cats = ["overall", "yes/no", "number", "other"]
    stats = {
        cat: {"total": 0, "more_wrong": 0, "less_wrong": 0}
        for cat in cats
    }

    for row, a0, a1 in zip(rows_r0, r0_acc, r1_acc):
        cat = row.get("answer_type", "other")
        if cat not in stats:
            cat = "other"
        stats["overall"]["total"] += 1
        stats[cat]["total"] += 1
        if a0 and not a1:
            stats["overall"]["more_wrong"] += 1
            stats[cat]["more_wrong"] += 1
        elif (not a0) and a1:
            stats["overall"]["less_wrong"] += 1
            stats[cat]["less_wrong"] += 1

    out = []
    for cat in cats:
        total = stats[cat]["total"]
        more_wrong = stats[cat]["more_wrong"]
        less_wrong = stats[cat]["less_wrong"]
        out.append(
            [
                cat,
                str(total),
                str(more_wrong),
                f"{(more_wrong / total * 100) if total else 0:.2f}",
                str(less_wrong),
                f"{(less_wrong / total * 100) if total else 0:.2f}",
                f"{((less_wrong - more_wrong) / total * 100) if total else 0:.2f}",
            ]
        )
    return out


def render_table(
    ax,
    headers: list[str],
    rows: list[list[str]],
    title: str,
    col_widths: list[float],
    header_bg: str = "#D9EAF7",
    highlight_rows: set[int] | None = None,
) -> None:
    ax.set_axis_off()
    table = Table(ax, bbox=[0, 0, 1, 1])
    row_h = 1.0 / (len(rows) + 2)

    for i, header in enumerate(headers):
        cell = table.add_cell(0, i, col_widths[i], row_h, text=header, loc="center", facecolor=header_bg)
        cell.get_text().set_fontsize(12)
        cell.get_text().set_weight("bold")

    for r, row in enumerate(rows, start=1):
        for c, value in enumerate(row):
            if c == 0:
                bg = "#EFE7DA"
            elif r % 2 == 0:
                bg = "#F8F8F8"
            else:
                bg = "#FFFFFF"
            cell = table.add_cell(r, c, col_widths[c], row_h, text=value, loc="center", facecolor=bg)
            cell.get_text().set_fontsize(11)
            if highlight_rows and r in highlight_rows:
                cell.get_text().set_color("#C62828")
                if c in {0, 1}:
                    cell.get_text().set_weight("bold")

    ax.add_table(table)
    ax.set_title(title, fontsize=16, weight="bold", pad=12)


def load_first_table_rows() -> list[list[str]]:
    p = SUMMARY_DIR / "qwenkeepmask_epoch_metrics_table.json"
    with p.open() as f:
        return json.load(f)["rows"]


def build_latex_delta(rows: list[list[str]]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{6pt}",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Category & Total & MoreWrong & MoreWrong\\% & LessWrong & LessWrong\\% & NetGain\\% \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{On the 75,196 train\\_raw\\_filtered\\_masked subset, question counts that became wrong (MoreWrong) or became correct (LessWrong) when gate patch suppression changed from 0\\% to 100\\%. Percentages are normalized by each answer-type total.}",
            "\\label{tab:suppression-delta}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    configure_chinese_font()
    first_rows = load_first_table_rows()
    delta_rows = build_delta_rows()

    (SUMMARY_DIR / "qwenkeepmask_suppression_delta_table.tex").write_text(build_latex_delta(delta_rows))
    (SUMMARY_DIR / "qwenkeepmask_suppression_delta_table.json").write_text(
        json.dumps(
            {
                "columns": ["category", "total", "more_wrong", "more_wrong_pct", "less_wrong", "less_wrong_pct", "net_gain_pct"],
                "rows": delta_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )

    pdf_path = SUMMARY_DIR / "qwenkeepmask_tables_combined.pdf"
    png_path = SUMMARY_DIR / "qwenkeepmask_suppression_delta_table.png"

    with PdfPages(pdf_path) as pdf:
        page_size = (22, 8.5)

        fig1, ax1 = plt.subplots(figsize=page_size)
        render_table(
            ax1,
            headers=["Model", "Epoch", "Overall\nAcc", "Overall\nSR", "YN\nAcc", "YN\nSR", "Num\nAcc", "Num\nSR", "Other\nAcc", "Other\nSR"],
            rows=first_rows,
            title="各训练变体在不同 Epoch 与题型上的准确率和捷径率",
            col_widths=[0.17, 0.08] + [0.075] * 8,
            highlight_rows={3, len(first_rows)},
        )
        fig1.text(
            0.015,
            0.015,
            "结论：当前加入 mask loss 后，Acc 仍比不加低 0.05%，但 SR 下降了 0.24%。"
            "从题目分类看，mask loss 对 number 类问题损害更大，主要差异集中在这里。"
            "目前的参数敏感性实验也打算按上表同样的方式整理。",
            fontsize=12,
            color="#222222",
        )
        pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=page_size)
        render_table(
            ax2,
            headers=["题型", "总数", "多错", "多错\n占比(%)", "少错", "少错\n占比(%)", "净变化\n占比(%)"],
            rows=delta_rows,
            title="压制 100% 相对 0% 在各 answer type 上的多错与少错统计",
            col_widths=[0.22, 0.11, 0.12, 0.13, 0.12, 0.13, 0.13],
            header_bg="#DDEFD8",
        )
        fig2.text(
            0.015,
            0.015,
            "结论：在重置掩码前，压制实验会使捷径率上升、准确率下降；"
            "在重新生成掩码并完成数据过滤后，这一现象发生了逆转。",
            fontsize=12,
            color="#222222",
        )
        pdf.savefig(fig2, bbox_inches="tight")
        fig2.savefig(png_path, dpi=220, bbox_inches="tight")
        plt.close(fig2)

    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
