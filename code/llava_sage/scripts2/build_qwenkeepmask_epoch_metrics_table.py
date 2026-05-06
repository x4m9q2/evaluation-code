#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager
from matplotlib.table import Table


ROOT = Path("/path/to/sage_repro_bundle")
OUT_DIR = ROOT / "analysis" / "qwenkeepmask_epoch_metrics_summary"


MODELS = [
    {
        "label": "maskloss=base",
        "key": "full",
        "epochs": {
            "epoch1": ROOT / "infer_result_test_raw_gate_suppress_r0_epoch1" / "checkpoint-1716" / "test_raw_with_shortcut_answer.xverify_metrics.json",
            "epoch2": ROOT / "infer_result" / "checkpoint-3432" / "test_raw_with_shortcut_answer.xverify_metrics.json",
            "epoch3": ROOT / "infer_result" / "finetune_stage2_qwenratio_oldbase_sam3_qwenkeepmask_full_bs32_20260421_0310" / "test_raw_with_shortcut_answer.xverify_metrics.json",
        },
        "ckpt": str(ROOT / "checkpoints" / "finetune_stage2_qwenratio_oldbase_sam3_qwenkeepmask_full_bs32_20260421_0310"),
    },
    {
        "label": "maskloss=base/3",
        "key": "div3",
        "epochs": {
            "epoch1": ROOT / "infer_result" / "checkpoint-1716" / "test_raw_with_shortcut_answer.xverify_metrics.json",
            "epoch2": None,
            "epoch3": None,
        },
        "ckpt": str(ROOT / "checkpoints" / "finetune_stage2_qwenratio_oldbase_sam3_qwenkeepmask_maskloss_div3_full_bs32_20260422_022647"),
    },
    {
        "label": "maskloss=base*2",
        "key": "x2",
        "epochs": {
            "epoch1": ROOT / "infer_result_maskloss_x2_fullsched_bs32_20260422_1336_epoch1" / "checkpoint-1716" / "test_raw_with_shortcut_answer.xverify_metrics.json",
            "epoch2": ROOT / "infer_result_maskloss_x2_fullsched_bs32_20260422_1336_epoch2" / "checkpoint-3432" / "test_raw_with_shortcut_answer.xverify_metrics.json",
            "epoch3": ROOT / "infer_result_maskloss_x2_fullsched_bs32_20260422_1336_epoch3" / "checkpoint-5148" / "test_raw_with_shortcut_answer.xverify_metrics.json",
        },
        "ckpt": str(ROOT / "checkpoints" / "finetune_stage2_qwenratio_oldbase_sam3_qwenkeepmask_maskloss_x2_fullsched_bs32_20260422_1336"),
    },
    {
        "label": "maskloss=0",
        "key": "nomask",
        "epochs": {
            "epoch1": ROOT / "infer_result_test_raw_nomaskloss_epoch1" / "checkpoint-1716" / "test_raw_with_shortcut_answer.xverify_metrics.json",
            "epoch2": ROOT / "infer_result_test_raw_nomaskloss_epoch2" / "checkpoint-3432" / "test_raw_with_shortcut_answer.xverify_metrics.json",
            "epoch3": ROOT / "infer_result_test_raw_nomaskloss_epoch3" / "checkpoint-5148" / "test_raw_with_shortcut_answer.xverify_metrics.json",
        },
        "ckpt": str(ROOT / "checkpoints" / "finetune_stage2_qwenratio_oldbase_sam3_qwenkeepmask_nomaskloss_full_bs32_20260421_091640"),
    },
]

ANSWER_TYPES = ["overall", "yes/no", "number", "other"]


def pct(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value * 100:.2f}"


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


def load_metric(path: Path | None) -> dict[str, dict[str, float | None]]:
    if path is None or not path.exists():
        return {
            answer_type: {"acc": None, "shortcut": None}
            for answer_type in ANSWER_TYPES
        }
    with path.open() as f:
        data = json.load(f)
    result = {
        "overall": {
            "acc": data["accuracy"]["stat_info"]["Accuracy"],
            "shortcut": data["shortcut_rate"]["stat_info"]["Accuracy"],
        }
    }
    for answer_type in ANSWER_TYPES[1:]:
        result[answer_type] = {
            "acc": data["accuracy"]["by_answer_type"][answer_type]["accuracy"],
            "shortcut": data["shortcut_rate"]["by_answer_type"][answer_type]["accuracy"],
        }
    return result


def build_rows() -> list[list[str]]:
    rows: list[list[str]] = []
    for model in MODELS:
        for epoch in ["epoch1", "epoch2", "epoch3"]:
            metrics = load_metric(model["epochs"][epoch])
            row = [model["label"], epoch]
            for answer_type in ANSWER_TYPES:
                row.append(pct(metrics[answer_type]["acc"]))
                row.append(pct(metrics[answer_type]["shortcut"]))
            rows.append(row)
    return rows


def build_latex(rows: list[list[str]]) -> str:
    headers = [
        "Model",
        "Epoch",
        "Overall Acc",
        "Overall SR",
        "YN Acc",
        "YN SR",
        "Num Acc",
        "Num SR",
        "Other Acc",
        "Other SR",
    ]
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{llrrrrrrrr}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    last_model = None
    for row in rows:
        if last_model is not None and row[0] != last_model:
            lines.append("\\midrule")
        lines.append(" & ".join(row) + " \\\\")
        last_model = row[0]
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Test-set xVerify accuracy (Acc, \\%) and shortcut rate (SR, \\%) by answer type for four qwenkeepmask training variants across epochs. Missing values indicate no located evaluation artifact.}",
            "\\label{tab:qwenkeepmask-epoch-metrics}",
            "\\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def render_table_figure(rows: list[list[str]]):
    headers = [
        "Model",
        "Epoch",
        "Overall\nAcc",
        "Overall\nSR",
        "YN\nAcc",
        "YN\nSR",
        "Num\nAcc",
        "Num\nSR",
        "Other\nAcc",
        "Other\nSR",
    ]
    fig, ax = plt.subplots(figsize=(22, 8.5))
    ax.set_axis_off()
    table = Table(ax, bbox=[0, 0, 1, 1])

    col_widths = [0.17, 0.08] + [0.075] * 8
    row_h = 1.0 / (len(rows) + 2)

    for i, header in enumerate(headers):
        cell = table.add_cell(0, i, col_widths[i], row_h, text=header, loc="center", facecolor="#D9EAF7")
        cell.get_text().set_fontsize(12)
        cell.get_text().set_weight("bold")

    highlight_rows = {3, len(rows)}
    for r, row in enumerate(rows, start=1):
        model_bg = "#F5F5F5" if r % 2 else "#FFFFFF"
        for c, value in enumerate(row):
            face = model_bg
            if c == 0:
                face = "#EFE7DA"
            elif c == 1:
                face = "#F7F1E8"
            cell = table.add_cell(r, c, col_widths[c], row_h, text=value, loc="center", facecolor=face)
            cell.get_text().set_fontsize(11)
            if r in highlight_rows:
                cell.get_text().set_color("#C62828")
                if c in {0, 1}:
                    cell.get_text().set_weight("bold")

    ax.add_table(table)
    ax.set_title(
        "各训练变体在不同 Epoch 与题型上的准确率和捷径率",
        fontsize=16,
        weight="bold",
        pad=12,
    )
    return fig


def render_png(rows: list[list[str]], out_path: Path) -> None:
    fig = render_table_figure(rows)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_pdf(rows: list[list[str]], out_path: Path) -> None:
    conclusion = (
        "结论：当前加入 mask loss 后，Acc 仍比不加低 0.05%，但 SR 下降了 0.24%。"
        "从题目分类看，mask loss 对 number 类问题损害更大，主要差异集中在这里。"
        "目前的参数敏感性实验也打算按上表同样的方式整理。"
    )
    with PdfPages(out_path) as pdf:
        fig = render_table_figure(rows)
        fig.text(0.015, 0.015, conclusion, fontsize=12, color="#222222")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def build_metadata(rows: list[list[str]]) -> dict:
    return {
        "models": [{"label": model["label"], "checkpoint_root": model["ckpt"]} for model in MODELS],
        "columns": [
            "model",
            "epoch",
            "overall_acc_pct",
            "overall_shortcut_pct",
            "yesno_acc_pct",
            "yesno_shortcut_pct",
            "number_acc_pct",
            "number_shortcut_pct",
            "other_acc_pct",
            "other_shortcut_pct",
        ],
        "rows": rows,
        "note": "div3 epoch2/3 are missing because no test-set xverify metric files were located.",
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_chinese_font()
    rows = build_rows()
    latex = build_latex(rows)
    (OUT_DIR / "qwenkeepmask_epoch_metrics_table.tex").write_text(latex)
    (OUT_DIR / "qwenkeepmask_epoch_metrics_table.json").write_text(
        json.dumps(build_metadata(rows), ensure_ascii=False, indent=2) + "\n"
    )
    render_png(rows, OUT_DIR / "qwenkeepmask_epoch_metrics_table.png")
    render_pdf(rows, OUT_DIR / "qwenkeepmask_epoch_metrics_table.pdf")
    print(OUT_DIR / "qwenkeepmask_epoch_metrics_table.tex")
    print(OUT_DIR / "qwenkeepmask_epoch_metrics_table.png")
    print(OUT_DIR / "qwenkeepmask_epoch_metrics_table.pdf")


if __name__ == "__main__":
    main()
