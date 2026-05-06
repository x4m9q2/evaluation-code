#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


DEFAULT_XVERIFY_ROOT = Path("/path/to/sage_repro_bundle/x_verify")
DEFAULT_XVERIFY_MODEL = Path("/path/to/sage_repro_bundle/x_verify/xVerify-0.5B-I")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run xVerify-based accuracy and shortcut-rate evaluation from a merged inference JSON file."
    )
    parser.add_argument("--input-path", type=Path, required=True, help="Merged JSON with answer, shortcut_answer and model_pred.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults to the input file's parent directory.")
    parser.add_argument("--xverify-root", type=Path, default=DEFAULT_XVERIFY_ROOT)
    parser.add_argument("--xverify-model-path", type=Path, default=DEFAULT_XVERIFY_MODEL)
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value for xVerify.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_rows(path: Path) -> List[dict]:
    with path.open() as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return rows


def build_xverify_input(rows: List[dict], answer_key: str) -> List[dict]:
    output = []
    for row in rows:
        output.append(
            {
                "question": row.get("question", ""),
                "llm_output": row.get("model_pred", ""),
                "correct_answer": row.get(answer_key, ""),
                "answer_type": row.get("answer_type", ""),
            }
        )
    return output


def save_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def run_xverify(
    xverify_root: Path,
    xverify_model_path: Path,
    input_path: Path,
    output_dir: Path,
    gpu: str,
    batch_size: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = f"{xverify_root}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(xverify_root)
    cmd = [
        sys.executable,
        "run_local_xverify.py",
        "--data-path",
        str(input_path),
        "--output-path",
        str(output_dir),
        "--model-path",
        str(xverify_model_path),
        "--batch-size",
        str(batch_size),
    ]
    subprocess.run(cmd, cwd=xverify_root, env=env, check=True)
    result_files = sorted(output_dir.glob("Eval_Judge_*.json"), key=lambda p: p.stat().st_mtime)
    if not result_files:
        raise FileNotFoundError(f"No xVerify result file found under {output_dir}")
    return result_files[-1]


def load_stat_info(path: Path) -> Dict[str, object]:
    with path.open() as f:
        data = json.load(f)
    stat_info = data.get("stat_info")
    if not isinstance(stat_info, dict):
        raise ValueError(f"stat_info not found in {path}")
    return stat_info


def build_answer_type_stats(result_path: Path) -> Dict[str, Dict[str, float]]:
    with result_path.open() as f:
        data = json.load(f)
    rows = data.get("results", [])
    if not isinstance(rows, list):
        raise ValueError(f"results not found in {result_path}")
    totals: Dict[str, int] = {}
    corrects: Dict[str, int] = {}
    for row in rows:
        if str(row.get("judge_valid", "")).lower() != "true":
            continue
        answer_type = str(row.get("answer_type", "")).strip() or "unknown"
        totals[answer_type] = totals.get(answer_type, 0) + 1
        if str(row.get("xVerify-0.5B-I_judgment_result", "")).lower() == "correct":
            corrects[answer_type] = corrects.get(answer_type, 0) + 1
    stats: Dict[str, Dict[str, float]] = {}
    for answer_type, total in totals.items():
        correct = corrects.get(answer_type, 0)
        incorrect = total - correct
        stats[answer_type] = {
            "valid_num": total,
            "correct_num": correct,
            "incorrect_num": incorrect,
            "accuracy": (correct / total) if total else 0.0,
        }
    return stats


def default_output_dir(input_path: Path) -> Path:
    return input_path.parent


def summary_output_path(input_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{input_path.stem}.xverify_metrics.json"


def cleanup_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_path)
    output_dir = args.output_dir or default_output_dir(args.input_path)
    summary_path = summary_output_path(args.input_path, output_dir)
    if summary_path.exists() and not args.overwrite:
        print(summary_path)
        return

    accuracy_input = output_dir / f"{args.input_path.stem}.accuracy_xverify_input.json"
    shortcut_input = output_dir / f"{args.input_path.stem}.shortcut_xverify_input.json"
    accuracy_out_dir = output_dir / "xverify_accuracy"
    shortcut_out_dir = output_dir / "xverify_shortcut"

    save_json(build_xverify_input(rows, "answer"), accuracy_input)
    save_json(build_xverify_input(rows, "shortcut_answer"), shortcut_input)

    accuracy_result = run_xverify(
        xverify_root=args.xverify_root,
        xverify_model_path=args.xverify_model_path,
        input_path=accuracy_input,
        output_dir=accuracy_out_dir,
        gpu=args.gpu,
        batch_size=args.batch_size,
    )
    shortcut_result = run_xverify(
        xverify_root=args.xverify_root,
        xverify_model_path=args.xverify_model_path,
        input_path=shortcut_input,
        output_dir=shortcut_out_dir,
        gpu=args.gpu,
        batch_size=args.batch_size,
    )

    report = {
        "input_path": str(args.input_path),
        "xverify_model_path": str(args.xverify_model_path),
        "accuracy": {
            "stat_info": load_stat_info(accuracy_result),
            "by_answer_type": build_answer_type_stats(accuracy_result),
        },
        "shortcut_rate": {
            "stat_info": load_stat_info(shortcut_result),
            "by_answer_type": build_answer_type_stats(shortcut_result),
        },
    }
    save_json(report, summary_path)

    cleanup_path(accuracy_input)
    cleanup_path(shortcut_input)
    cleanup_path(accuracy_out_dir)
    cleanup_path(shortcut_out_dir)
    print(summary_path)


if __name__ == "__main__":
    main()
