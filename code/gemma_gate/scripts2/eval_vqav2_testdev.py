#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llava.eval.m4c_evaluator import EvalAIAnswerProcessor


DEFAULT_DATA_PATH = REPO_ROOT / "data/eval/llava_vqav2_mscoco_test-dev2015.jsonl"
DEFAULT_IMAGE_FOLDER = REPO_ROOT / "data/playground_data/eval/vqav2/test2015"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/infer_result"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run VQAv2 test-dev inference for a single model and write merged "
            "predictions plus an EvalAI submission file."
        )
    )
    parser.add_argument("model_path", type=Path, help="模型路径")
    parser.add_argument(
        "has_gate",
        choices=["true", "false"],
        help="推理时是否强制启用双输入门控",
    )
    parser.add_argument("batch_size", type=int, help="推理 batch size")
    parser.add_argument("gpu", help="使用哪些 GPU，例如 0,1,2")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--image-folder", type=Path, default=DEFAULT_IMAGE_FOLDER)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果输出已存在则覆盖",
    )
    return parser.parse_args()


def ensure_exists(path: Path, kind: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{kind} not found: {path}")


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
    return rows


def load_predictions(path: Path) -> Dict[int, str]:
    preds: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid prediction JSONL at {path}:{line_no}") from exc
            preds[int(row["question_id"])] = row.get("text", "")
    return preds


def merge_prediction_files(paths: List[Path]) -> Dict[int, str]:
    merged: Dict[int, str] = {}
    for path in paths:
        preds = load_predictions(path)
        overlap = set(merged).intersection(preds)
        if overlap:
            raise RuntimeError(
                f"Duplicate question_id found across chunk outputs in {path}. "
                f"First few: {sorted(overlap)[:10]}"
            )
        merged.update(preds)
    return merged


def write_submission_json(src_rows: List[dict], pred_map: Dict[int, str], out_path: Path) -> None:
    processor = EvalAIAnswerProcessor()
    submission = []
    missing = []
    for item in src_rows:
        qid = int(item["question_id"])
        if qid not in pred_map:
            missing.append(qid)
        answer = pred_map.get(qid, "")
        submission.append(
            {
                "question_id": qid,
                "answer": processor(answer),
            }
        )
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False)
        f.write("\n")
    if missing:
        raise RuntimeError(
            f"Missing predictions for {len(missing)} questions. "
            f"First few question_ids: {missing[:10]}"
        )


def write_model_info(
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    info_path = out_dir / "model_info.txt"
    if info_path.exists():
        return
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"生成时间：{now_str}",
        f"模型路径：{args.model_path}",
        "",
        "请在此处人工补充模型训练信息，只记录模型如何训练得到以及常规训练参数。",
        "不要在该文件中记录推理数据集、推理命令、输出文件名、GPU 分配等运行期信息。",
        "训练背景信息应来自模型目录中的 README、训练日志、配置或人工确认结果，不要由脚本自动猜测。",
    ]
    info_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cleanup_intermediate_files(paths: List[Path]) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def main() -> None:
    args = parse_args()
    ensure_exists(args.model_path, "model path")
    ensure_exists(args.data_path, "data file")
    ensure_exists(args.image_folder, "image folder")
    if not (args.model_path / "config.json").exists():
        raise FileNotFoundError(f"config.json not found under model path: {args.model_path}")

    src_rows = load_jsonl(args.data_path)
    model_tag = args.model_path.name
    out_dir = args.output_root / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    submission_file = out_dir / f"{args.data_path.stem}.submission.json"
    answers_file = out_dir / "predictions.jsonl"

    if submission_file.exists() and not args.overwrite:
        print(f"[skip] {model_tag}: {submission_file} already exists", flush=True)
        return

    gpu_list = [x.strip() for x in args.gpu.split(",") if x.strip()]
    if not gpu_list:
        raise ValueError("At least one GPU must be provided")

    base_cmd = [
        sys.executable,
        "-m",
        "llava.eval.model_vqa_loader",
        "--model-path",
        str(args.model_path),
        "--image-folder",
        str(args.image_folder),
        "--question-file",
        str(args.data_path),
        "--conv-mode",
        args.conv_mode,
        "--temperature",
        str(args.temperature),
        "--num_beams",
        str(args.num_beams),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--force-use-dual-input-gate",
        args.has_gate,
    ]

    pred_files: List[Path] = []
    if len(gpu_list) == 1:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_list[0]
        env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
        env["PYTHONPATH"] = (
            f"{REPO_ROOT}{os.pathsep}{env['PYTHONPATH']}"
            if env.get("PYTHONPATH")
            else str(REPO_ROOT)
        )
        cmd = base_cmd + ["--answers-file", str(answers_file)]
        print(f"[run] {model_tag}", flush=True)
        subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)
        pred_files = [answers_file]
        info_cmd = cmd
    else:
        procs = []
        cmds: List[List[str]] = []
        for chunk_idx, gpu_id in enumerate(gpu_list):
            chunk_answers = out_dir / f"predictions_chunk{chunk_idx}.jsonl"
            pred_files.append(chunk_answers)
            cmd = base_cmd + [
                "--answers-file",
                str(chunk_answers),
                "--num-chunks",
                str(len(gpu_list)),
                "--chunk-idx",
                str(chunk_idx),
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
            env["PYTHONPATH"] = (
                f"{REPO_ROOT}{os.pathsep}{env['PYTHONPATH']}"
                if env.get("PYTHONPATH")
                else str(REPO_ROOT)
            )
            print(f"[run] {model_tag} chunk={chunk_idx} gpu={gpu_id}", flush=True)
            procs.append(subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env))
            cmds.append(cmd)

        for idx, proc in enumerate(procs):
            ret = proc.wait()
            if ret != 0:
                raise subprocess.CalledProcessError(ret, cmds[idx])

        with answers_file.open("w", encoding="utf-8") as fout:
            for pred_file in pred_files:
                with pred_file.open("r", encoding="utf-8") as fin:
                    for line in fin:
                        if line.strip():
                            fout.write(line)
        info_cmd = base_cmd + [
            "--answers-file",
            str(answers_file),
            "--num-chunks",
            str(len(gpu_list)),
        ]

    pred_map = merge_prediction_files(pred_files)
    write_submission_json(src_rows, pred_map, submission_file)
    write_model_info(out_dir, args)

    cleanup_targets = [answers_file]
    cleanup_targets.extend(pred_files)
    cleanup_intermediate_files(cleanup_targets)
    print(f"[done] {model_tag}: {submission_file}", flush=True)


if __name__ == "__main__":
    main()
