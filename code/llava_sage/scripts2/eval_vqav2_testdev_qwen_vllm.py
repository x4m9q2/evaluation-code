#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

REPO_ROOT = Path("/path/to/sage_repro_bundle")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llava.eval.m4c_evaluator import EvalAIAnswerProcessor


DEFAULT_DATA_PATH = Path("/path/to/sage_repro_bundle/test_data/llava_vqav2_mscoco_test-dev2015.jsonl")
DEFAULT_IMAGE_FOLDER = Path("/path/to/sage_repro_bundle/playground/data/eval/vqav2/test2015")
DEFAULT_OUTPUT_ROOT = Path("/path/to/sage_repro_bundle/infer_result")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use vLLM to run Qwen3.5-VL style inference on VQAv2 test-dev2015 "
            "and export an EvalAI submission file."
        )
    )
    parser.add_argument("model_path", type=Path, help="模型路径")
    parser.add_argument("batch_size", type=int, help="每次送入 vLLM.chat 的样本数")
    parser.add_argument("gpu", help="使用哪些 GPU，例如 0,1,2")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--image-folder", type=Path, default=DEFAULT_IMAGE_FOLDER)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="启用 thinking 模式；默认关闭，避免输出冗长推理。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有的 submission 或临时结果",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅处理前 N 条样本，便于 smoke test",
    )
    parser.add_argument("--num-chunks", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--chunk-idx", type=int, default=0, help=argparse.SUPPRESS)
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


def batched(items: List[dict], batch_size: int) -> Iterable[List[dict]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def split_list(lst: List[dict], n: int) -> List[List[dict]]:
    chunk_size = (len(lst) + n - 1) // n
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst: List[dict], n: int, k: int) -> List[dict]:
    chunks = split_list(lst, n)
    if k < 0 or k >= len(chunks):
        return []
    return chunks[k]


def write_model_info_template(out_dir: Path, model_path: Path) -> None:
    info_path = out_dir / "model_info.txt"
    if info_path.exists():
        return

    lines = [
        f"模型路径：{model_path}",
        "底座模型：",
        "训练方法：",
        "训练数据：",
        "训练参数：",
        "训练结果摘要：",
        "",
        "请根据模型 README、训练日志、配置或人工确认结果填写。",
        "不要在这里记录推理数据集、推理命令、GPU 分配等运行期信息。",
    ]
    info_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_completed_count(tmp_path: Path) -> int:
    if not tmp_path.exists():
        return 0

    count = 0
    with tmp_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid temp JSONL at {tmp_path}:{line_no}") from exc
            count += 1
    return count


def convert_tmp_to_submission(tmp_path: Path, out_path: Path) -> None:
    rows = []
    with tmp_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid temp JSONL at {tmp_path}:{line_no}") from exc

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
        f.write("\n")


def load_tmp_predictions(tmp_path: Path) -> Dict[int, str]:
    pred_map: Dict[int, str] = {}
    with tmp_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid temp JSONL at {tmp_path}:{line_no}") from exc
            pred_map[int(row["question_id"])] = row["answer"]
    return pred_map


def build_messages(rows: List[dict], image_folder: Path) -> List[list[dict]]:
    messages: List[list[dict]] = []
    for row in rows:
        image_path = image_folder / row["image"]
        if not image_path.exists():
            raise FileNotFoundError(f"image not found: {image_path}")
        messages.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": row["text"]},
                        {"type": "image_url", "image_url": {"url": f"file://{image_path}"}},
                    ],
                }
            ]
        )
    return messages


def get_paths(
    args: argparse.Namespace,
    model_tag: str,
) -> tuple[Path, Path, Path]:
    out_dir = args.output_root / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    submission_file = out_dir / f"{args.data_path.stem}.submission.json"
    if args.num_chunks > 1:
        tmp_submission_file = out_dir / (
            f"{args.data_path.stem}.submission.chunk{args.chunk_idx}of{args.num_chunks}.tmp.jsonl"
        )
    else:
        tmp_submission_file = out_dir / f"{args.data_path.stem}.submission.tmp.jsonl"
    return out_dir, submission_file, tmp_submission_file


def run_worker(args: argparse.Namespace, rows: List[dict], model_tag: str) -> None:
    gpu_list = [x.strip() for x in args.gpu.split(",") if x.strip()]
    if len(gpu_list) != 1:
        raise ValueError("Worker mode requires exactly one GPU id")

    out_dir, submission_file, tmp_submission_file = get_paths(args, model_tag)

    if args.overwrite and tmp_submission_file.exists():
        tmp_submission_file.unlink()
    if args.num_chunks == 1 and args.overwrite and submission_file.exists():
        submission_file.unlink()

    completed = load_completed_count(tmp_submission_file)
    if completed > len(rows):
        raise RuntimeError(
            f"Temp output has {completed} rows, but source only has {len(rows)} rows: {tmp_submission_file}"
        )

    write_model_info_template(out_dir, args.model_path)

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list[0]
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(args.model_path),
        tensor_parallel_size=1,
        trust_remote_code=True,
        allowed_local_media_path=str(args.image_folder),
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.batch_size,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    start_time = time.time()
    answer_processor = EvalAIAnswerProcessor()
    remaining_rows = rows[completed:]
    worker_name = f"{model_tag}:chunk{args.chunk_idx}/{args.num_chunks - 1}"

    if completed:
        print(f"[resume] {worker_name}: continue from {completed}/{len(rows)}", flush=True)
    else:
        print(
            f"[run] {worker_name}: total={len(rows)}, batch_size={args.batch_size}, gpu={gpu_list[0]}",
            flush=True,
        )

    with tmp_submission_file.open("a", encoding="utf-8") as fout:
        processed = completed
        for batch_rows in batched(remaining_rows, args.batch_size):
            batch_start = time.time()
            messages = build_messages(batch_rows, args.image_folder)
            outputs = llm.chat(
                messages,
                sampling_params=sampling_params,
                use_tqdm=False,
                chat_template_content_format="openai",
                chat_template_kwargs={"enable_thinking": args.enable_thinking},
            )

            if len(outputs) != len(batch_rows):
                raise RuntimeError(
                    f"Output size mismatch: got {len(outputs)} outputs for {len(batch_rows)} inputs"
                )

            for row, output in zip(batch_rows, outputs):
                text = ""
                if output.outputs:
                    text = output.outputs[0].text.strip()
                fout.write(
                    json.dumps(
                        {
                            "question_id": int(row["question_id"]),
                            "answer": answer_processor(text),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            fout.flush()

            processed += len(batch_rows)
            elapsed = time.time() - start_time
            batch_elapsed = time.time() - batch_start
            rate = processed / elapsed if elapsed > 0 else 0.0
            remaining = len(rows) - processed
            eta_sec = remaining / rate if rate > 0 else -1
            eta_str = "unknown" if eta_sec < 0 else f"{eta_sec / 3600:.2f}h"
            print(
                f"[progress] {worker_name}: {processed}/{len(rows)} "
                f"({processed / len(rows):.2%}), "
                f"batch_time={batch_elapsed:.1f}s, rate={rate:.2f} q/s, eta={eta_str}",
                flush=True,
            )

    if args.num_chunks == 1:
        convert_tmp_to_submission(tmp_submission_file, submission_file)
        tmp_submission_file.unlink()
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[done] {model_tag}: {submission_file} ({end_time})", flush=True)


def merge_chunk_outputs(
    args: argparse.Namespace,
    model_tag: str,
    src_rows: List[dict],
    num_chunks: int,
) -> Path:
    out_dir, submission_file, _ = get_paths(args, model_tag)
    pred_map: Dict[int, str] = {}
    chunk_files: List[Path] = []
    for chunk_idx in range(num_chunks):
        chunk_file = out_dir / (
            f"{args.data_path.stem}.submission.chunk{chunk_idx}of{num_chunks}.tmp.jsonl"
        )
        if not chunk_file.exists():
            raise FileNotFoundError(f"Missing chunk output: {chunk_file}")
        chunk_files.append(chunk_file)
        for question_id, answer in load_tmp_predictions(chunk_file).items():
            if question_id in pred_map:
                raise RuntimeError(f"Duplicate question_id found while merging: {question_id}")
            pred_map[question_id] = answer

    merged_rows = []
    missing = []
    for row in src_rows:
        question_id = int(row["question_id"])
        answer = pred_map.get(question_id)
        if answer is None:
            missing.append(question_id)
            continue
        merged_rows.append({"question_id": question_id, "answer": answer})

    if missing:
        raise RuntimeError(
            f"Missing predictions for {len(missing)} questions. First few: {missing[:10]}"
        )

    with submission_file.open("w", encoding="utf-8") as f:
        json.dump(merged_rows, f, ensure_ascii=False)
        f.write("\n")

    for chunk_file in chunk_files:
        chunk_file.unlink()

    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[done] {model_tag}: {submission_file} ({end_time})", flush=True)
    return submission_file


def run_data_parallel(args: argparse.Namespace, rows: List[dict], model_tag: str) -> None:
    gpu_list = [x.strip() for x in args.gpu.split(",") if x.strip()]
    out_dir, submission_file, _ = get_paths(args, model_tag)
    if args.overwrite and submission_file.exists():
        submission_file.unlink()
    elif submission_file.exists():
        print(f"[skip] {model_tag}: {submission_file} already exists", flush=True)
        return

    write_model_info_template(out_dir, args.model_path)

    procs = []
    cmds = []
    num_chunks = len(gpu_list)
    for chunk_idx, gpu_id in enumerate(gpu_list):
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            str(args.model_path),
            str(args.batch_size),
            gpu_id,
            "--data-path",
            str(args.data_path),
            "--image-folder",
            str(args.image_folder),
            "--output-root",
            str(args.output_root),
            "--temperature",
            str(args.temperature),
            "--max-tokens",
            str(args.max_tokens),
            "--max-model-len",
            str(args.max_model_len),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--num-chunks",
            str(num_chunks),
            "--chunk-idx",
            str(chunk_idx),
        ]
        if args.enable_thinking:
            cmd.append("--enable-thinking")
        if args.overwrite:
            cmd.append("--overwrite")
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])

        print(f"[launch] {model_tag}: chunk={chunk_idx}/{num_chunks - 1}, gpu={gpu_id}", flush=True)
        procs.append(subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=os.environ.copy()))
        cmds.append(cmd)

    for idx, proc in enumerate(procs):
        ret = proc.wait()
        if ret != 0:
            raise subprocess.CalledProcessError(ret, cmds[idx])

    merge_chunk_outputs(args, model_tag, rows, num_chunks)


def main() -> None:
    args = parse_args()
    ensure_exists(args.model_path, "model path")
    ensure_exists(args.data_path, "data file")
    ensure_exists(args.image_folder, "image folder")
    if not (args.model_path / "config.json").exists():
        raise FileNotFoundError(f"config.json not found under model path: {args.model_path}")

    gpu_list = [x.strip() for x in args.gpu.split(",") if x.strip()]
    if not gpu_list:
        raise ValueError("At least one GPU must be provided")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    rows = load_jsonl(args.data_path)
    if args.limit is not None:
        rows = rows[:args.limit]
    total = len(rows)
    if total == 0:
        raise ValueError(f"No rows found in {args.data_path}")

    model_tag = args.model_path.name
    if args.num_chunks > 1:
        if len(gpu_list) != 1:
            raise ValueError("Worker mode expects exactly one GPU id")
        chunk_rows = get_chunk(rows, args.num_chunks, args.chunk_idx)
        if not chunk_rows:
            raise ValueError(
                f"Empty chunk: chunk_idx={args.chunk_idx}, num_chunks={args.num_chunks}, total={len(rows)}"
            )
        run_worker(args, chunk_rows, model_tag)
        return

    if len(gpu_list) > 1:
        run_data_parallel(args, rows, model_tag)
        return

    run_worker(args, rows, model_tag)


if __name__ == "__main__":
    main()
