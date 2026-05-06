#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


REPO_ROOT = Path("/path/to/sage_repro_bundle")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_MODEL_PATH = Path("models/Qwen2.5-VL-7B-Instruct")
DEFAULT_DATA_PATH = Path("/path/to/sage_repro_bundle/test_data/test_raw_with_shortcut_answer.json")
DEFAULT_IMAGE_FOLDER = Path("data/images/coco/train2014")
DEFAULT_OUTPUT_ROOT = Path("/path/to/sage_repro_bundle/infer_result")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use vLLM to run Qwen2.5-VL style inference on test_raw_with_shortcut_answer.json "
            "and export a merged JSON with model_pred, then optionally run xVerify."
        )
    )
    parser.add_argument("model_path", type=Path, nargs="?", default=DEFAULT_MODEL_PATH, help="模型路径")
    parser.add_argument("batch_size", type=int, help="每张卡每次送入 vLLM.chat 的样本数")
    parser.add_argument("gpu", help="使用哪些 GPU，例如 0,1,2")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--image-folder", type=Path, default=DEFAULT_IMAGE_FOLDER)
    parser.add_argument(
        "--question-image-dir",
        type=Path,
        default=None,
        help="Optional directory containing question-specific images named {question_id}_{image_id}.jpg.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="启用 thinking 模式；默认关闭，避免输出冗长推理。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有的合并结果、临时结果和评测结果。",
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


def load_json_array(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON array in {path}")
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


def image_path_from_row(row: dict, image_folder: Path, question_image_dir: Path | None = None) -> Path:
    if question_image_dir is not None:
        qid = int(row["question_id"])
        image_id = int(row["image_id"])
        qimg = question_image_dir / f"{qid}_{image_id}.jpg"
        if qimg.exists():
            return qimg
    image_id = int(row["image_id"])
    return image_folder / f"COCO_train2014_{image_id:012d}.jpg"


def build_messages(
    rows: List[dict],
    image_folder: Path,
    question_image_dir: Path | None = None,
) -> List[list[dict]]:
    messages: List[list[dict]] = []
    for row in rows:
        image_path = image_path_from_row(row, image_folder, question_image_dir)
        if not image_path.exists():
            raise FileNotFoundError(f"image not found: {image_path}")
        messages.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": row["question"]},
                        {"type": "image_url", "image_url": {"url": f"file://{image_path}"}},
                    ],
                }
            ]
        )
    return messages


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


def get_paths(args: argparse.Namespace, model_tag: str) -> tuple[Path, Path, Path]:
    out_dir = args.output_root / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_file = out_dir / args.data_path.name
    stats_file = out_dir / f"{args.data_path.stem}.generation_stats.json"
    return out_dir, merged_file, stats_file


def chunk_tmp_file(out_dir: Path, data_stem: str, chunk_idx: int, num_chunks: int) -> Path:
    return out_dir / f"{data_stem}.chunk{chunk_idx}of{num_chunks}.tmp.jsonl"


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


def load_tmp_predictions(tmp_path: Path) -> Dict[int, dict]:
    pred_map: Dict[int, dict] = {}
    with tmp_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid temp JSONL at {tmp_path}:{line_no}") from exc
            question_id = int(row["question_id"])
            pred_map[question_id] = {
                "model_pred": row.get("model_pred", ""),
                "model_pred_num_output_tokens": int(row.get("model_pred_num_output_tokens", 0)),
                "model_pred_hit_max_tokens": bool(row.get("model_pred_hit_max_tokens", False)),
            }
    return pred_map


def save_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def run_worker(args: argparse.Namespace, rows: List[dict], model_tag: str) -> None:
    gpu_list = [x.strip() for x in args.gpu.split(",") if x.strip()]
    if len(gpu_list) != 1:
        raise ValueError("Worker mode requires exactly one GPU id")

    out_dir, _, _ = get_paths(args, model_tag)
    tmp_pred_file = chunk_tmp_file(out_dir, args.data_path.stem, args.chunk_idx, args.num_chunks)

    if args.overwrite and tmp_pred_file.exists():
        tmp_pred_file.unlink()

    completed = load_completed_count(tmp_pred_file)
    if completed > len(rows):
        raise RuntimeError(
            f"Temp output has {completed} rows, but source only has {len(rows)} rows: {tmp_pred_file}"
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
    remaining_rows = rows[completed:]
    worker_name = f"{model_tag}:chunk{args.chunk_idx}/{args.num_chunks - 1}"
    hit_max_tokens_total = 0

    if completed:
        existing_pred_map = load_tmp_predictions(tmp_pred_file)
        hit_max_tokens_total = sum(
            1 for item in existing_pred_map.values() if item["model_pred_hit_max_tokens"]
        )
        print(f"[resume] {worker_name}: continue from {completed}/{len(rows)}", flush=True)
    else:
        print(
            f"[run] {worker_name}: total={len(rows)}, batch_size={args.batch_size}, "
            f"max_tokens={args.max_tokens}, max_model_len={args.max_model_len}, gpu={gpu_list[0]}",
            flush=True,
        )

    with tmp_pred_file.open("a", encoding="utf-8") as fout:
        processed = completed
        for batch_rows in batched(remaining_rows, args.batch_size):
            batch_start = time.time()
            messages = build_messages(batch_rows, args.image_folder, args.question_image_dir)
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
                num_output_tokens = 0
                hit_max_tokens = False
                if output.outputs:
                    text = output.outputs[0].text.strip()
                    num_output_tokens = len(output.outputs[0].token_ids)
                    hit_max_tokens = num_output_tokens >= args.max_tokens
                if hit_max_tokens:
                    hit_max_tokens_total += 1
                fout.write(
                    json.dumps(
                        {
                            "question_id": int(row["question_id"]),
                            "model_pred": text,
                            "model_pred_num_output_tokens": num_output_tokens,
                            "model_pred_hit_max_tokens": hit_max_tokens,
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
                f"({processed / len(rows):.2%}), batch_time={batch_elapsed:.1f}s, "
                f"rate={rate:.2f} q/s, hit_max_tokens={hit_max_tokens_total}, eta={eta_str}",
                flush=True,
            )

    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[done-worker] {worker_name}: {tmp_pred_file} ({end_time})", flush=True)


def merge_chunk_outputs(
    args: argparse.Namespace,
    model_tag: str,
    src_rows: List[dict],
    num_chunks: int,
) -> Path:
    out_dir, merged_file, stats_file = get_paths(args, model_tag)
    pred_map: Dict[int, dict] = {}
    chunk_files: List[Path] = []

    for chunk_idx in range(num_chunks):
        chunk_file = chunk_tmp_file(out_dir, args.data_path.stem, chunk_idx, num_chunks)
        if not chunk_file.exists():
            raise FileNotFoundError(f"Missing chunk output: {chunk_file}")
        chunk_files.append(chunk_file)
        chunk_preds = load_tmp_predictions(chunk_file)
        overlap = set(pred_map).intersection(chunk_preds)
        if overlap:
            raise RuntimeError(f"Duplicate question_id found while merging: {sorted(overlap)[:10]}")
        pred_map.update(chunk_preds)

    merged_rows = []
    missing = []
    output_token_total = 0
    max_output_tokens_observed = 0
    hit_max_tokens_count = 0
    for row in src_rows:
        question_id = int(row["question_id"])
        pred = pred_map.get(question_id)
        if pred is None:
            missing.append(question_id)
            continue
        output_token_total += pred["model_pred_num_output_tokens"]
        max_output_tokens_observed = max(max_output_tokens_observed, pred["model_pred_num_output_tokens"])
        if pred["model_pred_hit_max_tokens"]:
            hit_max_tokens_count += 1
        merged_row = dict(row)
        merged_row["model_pred"] = pred["model_pred"]
        merged_row["model_pred_num_output_tokens"] = pred["model_pred_num_output_tokens"]
        merged_row["model_pred_hit_max_tokens"] = pred["model_pred_hit_max_tokens"]
        merged_rows.append(merged_row)

    if missing:
        raise RuntimeError(
            f"Missing predictions for {len(missing)} questions. First few: {missing[:10]}"
        )

    save_json(merged_rows, merged_file)
    save_json(
        {
            "model_path": str(args.model_path),
            "input_path": str(args.data_path),
            "sample_count": len(src_rows),
            "num_gpus": num_chunks,
            "batch_size_per_gpu": args.batch_size,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "temperature": args.temperature,
            "hit_max_tokens_count": hit_max_tokens_count,
            "hit_max_tokens_rate": hit_max_tokens_count / len(src_rows) if src_rows else 0.0,
            "max_output_tokens_observed": max_output_tokens_observed,
            "avg_output_tokens": output_token_total / len(src_rows) if src_rows else 0.0,
        },
        stats_file,
    )

    for chunk_file in chunk_files:
        chunk_file.unlink()

    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[done-merge] {model_tag}: {merged_file} ({end_time})", flush=True)
    print(f"[done-stats] {model_tag}: {stats_file}", flush=True)
    return merged_file


def run_data_parallel(args: argparse.Namespace, rows: List[dict], model_tag: str) -> None:
    gpu_list = [x.strip() for x in args.gpu.split(",") if x.strip()]
    out_dir, merged_file, stats_file = get_paths(args, model_tag)

    if args.overwrite:
        if merged_file.exists():
            merged_file.unlink()
        if stats_file.exists():
            stats_file.unlink()
    elif merged_file.exists():
        print(f"[skip] {model_tag}: {merged_file} already exists", flush=True)
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


def run_single_gpu(args: argparse.Namespace, rows: List[dict], model_tag: str) -> None:
    out_dir, merged_file, stats_file = get_paths(args, model_tag)
    if args.overwrite:
        if merged_file.exists():
            merged_file.unlink()
        if stats_file.exists():
            stats_file.unlink()
    elif merged_file.exists():
        print(f"[skip] {model_tag}: {merged_file} already exists", flush=True)
        return

    run_worker(args, rows, model_tag)
    merge_chunk_outputs(args, model_tag, rows, 1)


def main() -> None:
    args = parse_args()
    ensure_exists(args.model_path, "model path")
    ensure_exists(args.data_path, "data file")
    ensure_exists(args.image_folder, "image folder")
    if args.question_image_dir is not None:
        ensure_exists(args.question_image_dir, "question image dir")
    if not (args.model_path / "config.json").exists():
        raise FileNotFoundError(f"config.json not found under model path: {args.model_path}")

    gpu_list = [x.strip() for x in args.gpu.split(",") if x.strip()]
    if not gpu_list:
        raise ValueError("At least one GPU must be provided")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    rows = load_json_array(args.data_path)
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

    run_single_gpu(args, rows, model_tag)


if __name__ == "__main__":
    main()
