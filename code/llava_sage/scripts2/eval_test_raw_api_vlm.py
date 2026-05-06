#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import threading
import time
from io import BytesIO
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import requests
from PIL import Image


REPO_ROOT = Path("/path/to/sage_repro_bundle")
DEFAULT_DATA_PATH = REPO_ROOT / "test_data" / "test_raw_with_shortcut_answer.json"
DEFAULT_IMAGE_FOLDER = Path("/root/train2014")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "infer_result"
DEFAULT_XVERIFY_ROOT = REPO_ROOT / "x_verify"
DEFAULT_XVERIFY_MODEL = DEFAULT_XVERIFY_ROOT / "xVerify-0.5B-I"
DEFAULT_BASE_URL = "https://yunwu.ai/v1"
DEFAULT_COCO_URL_PREFIX = "https://images.cocodataset.org/train2014"

PROMPT_PREFIX = (
    "Answer the question using the image. "
    "Return only the short final answer with no explanation."
)

_THREAD_LOCAL = threading.local()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run multimodal API inference on test_raw_with_shortcut_answer.json, "
            "write merged JSON with model_pred, then optionally run xVerify."
        )
    )
    parser.add_argument("--model", required=True, help="Remote model name, e.g. qwen3.5-plus")
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key. If omitted, uses API_KEY, OPENAI_API_KEY, or YUNWU_API_KEY from the environment.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--image-folder", type=Path, default=DEFAULT_IMAGE_FOLDER)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Pass enable_thinking=false for API models that expose hidden reasoning tokens.",
    )
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help="Maximum retries per sample before failing.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--image-source",
        choices=["coco_url", "local_data_url"],
        default="coco_url",
        help="Use public COCO URLs or upload local image bytes as a data URL.",
    )
    parser.add_argument(
        "--coco-url-prefix",
        default=DEFAULT_COCO_URL_PREFIX,
        help="Public URL prefix used when --image-source=coco_url.",
    )
    parser.add_argument(
        "--image-max-side",
        type=int,
        default=896,
        help="Resize images so the longer side is at most this value before upload.",
    )
    parser.add_argument(
        "--image-jpeg-quality",
        type=int,
        default=85,
        help="JPEG quality used for uploaded images.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run-xverify", action="store_true")
    parser.add_argument("--xverify-root", type=Path, default=DEFAULT_XVERIFY_ROOT)
    parser.add_argument("--xverify-model-path", type=Path, default=DEFAULT_XVERIFY_MODEL)
    parser.add_argument("--xverify-gpu", default="0")
    parser.add_argument("--xverify-batch-size", type=int, default=32)
    return parser.parse_args()


def ensure_exists(path: Path, kind: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{kind} not found: {path}")


def normalize_base_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


def load_json_array(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return rows


def image_path_from_row(row: dict, image_folder: Path) -> Path:
    image_id = int(row["image_id"])
    return image_folder / f"COCO_train2014_{image_id:012d}.jpg"


def image_url_from_row(row: dict, coco_url_prefix: str) -> str:
    image_id = int(row["image_id"])
    return f"{coco_url_prefix.rstrip('/')}/COCO_train2014_{image_id:012d}.jpg"


def to_data_url(image_path: Path, image_max_side: int, image_jpeg_quality: int) -> str:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        longest_side = max(width, height)
        if image_max_side > 0 and longest_side > image_max_side:
            scale = image_max_side / float(longest_side)
            image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)

        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=image_jpeg_quality, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"


def write_model_info_template(out_dir: Path, args: argparse.Namespace) -> None:
    info_path = out_dir / "model_info.txt"
    if info_path.exists():
        return

    lines = [
        f"模型名称：{args.model}",
        "来源：远程 API 模型",
        "底座模型：",
        "训练方法：",
        "训练数据：",
        "训练参数：",
        "训练结果摘要：",
        "",
        "请根据模型官方说明或人工确认结果填写。",
        "不要在这里记录推理数据集、推理命令、GPU 分配、API 地址等运行期信息。",
    ]
    info_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_tmp_predictions(tmp_path: Path) -> Dict[int, dict]:
    preds: Dict[int, dict] = {}
    if not tmp_path.exists():
        return preds
    with tmp_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid temp JSONL at {tmp_path}:{line_no}") from exc
            preds[int(row["question_id"])] = row
    return preds


def append_jsonl_row(tmp_path: Path, row: dict, lock: threading.Lock) -> None:
    with lock:
        with tmp_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_thread_session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": "LLaVA-eval_test_raw_api_vlm/1.0"})
        _THREAD_LOCAL.session = session
    return session


def extract_text_from_response(data: dict) -> str:
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            texts: List[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(str(item.get("text", "")))
            return "\n".join(x for x in texts if x).strip()
    output = data.get("output")
    if isinstance(output, list):
        texts: List[str] = []
        for item in output:
            content = item.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") in {"output_text", "text"}:
                    texts.append(str(block.get("text", "")))
        return "\n".join(x for x in texts if x).strip()
    raise ValueError(f"Unable to extract text from API response keys={sorted(data.keys())}")


def extract_finish_reason(data: dict) -> str:
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        return str(choices[0].get("finish_reason", "") or "")
    return ""


def should_retry_prediction(pred: str, finish_reason: str) -> bool:
    normalized = pred.strip().lower()
    if not normalized:
        return True
    if normalized in {"no image", "unknown", "image not available", "unable to determine"}:
        return True
    if finish_reason.lower() == "length" and len(normalized.split()) <= 2:
        return True
    return False


def build_payload(args: argparse.Namespace, row: dict, image_ref: str) -> dict:
    payload = {
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{PROMPT_PREFIX}\nQuestion: {row['question']}"},
                    {"type": "image_url", "image_url": {"url": image_ref}},
                ],
            }
        ],
    }
    if args.disable_thinking:
        payload["enable_thinking"] = False
    return payload


def infer_one(
    args: argparse.Namespace,
    row: dict,
) -> Tuple[dict, int]:
    url = f"{normalize_base_url(args.base_url)}/chat/completions"
    if args.image_source == "coco_url":
        image_ref = image_url_from_row(row, args.coco_url_prefix)
    else:
        image_path = image_path_from_row(row, args.image_folder)
        ensure_exists(image_path, "image")
        image_ref = to_data_url(
            image_path,
            image_max_side=args.image_max_side,
            image_jpeg_quality=args.image_jpeg_quality,
        )
    payload = build_payload(args, row, image_ref)
    headers = {
        "Authorization": f"Bearer {args.api_key}",
        "Content-Type": "application/json",
    }
    session = get_thread_session()

    qid = int(row["question_id"])
    print(f"[request_start] qid={qid}", flush=True)
    last_error: Exception | None = None
    for attempt in range(1, args.max_attempts + 1):
        try:
            response = session.post(
                url,
                headers=headers,
                json=payload,
                timeout=(15, args.timeout),
            )
            if response.status_code >= 400:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
            data = response.json()
            pred = extract_text_from_response(data)
            finish_reason = extract_finish_reason(data)
            if should_retry_prediction(pred, finish_reason):
                raise RuntimeError(
                    f"Suspicious prediction with finish_reason={finish_reason!r}, pred={pred!r}"
                )
            usage = data.get("usage", {})
            completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            return (
                {
                    "question_id": qid,
                    "model_pred": pred,
                    "model_pred_num_output_tokens": completion_tokens,
                    "model_pred_hit_max_tokens": completion_tokens >= args.max_tokens if completion_tokens else False,
                },
                attempt,
            )
        except Exception as exc:
            last_error = exc
            print(
                f"[request_retry] qid={qid} attempt={attempt}/{args.max_attempts} error={type(exc).__name__}: {exc}",
                flush=True,
            )
            if attempt == args.max_attempts:
                break
            error_text = str(exc)
            if "HTTP 429" in error_text or "负载已饱和" in error_text or "get_channel_failed" in error_text:
                time.sleep(min(60, 10 * attempt))
            else:
                time.sleep(min(10, 2 ** (attempt - 1)))

    skipped_reason = type(last_error).__name__ if last_error is not None else "unknown_error"
    if last_error is not None:
        error_text = str(last_error).strip().replace("\n", " ")
        skipped_reason = f"{skipped_reason}: {error_text[:200]}"
    print(f"[request_skip] qid={qid} reason={skipped_reason}", flush=True)
    return (
        {
            "question_id": qid,
            "model_pred": "",
            "model_pred_num_output_tokens": 0,
            "model_pred_hit_max_tokens": False,
            "skipped_reason": skipped_reason,
        },
        args.max_attempts,
    )


def run_pending_requests(
    args: argparse.Namespace,
    rows: List[dict],
    pending_rows: List[dict],
    completed: Dict[int, dict],
    tmp_file: Path,
) -> None:
    write_lock = threading.Lock()
    done_count = len(completed)
    total = len(rows)
    pending_iter = iter(pending_rows)
    inflight: Dict[object, Tuple[dict, float]] = {}
    stall_log_interval = max(30, args.timeout // 2)
    last_stall_log_at = 0.0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        while len(inflight) < args.workers:
            row = next(pending_iter, None)
            if row is None:
                break
            future = executor.submit(infer_one, args, row)
            inflight[future] = (row, time.time())

        while inflight:
            done, _ = wait(inflight.keys(), timeout=5, return_when=FIRST_COMPLETED)
            now = time.time()
            if not done:
                if now - last_stall_log_at >= stall_log_interval:
                    oldest = sorted(
                        (
                            (int(meta_row["question_id"]), int(now - started_at))
                            for meta_row, started_at in inflight.values()
                        ),
                        key=lambda item: item[1],
                        reverse=True,
                    )[: min(5, len(inflight))]
                    print(
                        f"[stall] waiting_on={len(inflight)} oldest={oldest}",
                        flush=True,
                    )
                    last_stall_log_at = now
                continue

            for future in done:
                row, _ = inflight.pop(future)
                result, attempts = future.result()
                completed[int(result["question_id"])] = result
                append_jsonl_row(tmp_file, result, write_lock)
                done_count += 1
                print(
                    f"[request_done] qid={result['question_id']} attempts={attempts} "
                    f"pred={result['model_pred']!r}",
                    flush=True,
                )
                if done_count % 50 == 0 or done_count == total:
                    print(
                        f"[progress] {done_count}/{total} complete; "
                        f"last_qid={result['question_id']} attempts={attempts}",
                        flush=True,
                    )

                next_row = next(pending_iter, None)
                if next_row is not None:
                    next_future = executor.submit(infer_one, args, next_row)
                    inflight[next_future] = (next_row, time.time())


def merge_predictions(src_rows: Iterable[dict], pred_map: Dict[int, dict]) -> List[dict]:
    merged: List[dict] = []
    missing: List[int] = []
    for row in src_rows:
        qid = int(row["question_id"])
        pred = pred_map.get(qid)
        if pred is None:
            missing.append(qid)
            continue
        merged.append(
            {
                **row,
                "model_pred": pred.get("model_pred", ""),
                "model_pred_num_output_tokens": int(pred.get("model_pred_num_output_tokens", 0)),
                "model_pred_hit_max_tokens": bool(pred.get("model_pred_hit_max_tokens", False)),
                **({"skipped_reason": pred.get("skipped_reason", "")} if pred.get("skipped_reason") else {}),
            }
        )
    if missing:
        raise RuntimeError(f"Missing predictions for {len(missing)} samples. First few question_ids: {missing[:10]}")
    return merged


def save_generation_stats(merged_rows: List[dict], out_path: Path, started_at: float, ended_at: float) -> None:
    total_output_tokens = sum(int(row.get("model_pred_num_output_tokens", 0) or 0) for row in merged_rows)
    hit_max_tokens = sum(int(bool(row.get("model_pred_hit_max_tokens", False))) for row in merged_rows)
    stats = {
        "num_samples": len(merged_rows),
        "total_output_tokens": total_output_tokens,
        "avg_output_tokens": (total_output_tokens / len(merged_rows)) if merged_rows else 0.0,
        "hit_max_tokens_count": hit_max_tokens,
        "wall_time_sec": ended_at - started_at,
        "started_at": datetime.utcfromtimestamp(started_at).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "ended_at": datetime.utcfromtimestamp(ended_at).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    save_json(stats, out_path)


def run_combined_xverify(merged_file: Path, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts2" / "eval_shortcut_metrics.py"),
        "--input-path",
        str(merged_file),
        "--xverify-root",
        str(args.xverify_root),
        "--xverify-model-path",
        str(args.xverify_model_path),
        "--gpu",
        args.xverify_gpu,
        "--batch-size",
        str(args.xverify_batch_size),
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(REPO_ROOT)
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def main() -> None:
    args = parse_args()
    if not args.api_key:
        args.api_key = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("YUNWU_API_KEY")
    if not args.api_key:
        raise ValueError("API key is required. Pass --api-key or set API_KEY/OPENAI_API_KEY/YUNWU_API_KEY.")
    ensure_exists(args.data_path, "data file")
    if args.image_source == "local_data_url":
        ensure_exists(args.image_folder, "image folder")

    rows = load_json_array(args.data_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    model_tag = args.model.replace("/", "__")
    out_dir = args.output_root / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    write_model_info_template(out_dir, args)

    merged_file = out_dir / args.data_path.name
    stats_file = out_dir / f"{args.data_path.stem}.generation_stats.json"
    tmp_file = out_dir / f"{args.data_path.stem}.api_tmp.jsonl"

    if merged_file.exists() and not args.overwrite:
        print(f"[skip] merged file already exists: {merged_file}")
        if args.run_xverify:
            run_combined_xverify(merged_file, args)
        return

    started_at = time.time()
    completed = load_tmp_predictions(tmp_file)
    pending_rows = [row for row in rows if int(row["question_id"]) not in completed]

    if pending_rows:
        run_pending_requests(args, rows, pending_rows, completed, tmp_file)

    merged_rows = merge_predictions(rows, completed)
    save_json(merged_rows, merged_file)
    save_generation_stats(merged_rows, stats_file, started_at, time.time())
    if tmp_file.exists():
        tmp_file.unlink()
    print(merged_file)

    if args.run_xverify:
        run_combined_xverify(merged_file, args)


if __name__ == "__main__":
    main()
