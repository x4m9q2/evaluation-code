#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import math
import mimetypes
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm


def split_list(lst: list[dict], n: int) -> list[list[dict]]:
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst: list[dict], n: int, k: int) -> list[dict]:
    return split_list(lst, n)[k]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch inference against a local Qwen3.5 VL vLLM endpoint.")
    parser.add_argument("--question-file", required=True, help="JSONL with question_id/image/text fields.")
    parser.add_argument("--answers-file", required=True, help="Output JSONL path.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="Qwen3.5-9B")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=5)
    return parser.parse_args()


def load_questions(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@lru_cache(maxsize=128)
def to_data_url(image_path_str: str) -> str:
    image_path = Path(image_path_str)
    mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def get_text_from_response(response) -> str:
    message = response.choices[0].message
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")).strip())
        merged = "\n".join(part for part in parts if part).strip()
        if merged:
            return merged
    reasoning = getattr(message, "reasoning", "")
    return str(reasoning or "").strip()


_THREAD_LOCAL = threading.local()


def get_client(base_url: str, api_key: str, timeout: float) -> OpenAI:
    client = getattr(_THREAD_LOCAL, "client", None)
    if client is None:
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        _THREAD_LOCAL.client = client
    return client


def infer_one(item: dict, args: argparse.Namespace) -> dict:
    client = get_client(args.base_url, args.api_key, args.timeout)
    content = [
        {"type": "text", "text": str(item["text"])},
        {"type": "image_url", "image_url": {"url": to_data_url(str(item["image"]))}},
    ]
    request_kwargs = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    if args.top_p is not None:
        request_kwargs["top_p"] = args.top_p

    last_error = None
    for attempt in range(1, args.retries + 1):
        try:
            response = client.chat.completions.create(**request_kwargs)
            return {
                "question_id": item["question_id"],
                "prompt": item["text"],
                "text": get_text_from_response(response),
                "answer_id": uuid.uuid4().hex,
                "model_id": args.model,
                "metadata": {
                    "base_url": args.base_url,
                },
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == args.retries:
                break
            time.sleep(min(2 ** (attempt - 1), 8))

    raise RuntimeError(f"Failed for question_id={item['question_id']}: {last_error}") from last_error


def main() -> None:
    args = parse_args()
    questions = get_chunk(load_questions(args.question_file), args.num_chunks, args.chunk_idx)

    Path(args.answers_file).parent.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(infer_one, item, args) for item in questions]
        for future in tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())

    results.sort(key=lambda x: x["question_id"])
    with open(args.answers_file, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "answers_file": args.answers_file,
                "count": len(results),
                "chunk_idx": args.chunk_idx,
                "num_chunks": args.num_chunks,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
