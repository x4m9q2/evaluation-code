#!/usr/bin/env python3
"""Submit stage-2 request JSONL rows to a Responses-compatible API."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib import error, request


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPE_ROOT = REPO_ROOT / "data" / "shortcut_pipeline"
DEFAULT_INPUT = PIPE_ROOT / "batch_inputs" / "cross_modality_qa_requests.jsonl"
DEFAULT_OUTPUT = PIPE_ROOT / "batch_outputs" / "cross_modality_qa_responses.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-jsonl", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--model", default=os.environ.get("MODEL", ""))
    parser.add_argument("--limit", type=int, default=-1, help="<= 0 keeps all rows.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=300)
    return parser.parse_args()


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_request_url(base_url: str, request_path: str) -> str:
    if base_url.endswith("/v1") and request_path.startswith("/v1/"):
        request_path = request_path[len("/v1") :]
    return base_url + request_path


def extract_output_text(payload: dict[str, Any]) -> str:
    texts: list[str] = []
    for item in payload.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                texts.append(str(content["text"]))
    return "\n".join(texts)


def post_json(url: str, api_key: str, payload: dict[str, Any], timeout: int) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, body
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return exc.code, body


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl).resolve()
    output_path = Path(args.output_jsonl).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    base_url = args.base_url.strip()
    api_key = args.api_key.strip()
    model = args.model.strip()

    if not base_url:
        raise RuntimeError("No base URL found. Pass --base-url or set OPENAI_BASE_URL.")
    if not api_key:
        raise RuntimeError("No API key found. Pass --api-key or set OPENAI_API_KEY.")
    if not model:
        raise RuntimeError("No model found. Pass --model or set MODEL.")

    rows = iter_jsonl(input_path)
    if args.limit > 0:
        rows = rows[: args.limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = 0
    errors = 0
    with output_path.open("w", encoding="utf-8") as out_f:
        for idx, row in enumerate(rows, start=1):
            body = dict(row.get("body", {}))
            body["model"] = model

            request_path = str(row.get("url", "")).strip()
            url = resolve_request_url(base_url, request_path)
            status_code, response_text = post_json(url=url, api_key=api_key, payload=body, timeout=args.timeout)

            result: dict[str, Any] = {
                "custom_id": row.get("custom_id"),
                "request_url": request_path,
                "resolved_url": url,
                "model": model,
                "status_code": status_code,
            }
            try:
                payload = json.loads(response_text)
                result["response"] = payload
                output_text = extract_output_text(payload) if isinstance(payload, dict) else ""
                if output_text:
                    result["output_text"] = output_text
            except json.JSONDecodeError:
                result["response_text"] = response_text

            out_f.write(json.dumps(result, ensure_ascii=False))
            out_f.write("\n")
            out_f.flush()

            if status_code == 200:
                success += 1
            else:
                errors += 1

            print(f"[{idx}/{len(rows)}] status={status_code} custom_id={row.get('custom_id')}", flush=True)

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    print(
        json.dumps(
            {
                "input_jsonl": str(input_path),
                "output_jsonl": str(output_path),
                "model": model,
                "base_url": base_url,
                "total": len(rows),
                "success": success,
                "errors": errors,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"[error] {exc}", file=sys.stderr)
        raise
