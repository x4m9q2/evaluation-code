#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import mimetypes
from pathlib import Path

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test a local Qwen2.5-VL-7B-Instruct vLLM endpoint.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--prompt", default="Describe the image in one short sentence.")
    parser.add_argument(
        "--image",
        default="/path/to/sage_repro_bundle/playground/data/gqa/images/2415254.jpg",
        help="Optional local image path for a multimodal check.",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    return parser.parse_args()


def to_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def main() -> None:
    args = parse_args()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    message_content: str | list[dict[str, object]]
    image_path = Path(args.image)
    if image_path.is_file():
        message_content = [
            {"type": "text", "text": args.prompt},
            {"type": "image_url", "image_url": {"url": to_data_url(image_path)}},
        ]
    else:
        message_content = args.prompt

    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": message_content}],
        max_tokens=args.max_tokens,
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
