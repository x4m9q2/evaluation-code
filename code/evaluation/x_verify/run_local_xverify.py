#!/usr/bin/env python3
import argparse
import json
import os

from eval import Evaluator
from model import Model


DEFAULT_MODEL_PATH = "xVerify-0.5B-I"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run xVerify local evaluation in single-thread multi-batch mode."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to the input JSON dataset.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Directory to save evaluation output JSON.",
    )
    parser.add_argument(
        "--model-name",
        default="xVerify-0.5B-I",
        help="xVerify model template name.",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=(
            "Local model path. You can pass model directory or vocab.json path. "
            f"Default: {DEFAULT_MODEL_PATH}"
        ),
    )
    parser.add_argument(
        "--data-size",
        type=int,
        default=None,
        help="Optional number of samples to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for single-thread local inference.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum new tokens for generation.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.7,
        help="Top-p for generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_path
    if not os.path.isabs(model_path):
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), model_path))

    model = Model(
        model_name=args.model_name,
        model_path_or_url=model_path,
        inference_mode="local",
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
    )

    evaluator = Evaluator(
        model=model,
        process_num=1,
        batch_size=args.batch_size,
    )

    stat_info = evaluator.evaluate(
        data_path=args.data_path,
        output_path=args.output_path,
        data_size=args.data_size,
    )
    print(json.dumps(stat_info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
