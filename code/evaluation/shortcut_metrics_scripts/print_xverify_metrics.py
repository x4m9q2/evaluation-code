#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def resolve_result_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(path)

    candidates = sorted(path.glob("Eval_Judge_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No Eval_Judge_*.json found under {path}")
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Print key xVerify metrics from a result JSON file or output directory.")
    parser.add_argument("path", help="Result JSON path or evaluation output directory.")
    args = parser.parse_args()

    result_path = resolve_result_path(args.path)
    with result_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    stat = data.get("stat_info", {})
    print(json.dumps(
        {
            "result_file": str(result_path),
            "Valid_num": stat.get("Valid_num"),
            "Correct_num": stat.get("Correct_num"),
            "Incorrect_num": stat.get("Incorrect_num"),
            "Accuracy": stat.get("Accuracy"),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
