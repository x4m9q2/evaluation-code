#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge sharded Qwen visual-cue filter outputs into a merged directory."
    )
    parser.add_argument("--filter-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def sort_key(row: dict[str, Any]) -> int:
    return int(row["question_id"])


def main() -> None:
    args = parse_args()
    filter_root = args.filter_root
    if not filter_root.exists():
        raise FileNotFoundError(filter_root)

    keep_rows: list[dict[str, Any]] = []
    remove_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    shard_summaries: list[dict[str, Any]] = []

    for shard in range(args.num_shards):
        run_dir = filter_root / f"run_{shard:02d}"
        if not run_dir.exists():
            raise FileNotFoundError(run_dir)

        keep_path = run_dir / "keep.json"
        remove_path = run_dir / "remove.json"
        summary_path = run_dir / "summary.json"
        audit_path = run_dir / "audit.jsonl"
        for path in (keep_path, remove_path, summary_path, audit_path):
            if not path.exists():
                raise FileNotFoundError(path)

        keep_rows.extend(load_json(keep_path))
        remove_rows.extend(load_json(remove_path))
        shard_summaries.append(load_json(summary_path))
        with audit_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    audit_rows.append(json.loads(line))

    keep_rows.sort(key=sort_key)
    remove_rows.sort(key=sort_key)
    audit_rows.sort(key=sort_key)

    keep_qids = {int(row["question_id"]) for row in keep_rows}
    remove_qids = {int(row["question_id"]) for row in remove_rows}
    if keep_qids & remove_qids:
        overlap = sorted(keep_qids & remove_qids)[:20]
        raise ValueError(f"keep/remove overlap: {overlap}")
    if len(keep_qids) != len(keep_rows):
        raise ValueError("Duplicate question_id found in keep rows.")
    if len(remove_qids) != len(remove_rows):
        raise ValueError("Duplicate question_id found in remove rows.")

    merged_dir = filter_root / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    write_json(merged_dir / "keep.json", keep_rows)
    write_json(merged_dir / "remove.json", remove_rows)

    with (merged_dir / "audit.jsonl").open("w", encoding="utf-8") as f:
        for row in audit_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "filter_root": str(filter_root),
        "total_rows": len(keep_rows) + len(remove_rows),
        "kept_rows": len(keep_rows),
        "removed_rows": len(remove_rows),
        "removed_ratio": len(remove_rows) / max(1, len(keep_rows) + len(remove_rows)),
        "merged_from_runs": [f"run_{shard:02d}" for shard in range(args.num_shards)],
        "shards": shard_summaries,
    }
    write_json(merged_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
