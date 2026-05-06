import argparse
import json
from pathlib import Path
from typing import Dict, List


def parse_args():
    parser = argparse.ArgumentParser(description="Merge sharded Gemma3 eval jsonl outputs into one ordered json/jsonl.")
    parser.add_argument("shard_files", nargs="+", help="Shard jsonl files to merge.")
    parser.add_argument("--question-file", required=True, help="Original jsonl dataset for restoring source order.")
    parser.add_argument("--output-file", required=True, help="Final merged .json output path.")
    return parser.parse_args()


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    args = parse_args()
    question_rows = load_jsonl(Path(args.question_file))
    question_id_to_index: Dict[int, int] = {
        int(row["question_id"]): idx for idx, row in enumerate(question_rows)
    }

    merged_by_index: Dict[int, dict] = {}
    for shard_file in args.shard_files:
        for item in load_jsonl(Path(shard_file)):
            if "source_index" in item:
                source_index = int(item["source_index"])
            else:
                source_index = question_id_to_index[int(item["question_id"])]
            if source_index in merged_by_index:
                raise ValueError(f"Duplicate source_index {source_index} from {shard_file}")
            merged_by_index[source_index] = item

    ordered_indices = sorted(merged_by_index)
    merged_jsonl_path = Path(str(args.output_file) + ".jsonl")
    with merged_jsonl_path.open("w", encoding="utf-8") as f:
        for source_index in ordered_indices:
            f.write(json.dumps(merged_by_index[source_index], ensure_ascii=False) + "\n")

    final_rows = [
        {
            "question": merged_by_index[source_index]["question"],
            "llm_output": merged_by_index[source_index]["llm_output"],
            "correct_answer": merged_by_index[source_index]["correct_answer"],
            "answer_type": merged_by_index[source_index]["answer_type"],
        }
        for source_index in ordered_indices
    ]
    with Path(args.output_file).open("w", encoding="utf-8") as f:
        json.dump(final_rows, f, ensure_ascii=False, indent=2)

    print(args.output_file)
    print(merged_jsonl_path)


if __name__ == "__main__":
    main()
