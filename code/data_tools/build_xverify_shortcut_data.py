import argparse
import json
from typing import List, Dict, Any


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    # First try normal JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: treat as JSONL
    data = []
    for line_idx, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse file as JSON or JSONL.\n"
                f"Path: {path}\n"
                f"Bad line: {line_idx}\n"
                f"Content: {line[:200]}\n"
                f"Error: {e}"
            )
    return data


def save_json(data: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_pred_text(text: str) -> str:
    """
    Convert prediction text to XVerify-style output.
    Example: 'right' -> 'Right'
    """
    if text is None:
        return ""
    text = str(text).strip()
    if not text:
        return ""
    return text[0].upper() + text[1:]


def build_vqa_index(vqa_data: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    Build an index using question_id as key.
    """
    index = {}
    for item in vqa_data:
        qid = item.get("question_id")
        if qid is not None:
            index[qid] = item
    return index


def convert_to_xverify_format(
    pred_path: str,
    vqa_path: str,
    output_path: str
) -> None:
    pred_data = load_json(pred_path)
    vqa_data = load_json(vqa_path)

    vqa_index = build_vqa_index(vqa_data)

    output_data = []
    missing_count = 0

    for item in pred_data:
        question_id = item.get("question_id")
        question = item.get("prompt", "")
        pred = item.get("text", "")
        answer = item.get("answer", "")

        if question_id not in vqa_index:
            missing_count += 1
            continue

        vqa_item = vqa_index[question_id]

        output_item = {
            "question": question,
            "llm_output": pred,
            "correct_answer": vqa_item.get("answer", ""),
            "answer_type": vqa_item.get("answer_type", "")
        }
        output_data.append(output_item)

    save_json(output_data, output_path)

    print(f"Done. Saved {len(output_data)} samples to: {output_path}")
    if missing_count > 0:
        print(f"Warning: {missing_count} samples could not be matched by question_id.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert prediction JSON/JSONL plus a VQA-style answer file into xVerify input format. "
            "When vqa_path points to the original VQA answers for anti-shortcut data, "
            "the output can be used to measure shortcut rate."
        )
    )
    parser.add_argument("--pred-path", required=True, help="Prediction file in JSON or JSONL format.")
    parser.add_argument("--vqa-path", required=True, help="VQA-style answer file with question_id/answer fields.")
    parser.add_argument("--output-path", required=True, help="Path to save converted xVerify JSON.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_to_xverify_format(args.pred_path, args.vqa_path, args.output_path)
