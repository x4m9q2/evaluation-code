import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from tqdm import tqdm
from transformers import set_seed

from causalmm_gemma3 import CausalMMGemma3


BUNDLE_ROOT = Path(__file__).resolve().parents[3]


def parse_args():
    parser = argparse.ArgumentParser(description="Run Gemma 3 + CausalMM on test_raw_llava.jsonl.")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(BUNDLE_ROOT / "models/Gemma-3-4B-IT"),
    )
    parser.add_argument(
        "--question-file",
        type=str,
        default=str(BUNDLE_ROOT / "outputs/causalmm_gemma3/test_raw_llava.jsonl"),
    )
    parser.add_argument(
        "--answer-file",
        type=str,
        default=str(BUNDLE_ROOT / "data/eval/test_raw_with_shortcut_answer.json"),
    )
    parser.add_argument(
        "--image-folder",
        type=str,
        default=str(BUNDLE_ROOT / "data/playground_data/coco/train2014"),
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=str(BUNDLE_ROOT / "outputs/causalmm_gemma3/causalmm_gemma3_test_raw_results.json"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cf-mode", choices=["language", "vision", "both"], default="language")
    parser.add_argument(
        "--attention-method",
        choices=["reverse", "reverse_and_normalize", "random", "uniform", "shuffle", "none"],
        default="reverse_and_normalize",
    )
    parser.add_argument(
        "--vision-method",
        choices=["shuffle", "uniform", "reverse", "random", "none"],
        default="shuffle",
    )
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    return parser.parse_args()


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_answers(path: str) -> Dict[int, dict]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return {}
    if text[0] == "[":
        data = json.loads(text)
    else:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    return {int(item["question_id"]): item for item in data}


def clean_question(text: str) -> str:
    return text.replace("<image>", "").strip()


def resolve_image(image_folder: str, image_value: str) -> str:
    if os.path.isabs(image_value):
        return image_value
    return os.path.join(image_folder, image_value)


def load_done_ids(path: Path) -> set:
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if "question_id" in item:
                done.add(int(item["question_id"]))
    return done


def write_final_json(tmp_file: Path, output_file: Path) -> None:
    rows = []
    with tmp_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            row = dict(item.get("source", {}))
            row["question_id"] = int(item["question_id"])
            row["question"] = item["question"]
            row["model_pred"] = item["llm_output"]
            row["llm_output"] = item["llm_output"]
            row["answer"] = item["correct_answer"]
            row["correct_answer"] = item["correct_answer"]
            row["answer_type"] = item["answer_type"]
            rows.append(row)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def chunks(items: List[dict], size: int) -> Iterable[List[dict]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def main():
    args = parse_args()
    set_seed(args.seed)

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    question_rows = load_jsonl(args.question_file)
    answer_by_id = load_answers(args.answer_file)
    if args.limit is not None:
        question_rows = question_rows[: args.limit]

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = output_file.with_suffix(output_file.suffix + ".jsonl")

    done_ids = load_done_ids(tmp_file) if args.resume else set()
    mode = "a" if args.resume else "w"

    runner = CausalMMGemma3(model_path=args.model_path, torch_dtype=dtype)

    pending_rows = [row for row in question_rows if int(row["question_id"]) not in done_ids]
    with tmp_file.open(mode, encoding="utf-8") as f:
        progress = tqdm(total=len(pending_rows), desc="Gemma3+CausalMM")
        for batch_rows in chunks(pending_rows, max(1, args.batch_size)):
            prompts = []
            image_paths = []
            answer_items = []
            question_ids = []

            for row in batch_rows:
                question_id = int(row["question_id"])
                answer_item = answer_by_id.get(question_id)
                if answer_item is None:
                    raise KeyError(f"question_id {question_id} not found in {args.answer_file}")

                question = clean_question(row["text"])
                prompts.append(question + "\nAnswer with only the final answer, using as few words as possible.")
                image_paths.append(resolve_image(args.image_folder, row["image"]))
                answer_items.append(answer_item)
                question_ids.append(question_id)

            results = runner.generate_batch(
                prompts=prompts,
                image_paths=image_paths,
                max_new_tokens=args.max_new_tokens,
                gamma=args.gamma,
                epsilon=args.epsilon,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                cf_mode=args.cf_mode,
                attention_method=args.attention_method,
                vision_method=args.vision_method,
            )

            for question_id, answer_item, result in zip(question_ids, answer_items, results):
                f.write(
                    json.dumps(
                        {
                            "question_id": question_id,
                            "question": answer_item["question"],
                            "llm_output": result.text,
                            "correct_answer": answer_item.get("answer", answer_item.get("correct_answer", "")),
                            "answer_type": answer_item.get("answer_type", ""),
                            "source": answer_item,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            f.flush()
            progress.update(len(batch_rows))
        progress.close()

    write_final_json(tmp_file, output_file)
    print(output_file)


if __name__ == "__main__":
    main()
