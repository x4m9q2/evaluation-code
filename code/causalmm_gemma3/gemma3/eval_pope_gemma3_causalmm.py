import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import set_seed

from causalmm_gemma3 import CausalMMGemma3


BUNDLE_ROOT = Path(__file__).resolve().parents[3]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate POPE-style json/jsonl questions with Gemma 3 + CausalMM.")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(BUNDLE_ROOT / "models/Gemma-3-4B-IT"),
    )
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
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


def load_questions(path):
    with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def main():
    args = parse_args()
    set_seed(args.seed)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    runner = CausalMMGemma3(model_path=args.model_path, torch_dtype=dtype)
    questions = load_questions(args.question_file)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    with open(answers_file, "w", encoding="utf-8") as ans_file:
        for line in tqdm(questions):
            question_id = line.get("question_id", line.get("id"))
            image_file = line["image"]
            prompt = line.get("text", line.get("question", ""))
            result = runner.generate(
                prompt=prompt + " Answer briefly.",
                image_path=os.path.join(args.image_folder, image_file),
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
            ans_file.write(
                json.dumps(
                    {
                        "question_id": question_id,
                        "prompt": prompt,
                        "text": result.text,
                        "model_id": "gemma-3-4b-it-causalmm",
                        "image": image_file,
                        "metadata": {
                            "gamma": args.gamma,
                            "epsilon": args.epsilon,
                            "cf_mode": args.cf_mode,
                            "attention_method": args.attention_method,
                            "vision_method": args.vision_method,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            ans_file.flush()


if __name__ == "__main__":
    main()
