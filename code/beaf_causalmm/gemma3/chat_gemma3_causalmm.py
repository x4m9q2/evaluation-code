import argparse

import torch
from transformers import set_seed

from causalmm_gemma3 import CausalMMGemma3


BUNDLE_ROOT = __import__("pathlib").Path(__file__).resolve().parents[3]


def parse_args():
    parser = argparse.ArgumentParser(description="Run CausalMM decoding on Gemma 3 4B.")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(BUNDLE_ROOT / "models/Gemma-3-4B-IT"),
    )
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--system", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
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


def main():
    args = parse_args()
    set_seed(args.seed)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    runner = CausalMMGemma3(model_path=args.model_path, torch_dtype=dtype)
    result = runner.generate(
        prompt=args.prompt,
        image_path=args.image,
        system=args.system,
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
    print(result.text)


if __name__ == "__main__":
    main()
