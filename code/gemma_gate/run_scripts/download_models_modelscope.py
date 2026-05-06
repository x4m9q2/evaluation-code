#!/usr/bin/env python3
"""Download external model weights used by the Gemma gate/mask experiments.

The bundle scripts expect these local paths by default:

  models/Gemma-3-4B-IT
  models/siglip-so400m-patch14-384
  code/gemma_gate/x_verify/xVerify-0.5B-I
"""

from __future__ import annotations

import argparse
from pathlib import Path

from modelscope import snapshot_download


BUNDLE_ROOT = Path(__file__).resolve().parents[3]


DEFAULT_MODELS = {
    # ModelScope mirror of google/gemma-3-4b-it used in the local experiments.
    "gemma3": {
        "model_id": "LLM-Research/Gemma-3-4B-IT",
        "local_dir": str(BUNDLE_ROOT / "models/Gemma-3-4B-IT"),
    },
    # Standalone SigLIP text encoder used by the dual-input gate.
    "siglip": {
        "model_id": "AI-ModelScope/siglip-so400m-patch14-384",
        "local_dir": str(BUNDLE_ROOT / "models/siglip-so400m-patch14-384"),
    },
    # xVerify judge model for accuracy and shortcut-rate measurement.
    "xverify": {
        "model_id": "IAAR-Shanghai/xVerify-0.5B-I",
        "local_dir": str(BUNDLE_ROOT / "code/gemma_gate/x_verify/xVerify-0.5B-I"),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--which",
        nargs="+",
        default=list(DEFAULT_MODELS),
        choices=sorted(DEFAULT_MODELS),
        help="Models to download.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(BUNDLE_ROOT / ".modelscope_cache"),
        help="ModelScope cache directory.",
    )
    args = parser.parse_args()

    for name in args.which:
        spec = DEFAULT_MODELS[name]
        local_dir = Path(spec["local_dir"])
        local_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"[download] {name}: {spec['model_id']} -> {local_dir}")
        snapshot_download(
            spec["model_id"],
            cache_dir=args.cache_dir,
            local_dir=str(local_dir),
        )


if __name__ == "__main__":
    main()
