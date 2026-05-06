# Make LLaVA training memory efficient.
#
# The legacy xformers monkey patch targets older transformers LLaMA internals.
# Newer transformers versions pass additional attention arguments such as
# cache_position, so prefer the built-in SDPA backend unless explicitly asked
# to use the legacy patch.
import os
from importlib.metadata import version as package_version

from packaging import version


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _use_legacy_xformers_patch() -> bool:
    env_value = os.environ.get("LLAVA_USE_LEGACY_XFORMERS_PATCH")
    if env_value is not None:
        return _truthy(env_value)
    try:
        return version.parse(package_version("transformers")) < version.parse("4.45.0")
    except Exception:
        return False


if _use_legacy_xformers_patch():
    # Need to call this before importing transformers via llava.train.train.
    from llava.train.llama_xformers_attn_monkey_patch import (
        replace_llama_attn_with_xformers_attn,
    )

    replace_llama_attn_with_xformers_attn()
    _ATTN_IMPLEMENTATION = None
else:
    _ATTN_IMPLEMENTATION = os.environ.get("LLAVA_ATTN_IMPLEMENTATION", "sdpa")

from llava.train.train import train

if __name__ == "__main__":
    train(attn_implementation=_ATTN_IMPLEMENTATION)
