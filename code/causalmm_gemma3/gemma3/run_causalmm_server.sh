#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"

exec uvicorn api_gemma3_causalmm:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8001}"
