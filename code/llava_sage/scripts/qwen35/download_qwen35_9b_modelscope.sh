#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-/root/venv/qwen35_vllm}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Python was not found in ${VENV_DIR}. Run scripts/qwen35/setup_qwen35_vllm_env.sh first." >&2
  exit 1
fi

source "${SCRIPT_DIR}/_common.sh"
qwen35_prepend_runtime_libs "${VENV_DIR}"

exec "${VENV_DIR}/bin/python" "${SCRIPT_DIR}/download_qwen35_9b_modelscope.py" "$@"
