#!/usr/bin/env bash
set -euo pipefail

UV_BIN="${UV_BIN:-$(command -v uv)}"
VENV_DIR="${VENV_DIR:-.venv_qwen35_vllm}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

if [[ -z "${UV_BIN}" ]]; then
  echo "uv is required but was not found in PATH." >&2
  exit 1
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${UV_BIN}" venv "${VENV_DIR}" --python "${PYTHON_VERSION}"
fi

"${UV_BIN}" pip install --python "${VENV_DIR}/bin/python" modelscope openai
"${UV_BIN}" pip install --python "${VENV_DIR}/bin/python" \
  vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly

echo "Environment ready at ${VENV_DIR}"
