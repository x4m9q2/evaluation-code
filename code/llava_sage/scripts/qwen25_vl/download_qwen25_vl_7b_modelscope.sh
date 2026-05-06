#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-/root/venv/unifolm}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Python was not found in ${VENV_DIR}." >&2
  exit 1
fi

exec "${VENV_DIR}/bin/python" "${SCRIPT_DIR}/download_qwen25_vl_7b_modelscope.py" "$@"
