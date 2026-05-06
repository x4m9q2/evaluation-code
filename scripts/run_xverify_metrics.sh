#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

INPUT_PATH="${INPUT_PATH:?Set INPUT_PATH to the merged inference JSON produced by run_eval_test_raw.sh}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
OVERWRITE="${OVERWRITE:-0}"

cmd=(
  "${PYTHON_BIN}" "${CODE_ROOT}/scripts2/eval_shortcut_metrics.py"
  --input-path "${INPUT_PATH}"
  --xverify-root "${XVERIFY_ROOT}"
  --xverify-model-path "${XVERIFY_MODEL}"
  --gpu "${GPU}"
  --batch-size "${BATCH_SIZE}"
)

if [[ "${OVERWRITE}" == "1" ]]; then
  cmd+=(--overwrite)
fi

"${cmd[@]}"

