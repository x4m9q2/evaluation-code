#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MODEL_PATH="${MODEL_PATH:-${STAGE2_CHECKPOINT}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BUNDLE_ROOT}/outputs/infer_test_raw}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
OVERWRITE="${OVERWRITE:-0}"
NO_SHORT_ANSWER_PROMPT="${NO_SHORT_ANSWER_PROMPT:-1}"
LIMIT="${LIMIT:-}"

cmd=(
  "${PYTHON_BIN}" "${CODE_ROOT}/scripts2/eval_test_raw_gemma3.py"
  --model-path "${MODEL_PATH}"
  --data-path "${TEST_RAW_WITH_SHORTCUT}"
  --image-folder "${STAGE2_IMAGE_FOLDER}"
  --output-root "${OUTPUT_ROOT}"
  --batch-size "${BATCH_SIZE}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE:-0.0}"
  --gpu "${GPU}"
  --gate-text-model-id "${GATE_TEXT_MODEL_ID}"
)

if [[ "${OVERWRITE}" == "1" ]]; then
  cmd+=(--overwrite)
fi
if [[ "${NO_SHORT_ANSWER_PROMPT}" == "1" ]]; then
  cmd+=(--no-short-answer-prompt)
fi
if [[ -n "${LIMIT}" ]]; then
  cmd+=(--limit "${LIMIT}")
fi

"${cmd[@]}"
