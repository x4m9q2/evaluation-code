#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_llava.sh"

SHORTCUT_CODE_DIR="${SHORTCUT_CODE_DIR:-${BUNDLE_ROOT}/code/shortcut_pipeline}"
SHORTCUT_PIPELINE_DIR="${SHORTCUT_PIPELINE_DIR:-${BUNDLE_ROOT}/data/shortcut_pipeline}"
MODEL="${MODEL:-gpt-5.4}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-400}"
STAGE2_LIMIT="${STAGE2_LIMIT:--1}"
SUBMIT_API="${SUBMIT_API:-0}"
SUBMIT_LIMIT="${SUBMIT_LIMIT:--1}"
SUBMIT_SLEEP_SECONDS="${SUBMIT_SLEEP_SECONDS:-0.0}"
SUBMIT_TIMEOUT="${SUBMIT_TIMEOUT:-300}"

INPUT_JSON="${INPUT_JSON:-${SHORTCUT_PIPELINE_DIR}/cross_modality_qa_input.json}"
MASK_ROOT="${MASK_ROOT:-${SHORTCUT_PIPELINE_DIR}/output_mask}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${SHORTCUT_PIPELINE_DIR}/batch_inputs/cross_modality_qa_requests.jsonl}"
BATCH_OUTPUT_JSONL="${BATCH_OUTPUT_JSONL:-${SHORTCUT_PIPELINE_DIR}/batch_outputs/cross_modality_qa_responses.jsonl}"

echo "Stage 2 shortcut request generation"
echo "Input:  ${INPUT_JSON}"
echo "Masks:  ${MASK_ROOT}"
echo "Output: ${OUTPUT_JSONL}"
echo "Model:  ${MODEL}"
echo

check_path "${INPUT_JSON}" "stage-2 input JSON"
check_path "${MASK_ROOT}" "masked image root"

CMD=(
  "${PYTHON_BIN}" "${SHORTCUT_CODE_DIR}/run_cross_modality_generation.py"
  --input-json "${INPUT_JSON}"
  --mask-root "${MASK_ROOT}"
  --output-jsonl "${OUTPUT_JSONL}"
  --limit "${STAGE2_LIMIT}"
  --model "${MODEL}"
  --max-output-tokens "${MAX_OUTPUT_TOKENS}"
)

run_or_echo "${CMD[@]}"

if [[ "${SUBMIT_API}" == "1" ]]; then
  echo
  echo "Stage 2 shortcut API submission"
  echo "Input:  ${OUTPUT_JSONL}"
  echo "Output: ${BATCH_OUTPUT_JSONL}"
  echo "Limit:  ${SUBMIT_LIMIT}"
  echo

  check_path "${OUTPUT_JSONL}" "stage-2 batch input JSONL"

  SUBMIT_CMD=(
    "${PYTHON_BIN}" "${SHORTCUT_CODE_DIR}/submit_batch_requests.py"
    --input-jsonl "${OUTPUT_JSONL}"
    --output-jsonl "${BATCH_OUTPUT_JSONL}"
    --model "${MODEL}"
    --limit "${SUBMIT_LIMIT}"
    --sleep-seconds "${SUBMIT_SLEEP_SECONDS}"
    --timeout "${SUBMIT_TIMEOUT}"
  )

  run_or_echo "${SUBMIT_CMD[@]}"
fi
