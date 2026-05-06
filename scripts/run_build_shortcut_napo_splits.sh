#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_llava.sh"

SHORTCUT_CODE_DIR="${SHORTCUT_CODE_DIR:-${BUNDLE_ROOT}/code/shortcut_pipeline}"
SHORTCUT_PIPELINE_DIR="${SHORTCUT_PIPELINE_DIR:-${BUNDLE_ROOT}/data/shortcut_pipeline}"
NAPO_ROOT="${NAPO_ROOT:-${BUNDLE_ROOT}/data/napo/shortcut_generated_vqa}"

INPUT_JSON="${INPUT_JSON:-${SHORTCUT_PIPELINE_DIR}/cross_modality_qa_input.json}"
BATCH_OUTPUT_JSONL="${BATCH_OUTPUT_JSONL:-${SHORTCUT_PIPELINE_DIR}/batch_outputs/cross_modality_qa_responses.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${NAPO_ROOT}}"
SPLIT_SEED="${SPLIT_SEED:-3407}"
QUESTION_SOURCE="${QUESTION_SOURCE:-generated}"

echo "Build shortcut NaPO preference splits"
echo "Input:     ${INPUT_JSON}"
echo "Responses: ${BATCH_OUTPUT_JSONL}"
echo "Target:    ${OUTPUT_DIR}"
echo "Seed:      ${SPLIT_SEED}"
echo "Question:  ${QUESTION_SOURCE}"
echo

check_path "${INPUT_JSON}" "stage-2 input JSON"
check_path "${BATCH_OUTPUT_JSONL}" "stage-2 batch output JSONL"

CMD=(
  "${PYTHON_BIN}" "${SHORTCUT_CODE_DIR}/build_shortcut_napo_splits.py"
  --input-json "${INPUT_JSON}"
  --batch-output-jsonl "${BATCH_OUTPUT_JSONL}"
  --output-dir "${OUTPUT_DIR}"
  --seed "${SPLIT_SEED}"
  --question-source "${QUESTION_SOURCE}"
)

run_or_echo "${CMD[@]}"
