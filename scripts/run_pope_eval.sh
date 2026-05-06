#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_llava.sh"

MODEL_PATH="${MODEL_PATH:-${LLAVA_EVAL_MODEL}}"
HAS_GATE="${HAS_GATE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES}}"

cd "${LLAVA_CODE_ROOT}"

check_path "${MODEL_PATH}" "model checkpoint"
check_path "${POPE_QUESTION_FILE}" "POPE question file"
check_path "${POPE_ANNOTATION_DIR}" "POPE annotation dir"
check_path "${POPE_IMAGE_ROOT}" "POPE image root"

run_or_echo "${PYTHON_BIN}" "${BUNDLE_ROOT}/code/evaluation/pope_beaf_gate/eval_pope_model.py" \
  --model-path "${MODEL_PATH}" \
  --has-gate "${HAS_GATE}" \
  --question-file "${POPE_QUESTION_FILE}" \
  --annotation-dir "${POPE_ANNOTATION_DIR}" \
  --image-folder "${POPE_IMAGE_ROOT}" \
  --output-root "${OUTPUT_ROOT}/pope" \
  --gpu "${GPU}" \
  --batch-size "${BATCH_SIZE}" \
  ${OVERWRITE:+--overwrite}
