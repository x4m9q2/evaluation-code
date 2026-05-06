#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_llava.sh"

MODEL_PATH="${MODEL_PATH:-${LLAVA_EVAL_MODEL}}"
HAS_GATE="${HAS_GATE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES}}"

cd "${LLAVA_CODE_ROOT}"

check_path "${MODEL_PATH}" "model checkpoint"
check_path "${BEAF_QNA_PATH}" "BEAF Q/A file"
check_path "${BEAF_IMAGE_ROOT}" "BEAF image root"

run_or_echo "${PYTHON_BIN}" scripts2/eval_beaf.py \
  "${MODEL_PATH}" \
  "${HAS_GATE}" \
  "${BATCH_SIZE}" \
  "${GPU}" \
  --qna-path "${BEAF_QNA_PATH}" \
  --image-folder "${BEAF_IMAGE_ROOT}" \
  --output-root "${OUTPUT_ROOT}/beaf" \
  ${OVERWRITE:+--overwrite}
