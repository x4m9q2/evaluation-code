#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_llava.sh"

MODEL_PATH="${MODEL_PATH:-${LLAVA_STAGE2_CHECKPOINT}}"
HAS_GATE="${HAS_GATE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES}}"

cd "${LLAVA_CODE_ROOT}"

check_path "${MODEL_PATH}" "model checkpoint"
check_path "${TEST_RAW_WITH_SHORTCUT}" "shortcut test JSON"
check_path "${TEST_IMAGE_ROOT}" "test image root"
check_path "${XVERIFY_MODEL}" "xVerify model"

run_or_echo "${PYTHON_BIN}" scripts2/batch_infer.py \
  --model-path "${MODEL_PATH}" \
  --data-path "${TEST_RAW_WITH_SHORTCUT}" \
  --has-gate "${HAS_GATE}" \
  --image-folder "${TEST_IMAGE_ROOT}" \
  --output-root "${OUTPUT_ROOT}/llava_infer" \
  --gpu "${GPU}" \
  --batch-size "${BATCH_SIZE}" \
  --run-xverify \
  --xverify-root "${XVERIFY_ROOT}" \
  --xverify-model-path "${XVERIFY_MODEL}" \
  --xverify-gpu "${XVERIFY_GPU:-0}" \
  --xverify-batch-size "${XVERIFY_BATCH_SIZE:-32}" \
  ${OVERWRITE:+--overwrite}
