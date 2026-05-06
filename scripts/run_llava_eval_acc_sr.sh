#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_llava.sh"

MODEL_PATH="${MODEL_PATH:-${LLAVA_STAGE2_CHECKPOINT}}"
HAS_GATE="${HAS_GATE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES}}"
LIMIT="${LIMIT:-}"
TORCH_DTYPE="${TORCH_DTYPE:-bf16}"

gpu_count() {
  local gpu_spec="$1"
  if [[ -z "${gpu_spec}" ]]; then
    echo 1
    return
  fi
  awk -F',' '{print NF}' <<< "${gpu_spec}"
}

NUM_CHUNKS="${NUM_CHUNKS:-$(gpu_count "${GPU}")}"

cd "${LLAVA_CODE_ROOT}"

check_path "${MODEL_PATH}" "model checkpoint"
check_path "${TEST_RAW_WITH_SHORTCUT}" "shortcut test JSON"
check_path "${TEST_IMAGE_ROOT}" "test image root"

if [[ -n "${LIMIT}" ]]; then
  run_or_echo "${PYTHON_BIN}" scripts2/batch_infer.py \
    --model-path "${MODEL_PATH}" \
    --data-path "${TEST_RAW_WITH_SHORTCUT}" \
    --dataset "${LLAVA_EVAL_DATASET}" \
    --has-gate "${HAS_GATE}" \
    --image-folder "${TEST_IMAGE_ROOT}" \
    --output-root "${OUTPUT_ROOT}/llava_infer" \
    --gpu "${GPU}" \
    --batch-size "${BATCH_SIZE}" \
    --num-chunks "${NUM_CHUNKS}" \
    --torch-dtype "${TORCH_DTYPE}" \
    --limit "${LIMIT}" \
    ${OVERWRITE:+--overwrite}
  exit 0
fi

run_or_echo "${PYTHON_BIN}" scripts2/batch_infer.py \
  --model-path "${MODEL_PATH}" \
  --data-path "${TEST_RAW_WITH_SHORTCUT}" \
  --dataset "${LLAVA_EVAL_DATASET}" \
  --has-gate "${HAS_GATE}" \
  --image-folder "${TEST_IMAGE_ROOT}" \
  --output-root "${OUTPUT_ROOT}/llava_infer" \
  --gpu "${GPU}" \
  --batch-size "${BATCH_SIZE}" \
  --num-chunks "${NUM_CHUNKS}" \
  --torch-dtype "${TORCH_DTYPE}" \
  ${OVERWRITE:+--overwrite}
