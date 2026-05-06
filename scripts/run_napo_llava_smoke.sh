#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_llava.sh"

STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-napo_llava_smoke_${STAMP}}"
SMOKE_ROOT="${SMOKE_ROOT:-/tmp/napo_llava_smoke}"
LOG_FILE="${LOG_FILE:-${BUNDLE_ROOT}/logs/${RUN_NAME}.log}"
SMOKE_SAMPLE_SIZE="${SMOKE_SAMPLE_SIZE:-32}"
SMOKE_DATA_DIR="${SMOKE_DATA_DIR:-${SMOKE_ROOT}/${RUN_NAME}_data}"
export SMOKE_SAMPLE_SIZE
export SMOKE_DATA_DIR

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
export PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-2}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
export NUM_EPOCHS="${NUM_EPOCHS:-1}"
export LOGGING_STEPS="${LOGGING_STEPS:-1}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
export DRY_RUN="${DRY_RUN:-0}"
export OUTPUT_DIR="${OUTPUT_DIR:-${SMOKE_ROOT}/${RUN_NAME}}"
export LOGGING_DIR="${LOGGING_DIR:-${OUTPUT_DIR}/logs}"
export NAPO_LLAVA_DATA_DIR="${SMOKE_DATA_DIR}"
export MASTER_PORT="${MASTER_PORT:-29611}"
export EXTRA_ARGS="--max_steps ${MAX_STEPS:-1} --save_strategy no ${EXTRA_ARGS:-}"

require_cuda_visible_devices_count 4 "NaPO LLaVA smoke"

mkdir -p "$(dirname "${LOG_FILE}")" "${SMOKE_ROOT}"

rm -rf "${SMOKE_DATA_DIR}"
"${PYTHON_BIN}" - <<'PY'
import os
from datasets import load_from_disk

src = os.environ["BUNDLE_ROOT"] + "/data/napo_llava/train_raw_pos_neg_shortcut_hf"
dst = os.environ["SMOKE_DATA_DIR"]
n = int(os.environ["SMOKE_SAMPLE_SIZE"])

ds = load_from_disk(src)
train = ds["train"].select(range(min(n, len(ds["train"]))))
train.save_to_disk(dst)
print(f"[smoke-data] wrote {len(train)} examples to {dst}")
PY

{
  echo "[run] ${RUN_NAME}"
  echo "[model] ${LLAVA_BASE_MODEL}"
  echo "[vision] ${LLAVA_VISION_TOWER}"
  echo "[data] ${NAPO_LLAVA_DATA_DIR}"
  echo "[output] ${OUTPUT_DIR}"
  echo "[gpus] ${CUDA_VISIBLE_DEVICES}"
  echo "[master_port] ${MASTER_PORT}"
  echo "[batch] per_device=${PER_DEVICE_TRAIN_BATCH_SIZE} grad_accum=${GRADIENT_ACCUMULATION_STEPS}"
  echo "[max_steps] ${MAX_STEPS:-1}"
  date -u
} | tee "${LOG_FILE}"

(
  cd "${BUNDLE_ROOT}"
  bash scripts/run_napo_llava.sh
) 2>&1 | tee -a "${LOG_FILE}"
status=${PIPESTATUS[0]}
if [[ "${status}" -ne 0 ]]; then
  echo "[error] NaPO LLaVA smoke test failed with exit code ${status}" | tee -a "${LOG_FILE}"
  exit "${status}"
fi

if rg -n -i '\bnan\b' "${LOG_FILE}" >/dev/null; then
  echo "[error] Found NaN in smoke log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
  exit 1
fi

echo "[ok] NaPO LLaVA smoke test completed without NaN: ${LOG_FILE}" | tee -a "${LOG_FILE}"
