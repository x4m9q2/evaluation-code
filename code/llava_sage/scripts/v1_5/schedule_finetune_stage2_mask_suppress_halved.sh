#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

BASE_SCRIPT="${BASE_SCRIPT:-${REPO_ROOT}/scripts/v1_5/finetune_stage2_mask_suppress_ddp_p2p.sh}"
START_AFTER_HOURS="${START_AFTER_HOURS:-4}"
DELAY_SECONDS="${DELAY_SECONDS:-$((START_AFTER_HOURS * 3600))}"
MASK_PATCH_LOSS_WEIGHT="${MASK_PATCH_LOSS_WEIGHT:-1.25}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MASTER_PORT="${MASTER_PORT:-29535}"
REPORT_TO="${REPORT_TO:-none}"
WAIT_FOR_IDLE_GPU="${WAIT_FOR_IDLE_GPU:-1}"
IDLE_POLL_SECONDS="${IDLE_POLL_SECONDS:-60}"

echo "Scheduler started at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Repository root: ${REPO_ROOT}"
echo "Base training script: ${BASE_SCRIPT}"
echo "Will wait ${DELAY_SECONDS} seconds before launching."
echo "Target MASK_PATCH_LOSS_WEIGHT=${MASK_PATCH_LOSS_WEIGHT}"
echo "Target CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Target MASTER_PORT=${MASTER_PORT}"

sleep "${DELAY_SECONDS}"

echo "Delay finished at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

if [[ "${WAIT_FOR_IDLE_GPU}" != "0" ]]; then
  while true; do
    BUSY_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d' | sort -u)
    if [[ -z "${BUSY_PIDS}" ]]; then
      echo "GPUs are idle. Proceeding to launch training."
      break
    fi
    echo "GPUs are still busy with PIDs: ${BUSY_PIDS}"
    echo "Sleeping ${IDLE_POLL_SECONDS}s before checking again."
    sleep "${IDLE_POLL_SECONDS}"
  done
fi

RUN_TS=$(date +%Y%m%d_%H%M%S)
RUN_NAME="finetune_stage2_train_raw_mask_suppress_maskloss1p25_ddp_p2p_assembled_meanreg_${RUN_TS}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/checkpoints/${RUN_NAME}}"
TRAIN_LOG="${TRAIN_LOG:-${REPO_ROOT}/logs/${RUN_NAME}.log}"

echo "Launching delayed training run at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Run name: ${RUN_NAME}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Train log: ${TRAIN_LOG}"

(
  cd "${REPO_ROOT}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  MASTER_PORT="${MASTER_PORT}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  REPORT_TO="${REPORT_TO}" \
  MASK_PATCH_LOSS_WEIGHT="${MASK_PATCH_LOSS_WEIGHT}" \
  bash "${BASE_SCRIPT}"
) 2>&1 | tee "${TRAIN_LOG}"
