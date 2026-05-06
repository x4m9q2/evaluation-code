#!/bin/bash
set -euo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

REPO_ROOT=${REPO_ROOT:-/path/to/sage_repro_bundle}
cd "${REPO_ROOT}"

RUN_TAG_BASE=${RUN_TAG_BASE:-$(date +%Y%m%d_%H%M%S)}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
REPORT_TO=${REPORT_TO:-none}
FORCE_REBUILD_TRAIN_JSON=${FORCE_REBUILD_TRAIN_JSON:-1}

run_one() {
  local dataset="$1"
  local run_tag="${RUN_TAG_BASE}_${dataset}"
  local log_path="${REPO_ROOT}/logs/finetune_data2_${dataset}_from_meanreg_p2p_ddp_${run_tag}.log"
  local output_dir="${REPO_ROOT}/checkpoints/finetune_data2_${dataset}_from_meanreg_p2p_ddp_${run_tag}"

  echo "[$(date '+%F %T')] start dataset=${dataset} run_tag=${run_tag}"
  echo "[$(date '+%F %T')] log=${log_path}"
  echo "[$(date '+%F %T')] output_dir=${output_dir}"

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  REPORT_TO="${REPORT_TO}" \
  FORCE_REBUILD_TRAIN_JSON="${FORCE_REBUILD_TRAIN_JSON}" \
  DATASET="${dataset}" \
  RUN_TAG="${run_tag}" \
  OUTPUT_DIR="${output_dir}" \
  stdbuf -oL -eL bash "${REPO_ROOT}/scripts/v1_5/finetune_data2_single_from_meanreg_p2p_ddp.sh" \
    > "${log_path}" 2>&1

  echo "[$(date '+%F %T')] finished dataset=${dataset} run_tag=${run_tag}"
}

run_one vg
run_one gqa
