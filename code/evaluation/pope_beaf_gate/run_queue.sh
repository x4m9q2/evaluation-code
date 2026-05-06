#!/usr/bin/env bash
set -uo pipefail

cd /path/to/sage_repro_bundle
export PYTHONPATH=/path/to/sage_repro_bundle:${PYTHONPATH:-}
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

DEBUG_DIR=/path/to/sage_repro_bundle/debug/eval_pope_beaf_gate_20260430
STATUS_LOG=${DEBUG_DIR}/status.tsv
OUT_ROOT=/path/to/sage_repro_bundle/infer_result_pope_beaf_nonumbermaskloss_20260430
GATE_OUT=/path/to/sage_repro_bundle/analysis/gate_patch_activation_nonumbermaskloss_20260430
MODEL=/path/to/sage_repro_bundle/checkpoints/finetune_stage2_qwenratio_oldbase_sam3_qwenkeepmask_nonumbermaskloss_full_bs32_20260425_175907/checkpoint-5148
BEAF_QNA=/path/to/sage_repro_bundle/playground/data/eval/beaf/BEAF_downloads/beaf_qna.json
BEAF_IMAGES=/path/to/sage_repro_bundle/playground/data/eval/beaf/beaf_dataset_ver1

mkdir -p "${DEBUG_DIR}" "${OUT_ROOT}" "${GATE_OUT}"
echo -e "time\tstatus\ttask\tdetail" > "${STATUS_LOG}"

log_status() {
  echo -e "$(date '+%F %T')\t$1\t$2\t$3" >> "${STATUS_LOG}"
}

gpu_busy() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | awk 'NF { found=1 } END { exit found ? 0 : 1 }'
}

wait_for_screen_done() {
  local screen_name=$1
  local max_checks=${2:-1440}
  local checks=0
  while screen -ls | grep -q "${screen_name}"; do
    checks=$((checks + 1))
    echo "[$(date '+%F %T')] waiting screen ${screen_name} (${checks}/${max_checks})"
    if [ "${checks}" -ge "${max_checks}" ]; then
      log_status "WARN" "wait_screen" "timeout waiting ${screen_name}; continuing"
      return 0
    fi
    sleep 60
  done
}

wait_for_gpu_idle() {
  local max_checks=${1:-720}
  local checks=0
  while gpu_busy; do
    checks=$((checks + 1))
    echo "[$(date '+%F %T')] waiting GPU idle (${checks}/${max_checks})"
    if [ "${checks}" -ge "${max_checks}" ]; then
      log_status "WARN" "wait_gpu" "timeout waiting GPU idle; continuing"
      return 0
    fi
    sleep 60
  done
}

run_task() {
  local name=$1
  shift
  echo "[$(date '+%F %T')] start ${name}"
  log_status "START" "${name}" "$*"
  if "$@"; then
    log_status "OK" "${name}" "$*"
    echo "[$(date '+%F %T')] done ${name}"
  else
    local ret=$?
    log_status "FAIL:${ret}" "${name}" "$*"
    echo "[$(date '+%F %T')] failed ${name} ret=${ret}; continue"
  fi
}

wait_for_screen_done eval_data2_epoch_acc_sr_20260430
wait_for_gpu_idle

run_task pope \
  python "${DEBUG_DIR}/eval_pope_model.py" \
    --model-path "${MODEL}" \
    --has-gate true \
    --output-root "${OUT_ROOT}" \
    --gpu 0,1,2,3 \
    --batch-size "${POPE_BATCH_SIZE:-16}" \
    --num-workers "${NUM_WORKERS:-4}" \
    --max-new-tokens 16 \
    --overwrite

wait_for_gpu_idle

run_task beaf \
  python /path/to/sage_repro_bundle/scripts2/eval_beaf.py \
    "${MODEL}" true "${BEAF_BATCH_SIZE:-16}" 0,1,2,3 \
    --qna-path "${BEAF_QNA}" \
    --image-folder "${BEAF_IMAGES}" \
    --output-root "${OUT_ROOT}" \
    --num-workers "${NUM_WORKERS:-4}" \
    --max-new-tokens 16 \
    --overwrite

wait_for_gpu_idle

run_task gate_maps \
  python "${DEBUG_DIR}/draw_gate_patch_activation.py" \
    --model-path "${MODEL}" \
    --output-dir "${GATE_OUT}" \
    --num-images "${GATE_NUM_IMAGES:-12}" \
    --max-new-tokens 32

log_status "DONE" "all" "-"
echo "[$(date '+%F %T')] queue finished"
