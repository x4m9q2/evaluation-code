#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${BUNDLE_ROOT}/scripts/common_llava.sh"

RUN_ROOT="${SCRIPT_DIR}"
OUT_ROOT="${OUT_ROOT:-${BUNDLE_ROOT}/analysis}"
NUM_SHARDS="${NUM_SHARDS:-4}"
BATCH_SIZE="${BATCH_SIZE:-32}"
RESOLUTION="${RESOLUTION:-1008}"
SCORE_THRESH="${SCORE_THRESH:-0.5}"
DEVICE="${DEVICE:-cuda}"
LOG_DIR="${LOG_DIR:-${LOG_ROOT}/gqa_vg_sam3_union_masks}"
TARGET_DATASET="${1:-${TARGET_DATASET:-all}}"

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

IFS=',' read -r -a GPU_LIST <<< "${MASK_GPUS:-${CUDA_VISIBLE_DEVICES}}"
if [[ "${#GPU_LIST[@]}" -lt "${NUM_SHARDS}" ]]; then
  echo "[error] NUM_SHARDS=${NUM_SHARDS} but only ${#GPU_LIST[@]} GPUs in MASK_GPUS/CUDA_VISIBLE_DEVICES=${MASK_GPUS:-${CUDA_VISIBLE_DEVICES}}" >&2
  exit 2
fi

run_dataset() {
  local name="$1"
  local qa_jsonl="$2"
  local mapping="$3"
  local out_dir="$4"
  shift 4
  local image_root_args=("$@")
  mkdir -p "$out_dir"
  echo "[$(date '+%F %T')] start ${name}" | tee -a "${LOG_DIR}/screen.log"
  for ((shard=0; shard<NUM_SHARDS; shard++)); do
    local gpu="${GPU_LIST[$shard]}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${BUNDLE_ROOT}/code/sam3/scripts/generate_union_masks_from_mapping.py" \
      --qa-jsonl "$qa_jsonl" \
      --mapping-json "$mapping" \
      --output-dir "$out_dir" \
      "${image_root_args[@]}" \
      --batch-size "${BATCH_SIZE}" \
      --resolution "${RESOLUTION}" \
      --score-thresh "${SCORE_THRESH}" \
      --checkpoint-path "${SAM3_CHECKPOINT}" \
      --device "${DEVICE}" \
      --no-load-from-hf \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "$shard" \
      > "${LOG_DIR}/${name}_sam3_union_masks_shard${shard}_of${NUM_SHARDS}.log" 2>&1 &
  done
  wait
  echo "[$(date '+%F %T')] done ${name}" | tee -a "${LOG_DIR}/screen.log"
}

case "${TARGET_DATASET}" in
  gqa)
    run_dataset gqa_sampled10000 \
      "${RUN_ROOT}/gqa_sampled10000.jsonl" \
      "${RUN_ROOT}/gqa_sampled10000_mapping.json" \
      "${OUT_ROOT}/gqa_sampled10000_sam3_union_masks" \
      --image-root "${BUNDLE_ROOT}/data/images/gqa/images"
    ;;
  vg)
    run_dataset vg_sampled10000 \
      "${RUN_ROOT}/vg_sampled10000.jsonl" \
      "${RUN_ROOT}/vg_sampled10000_mapping.json" \
      "${OUT_ROOT}/vg_sampled10000_sam3_union_masks" \
      --image-root "${BUNDLE_ROOT}/data/images/vg/VG_100K" \
      --image-root "${BUNDLE_ROOT}/data/images/vg/VG_100K_2"
    ;;
  all)
    run_dataset gqa_sampled10000 \
      "${RUN_ROOT}/gqa_sampled10000.jsonl" \
      "${RUN_ROOT}/gqa_sampled10000_mapping.json" \
      "${OUT_ROOT}/gqa_sampled10000_sam3_union_masks" \
      --image-root "${BUNDLE_ROOT}/data/images/gqa/images"
    run_dataset vg_sampled10000 \
      "${RUN_ROOT}/vg_sampled10000.jsonl" \
      "${RUN_ROOT}/vg_sampled10000_mapping.json" \
      "${OUT_ROOT}/vg_sampled10000_sam3_union_masks" \
      --image-root "${BUNDLE_ROOT}/data/images/vg/VG_100K" \
      --image-root "${BUNDLE_ROOT}/data/images/vg/VG_100K_2"
    ;;
  *)
    echo "[error] usage: $0 [gqa|vg|all]" >&2
    exit 2
    ;;
esac
