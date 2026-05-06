#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_llava.sh"

TARGET="${1:-all}"
NUM_SHARDS="${NUM_SHARDS:-4}"
FILTER_BATCH_SIZE="${FILTER_BATCH_SIZE:-64}"
FILTER_MAX_MODEL_LEN="${FILTER_MAX_MODEL_LEN:-2048}"
FILTER_MAX_TOKENS="${FILTER_MAX_TOKENS:-256}"
FILTER_GPU_MEMORY_UTILIZATION="${FILTER_GPU_MEMORY_UTILIZATION:-0.9}"
FILTER_TEMPERATURE="${FILTER_TEMPERATURE:-0.0}"
QWEN_MODEL="${QWEN_MODEL:-${BUNDLE_ROOT}/models/Qwen3.5-9B}"
FILTER_LOG_DIR="${FILTER_LOG_DIR:-${LOG_ROOT}/qwen_visual_cue_filter}"

VQA_FILTER_INPUT_ROOT="${VQA_FILTER_INPUT_ROOT:-${BUNDLE_ROOT}/outputs/sam3_train_raw_llava_union_masks}"
VQA_FILTER_OUTPUT_ROOT="${VQA_FILTER_OUTPUT_ROOT:-${BUNDLE_ROOT}/analysis/visual_cue_question_filter_qwen35_strict4/full_all_shards}"
GQA_FILTER_INPUT_ROOT="${GQA_FILTER_INPUT_ROOT:-${BUNDLE_ROOT}/analysis/gqa_sampled10000_sam3_union_masks}"
GQA_FILTER_OUTPUT_ROOT="${GQA_FILTER_OUTPUT_ROOT:-${BUNDLE_ROOT}/analysis/gqa_sampled10000_qwen35_filter}"
VG_FILTER_INPUT_ROOT="${VG_FILTER_INPUT_ROOT:-${BUNDLE_ROOT}/analysis/vg_sampled10000_sam3_union_masks}"
VG_FILTER_OUTPUT_ROOT="${VG_FILTER_OUTPUT_ROOT:-${BUNDLE_ROOT}/analysis/vg_sampled10000_qwen35_filter}"

mkdir -p "${FILTER_LOG_DIR}"

IFS=',' read -r -a GPU_LIST <<< "${FILTER_GPUS:-${CUDA_VISIBLE_DEVICES}}"
if [[ "${#GPU_LIST[@]}" -lt "${NUM_SHARDS}" ]]; then
  echo "[error] NUM_SHARDS=${NUM_SHARDS} but only ${#GPU_LIST[@]} GPUs in FILTER_GPUS/CUDA_VISIBLE_DEVICES=${FILTER_GPUS:-${CUDA_VISIBLE_DEVICES}}" >&2
  exit 2
fi

run_dataset() {
  local name="$1"
  local input_root="$2"
  local output_root="$3"

  local shard_meta_root="${input_root}/shard_meta"
  if [[ ! -d "${shard_meta_root}" ]]; then
    echo "[missing] shard_meta directory: ${shard_meta_root}" >&2
    echo "[hint] rerun the SAM3 mask generation step first so shard_meta/shard_XX.json is recreated." >&2
    exit 2
  fi

  mkdir -p "${output_root}" "${output_root}/logs"
  echo "[$(date '+%F %T')] start ${name}" | tee -a "${FILTER_LOG_DIR}/screen.log"
  for ((shard=0; shard<NUM_SHARDS; shard++)); do
    local gpu="${GPU_LIST[$shard]}"
    local ss
    ss=$(printf '%02d' "${shard}")
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${BUNDLE_ROOT}/code/llava_sage/scripts2/filter_visual_cue_mentions_qwen.py" \
      --input-json "${shard_meta_root}/shard_${ss}.json" \
      --output-dir "${output_root}/run_${ss}" \
      --model-path "${QWEN_MODEL}" \
      --gpu 0 \
      --batch-size "${FILTER_BATCH_SIZE}" \
      --max-model-len "${FILTER_MAX_MODEL_LEN}" \
      --max-tokens "${FILTER_MAX_TOKENS}" \
      --gpu-memory-utilization "${FILTER_GPU_MEMORY_UTILIZATION}" \
      --temperature "${FILTER_TEMPERATURE}" \
      --overwrite \
      > "${output_root}/logs/run_${ss}.log" 2>&1 &
  done
  wait
  "${PYTHON_BIN}" "${BUNDLE_ROOT}/code/llava_sage/scripts2/merge_qwen_filter_runs.py" \
    --filter-root "${output_root}" \
    --num-shards "${NUM_SHARDS}" \
    > "${output_root}/logs/merge.log" 2>&1
  echo "[$(date '+%F %T')] done ${name}" | tee -a "${FILTER_LOG_DIR}/screen.log"
}

case "${TARGET}" in
  vqa)
    run_dataset vqa_stage2 "${VQA_FILTER_INPUT_ROOT}" "${VQA_FILTER_OUTPUT_ROOT}"
    ;;
  gqa)
    run_dataset gqa_sampled10000 "${GQA_FILTER_INPUT_ROOT}" "${GQA_FILTER_OUTPUT_ROOT}"
    ;;
  vg)
    run_dataset vg_sampled10000 "${VG_FILTER_INPUT_ROOT}" "${VG_FILTER_OUTPUT_ROOT}"
    ;;
  all|gqa-vg)
    run_dataset gqa_sampled10000 "${GQA_FILTER_INPUT_ROOT}" "${GQA_FILTER_OUTPUT_ROOT}"
    run_dataset vg_sampled10000 "${VG_FILTER_INPUT_ROOT}" "${VG_FILTER_OUTPUT_ROOT}"
    ;;
  *)
    echo "[error] usage: $0 [vqa|gqa|vg|gqa-vg|all]" >&2
    exit 2
    ;;
esac
