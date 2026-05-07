#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BUNDLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export CODE_ROOT="${BUNDLE_ROOT}/code/gemma_gate"
export GEMMA_DIR="${CODE_ROOT}/gemma"

export PYTHONPATH="${CODE_ROOT}:${GEMMA_DIR}:${GEMMA_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

export PYTHON_BIN="${PYTHON_BIN:-python}"

export BASE_MODEL_ID="${BUNDLE_ROOT}/models/Gemma-3-4B-IT"
export GATE_TEXT_MODEL_ID="${BUNDLE_ROOT}/models/siglip-so400m-patch14-384"
export PRETRAIN_CHECKPOINT="${BUNDLE_ROOT}/checkpoints/gemma3_4b_pretrain_gate_projector_l1_sdpa"
export PRETRAIN_NOGATE_CHECKPOINT="${BUNDLE_ROOT}/checkpoints/gemma3_4b_pretrain_projector_sdpa"
export STAGE2_CHECKPOINT="${BUNDLE_ROOT}/checkpoints/gemma3_4b_stage2_gate_l1_mask_sdpa"
export STAGE2_NOGATE_CHECKPOINT="${BUNDLE_ROOT}/checkpoints/gemma3_4b_stage2_nogate_sdpa"

export IMAGE_DATA_ROOT="${BUNDLE_ROOT}/data/playground_data"
export PRETRAIN_DATA="${BUNDLE_ROOT}/data/pretrain_llava_v1_5_mix665k_single_noocr_max200_imageonly_strict_noocr.json"
export PRETRAIN_IMAGE_FOLDER="${IMAGE_DATA_ROOT}"
export SAGE_AS_ROOT="${SAGE_AS_ROOT:-${BUNDLE_ROOT}/data/sage_as}"
export SAGE_AS_DATASET="${SAGE_AS_DATASET:-vqa}"
case "${SAGE_AS_DATASET}" in
  vqa|vqa_v2_cmsv|vqa-v2-cmsv)
    SAGE_AS_TRAIN_REL="data/vqa_v2_cmsv/train.json"
    SAGE_AS_VAL_REL="data/vqa_v2_cmsv/val.json"
    SAGE_AS_TEST_REL="data/vqa_v2_cmsv/test.json"
    SAGE_AS_MASK_REL="masks/vqa_v2_cmsv_masks.npz"
    ;;
  gqa|gqa_cmsv|gqa-cmsv)
    SAGE_AS_TRAIN_REL="data/gqa_cmsv/train.jsonl"
    SAGE_AS_VAL_REL="data/gqa_cmsv/val.jsonl"
    SAGE_AS_TEST_REL="data/gqa_cmsv/test.jsonl"
    SAGE_AS_MASK_REL="masks/gqa_cmsv_masks.npz"
    ;;
  vg|vg_cmsv|vg-cmsv)
    SAGE_AS_TRAIN_REL="data/vg_cmsv/train.jsonl"
    SAGE_AS_VAL_REL="data/vg_cmsv/val.jsonl"
    SAGE_AS_TEST_REL="data/vg_cmsv/test.jsonl"
    SAGE_AS_MASK_REL="masks/vg_cmsv_masks.npz"
    ;;
  *)
    echo "[error] SAGE_AS_DATASET must be one of: vqa, gqa, vg, vqa_v2_cmsv, gqa_cmsv, vg_cmsv. Got: ${SAGE_AS_DATASET}" >&2
    exit 2
    ;;
esac
export STAGE2_DATA="${STAGE2_DATA:-${SAGE_AS_ROOT}/${SAGE_AS_TRAIN_REL}}"
export STAGE2_EVAL_DATA="${STAGE2_EVAL_DATA:-${SAGE_AS_ROOT}/${SAGE_AS_VAL_REL}}"
export STAGE2_IMAGE_FOLDER="${STAGE2_IMAGE_FOLDER:-${BUNDLE_ROOT}/data/images}"
export PATCH_MASK_ANALYSIS_PATH="${PATCH_MASK_ANALYSIS_PATH:-${SAGE_AS_ROOT}/${SAGE_AS_MASK_REL}}"
export TEST_RAW_WITH_SHORTCUT="${TEST_RAW_WITH_SHORTCUT:-${SAGE_AS_ROOT}/${SAGE_AS_TEST_REL}}"
export NAPO_DATA="${NAPO_DATA:-${BUNDLE_ROOT}/data/napo/train_raw_pos_neg_shortcut.json}"

mkdir -p "${BUNDLE_ROOT}/logs" "${BUNDLE_ROOT}/outputs" "${BUNDLE_ROOT}/checkpoints"

run_or_echo() {
  echo "[run] $*"
  "$@"
}

check_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "${path}" ]]; then
    echo "[missing] ${label}: ${path}" >&2
    return 1
  fi
  echo "[ok] ${label}: ${path}"
}
