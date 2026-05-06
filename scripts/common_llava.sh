#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BUNDLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export LLAVA_CODE_ROOT="${BUNDLE_ROOT}/code/llava_sage"
export SAM3_CODE_ROOT="${BUNDLE_ROOT}/code/sam3"
export LLAVA_REPO_ROOT="${LLAVA_CODE_ROOT}"
export LLAVA_PYTHONPATH="${LLAVA_CODE_ROOT}:${LLAVA_CODE_ROOT}/llava:${SAM3_CODE_ROOT}:${BUNDLE_ROOT}/code/data_tools:${BUNDLE_ROOT}/code/evaluation"
export PYTHONPATH="${LLAVA_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}"

export PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

export LLAVA_BASE_MODEL="${LLAVA_BASE_MODEL:-${BUNDLE_ROOT}/models/llava-v1.5-7b}"
export LLAVA_VISION_TOWER="${LLAVA_VISION_TOWER:-${BUNDLE_ROOT}/models/clip-vit-large-patch14-336}"
export XVERIFY_MODEL="${XVERIFY_MODEL:-${BUNDLE_ROOT}/models/xVerify-0.5B-I}"
export XVERIFY_ROOT="${XVERIFY_ROOT:-${BUNDLE_ROOT}/code/evaluation/x_verify}"
export SAM3_CHECKPOINT="${SAM3_CHECKPOINT:-${BUNDLE_ROOT}/models/sam3_ckpt/sam3.pt}"

export LLAVA_PRETRAIN_DATA="${LLAVA_PRETRAIN_DATA:-${BUNDLE_ROOT}/data/pretrain_llava_v1_5_mix665k_single_noocr_max200_imageonly_strict_noocr.json}"
export LLAVA_PRETRAIN_IMAGE_ROOT="${LLAVA_PRETRAIN_IMAGE_ROOT:-${BUNDLE_ROOT}/data/images}"
export SAGE_AS_ROOT="${SAGE_AS_ROOT:-${BUNDLE_ROOT}/data/sage_as}"
export SAGE_AS_DATASET="${SAGE_AS_DATASET:-vqa}"
case "${SAGE_AS_DATASET}" in
  vqa)
    SAGE_AS_TRAIN_REL="data/vqa/train.json"
    SAGE_AS_VAL_REL="data/vqa/val.json"
    SAGE_AS_MASK_REL="masks/vqa_masks.npz"
    ;;
  gqa)
    SAGE_AS_TRAIN_REL="data/gqa/train.jsonl"
    SAGE_AS_VAL_REL="data/gqa/val.jsonl"
    SAGE_AS_MASK_REL="masks/gqa_masks.npz"
    ;;
  vg)
    SAGE_AS_TRAIN_REL="data/vg/train.jsonl"
    SAGE_AS_VAL_REL="data/vg/val.jsonl"
    SAGE_AS_MASK_REL="masks/vg_masks.npz"
    ;;
  *)
    echo "[error] SAGE_AS_DATASET must be one of: vqa, gqa, vg. Got: ${SAGE_AS_DATASET}" >&2
    exit 2
    ;;
esac
export LLAVA_STAGE2_DATA="${LLAVA_STAGE2_DATA:-${SAGE_AS_ROOT}/${SAGE_AS_TRAIN_REL}}"
export LLAVA_STAGE2_EVAL_DATA="${LLAVA_STAGE2_EVAL_DATA:-${SAGE_AS_ROOT}/${SAGE_AS_VAL_REL}}"
export LLAVA_STAGE2_IMAGE_ROOT="${LLAVA_STAGE2_IMAGE_ROOT:-${BUNDLE_ROOT}/data/images}"
export LLAVA_PATCH_MASK_NPZ="${LLAVA_PATCH_MASK_NPZ:-${SAGE_AS_ROOT}/${SAGE_AS_MASK_REL}}"
export LLAVA_PRETRAIN_PROJECTOR="${LLAVA_PRETRAIN_PROJECTOR:-${BUNDLE_ROOT}/checkpoints/llava_pretrain_gate/mm_projector.bin}"
export LLAVA_STAGE2_CHECKPOINT="${LLAVA_STAGE2_CHECKPOINT:-${BUNDLE_ROOT}/checkpoints/llava_stage2_sage}"
export LLAVA_PRETRAIN_OUTPUT="${LLAVA_PRETRAIN_OUTPUT:-${BUNDLE_ROOT}/checkpoints/llava_pretrain_gate}"
export VQA_TRAIN2014_IMAGE_ROOT="${VQA_TRAIN2014_IMAGE_ROOT:-${BUNDLE_ROOT}/data/images/coco/train2014}"
export VQA_VAL2014_IMAGE_ROOT="${VQA_VAL2014_IMAGE_ROOT:-${BUNDLE_ROOT}/data/images/coco/val2014}"

export TEST_RAW_WITH_SHORTCUT="${TEST_RAW_WITH_SHORTCUT:-${BUNDLE_ROOT}/data/eval/test_raw_with_shortcut_answer.json}"
export TEST_IMAGE_ROOT="${TEST_IMAGE_ROOT:-${BUNDLE_ROOT}/data/images/coco/train2014}"

export POPE_QUESTION_FILE="${POPE_QUESTION_FILE:-${BUNDLE_ROOT}/data/pope/llava_pope_test.jsonl}"
export POPE_ANNOTATION_DIR="${POPE_ANNOTATION_DIR:-${BUNDLE_ROOT}/data/pope/coco}"
export POPE_IMAGE_ROOT="${POPE_IMAGE_ROOT:-${BUNDLE_ROOT}/data/pope/val2014}"
export BEAF_QNA_PATH="${BEAF_QNA_PATH:-${BUNDLE_ROOT}/data/beaf/beaf_qna.json}"
export BEAF_IMAGE_ROOT="${BEAF_IMAGE_ROOT:-${BUNDLE_ROOT}/data/beaf/images}"

export NAPO_LLAVA_DATA_DIR="${NAPO_LLAVA_DATA_DIR:-${BUNDLE_ROOT}/data/napo_llava/train_raw_pos_neg_shortcut_hf}"
export NAPO_LLAVA_OUTPUT_ROOT="${NAPO_LLAVA_OUTPUT_ROOT:-${BUNDLE_ROOT}/checkpoints/napo_llava}"

export OUTPUT_ROOT="${OUTPUT_ROOT:-${BUNDLE_ROOT}/outputs}"
export LOG_ROOT="${LOG_ROOT:-${BUNDLE_ROOT}/logs}"

mkdir -p "${BUNDLE_ROOT}/checkpoints" "${OUTPUT_ROOT}" "${LOG_ROOT}"

has_glob_matches() {
  compgen -G "$1" >/dev/null
}

if [[ -z "${LLAVA_EVAL_MODEL:-}" ]]; then
  if [[ -d "${BUNDLE_ROOT}/checkpoints/llava_stage2_sage" ]]; then
    export LLAVA_EVAL_MODEL="${BUNDLE_ROOT}/checkpoints/llava_stage2_sage"
  elif [[ -d "${BUNDLE_ROOT}/checkpoints/llava_pretrain_gate_smoke50_assembled" ]]; then
    export LLAVA_EVAL_MODEL="${BUNDLE_ROOT}/checkpoints/llava_pretrain_gate_smoke50_assembled"
  else
    export LLAVA_EVAL_MODEL="${LLAVA_BASE_MODEL}"
  fi
fi

run_or_echo() {
  if [[ "${DRY_RUN:-1}" == "0" || "${RUN:-0}" == "1" ]]; then
    echo "[run] $*"
    "$@"
  else
    echo "[dry-run] $*"
  fi
}

check_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "${path}" ]]; then
    echo "[missing] ${label}: ${path}" >&2
    if [[ "${DRY_RUN:-1}" == "0" || "${RUN:-0}" == "1" ]]; then
      return 1
    fi
    return 0
  fi
  echo "[ok] ${label}: ${path}"
}

require_cuda_visible_devices_count() {
  local expected_count="$1"
  local label="${2:-CUDA_VISIBLE_DEVICES}"
  local devices="${CUDA_VISIBLE_DEVICES}"
  local -a gpu_list=()

  IFS=',' read -r -a gpu_list <<< "${devices}"
  if [[ "${#gpu_list[@]}" -ne "${expected_count}" ]]; then
    echo "[error] ${label} requires exactly ${expected_count} GPUs, got ${#gpu_list[@]} from CUDA_VISIBLE_DEVICES=${devices}" >&2
    return 2
  fi

  echo "[ok] ${label}: using ${#gpu_list[@]} GPUs from CUDA_VISIBLE_DEVICES=${devices}"
}
