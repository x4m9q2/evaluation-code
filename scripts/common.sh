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
export XVERIFY_ROOT="${CODE_ROOT}/x_verify"
export XVERIFY_MODEL="${XVERIFY_ROOT}/xVerify-0.5B-I"
export PRETRAIN_CHECKPOINT="${BUNDLE_ROOT}/checkpoints/gemma3_4b_pretrain_gate_projector_l1_sdpa"
export PRETRAIN_NOGATE_CHECKPOINT="${BUNDLE_ROOT}/checkpoints/gemma3_4b_pretrain_projector_sdpa"
export STAGE2_CHECKPOINT="${BUNDLE_ROOT}/checkpoints/gemma3_4b_stage2_gate_l1_mask_sdpa"
export STAGE2_NOGATE_CHECKPOINT="${BUNDLE_ROOT}/checkpoints/gemma3_4b_stage2_nogate_sdpa"

export IMAGE_DATA_ROOT="${BUNDLE_ROOT}/data/playground_data"
export PRETRAIN_DATA="${BUNDLE_ROOT}/data/pretrain_llava_v1_5_mix665k_single_noocr_max200_imageonly_strict_noocr.json"
export PRETRAIN_IMAGE_FOLDER="${IMAGE_DATA_ROOT}"
export STAGE2_DATA="${BUNDLE_ROOT}/data/stage2/train_raw_mixed_qwenratio_oldbase_sam3_plus_vqa.json"
export STAGE2_IMAGE_FOLDER="${IMAGE_DATA_ROOT}/coco/train2014"
export PATCH_MASK_ANALYSIS_PATH="${BUNDLE_ROOT}/data/stage2/patch_mask_analysis_train_raw_qwenkeep_sam3_compat.npz"
export TEST_RAW_WITH_SHORTCUT="${BUNDLE_ROOT}/data/eval/test_raw_with_shortcut_answer.json"
export NAPO_DATA="${BUNDLE_ROOT}/data/napo/train_raw_pos_neg_shortcut.json"

mkdir -p "${BUNDLE_ROOT}/logs" "${BUNDLE_ROOT}/outputs" "${BUNDLE_ROOT}/checkpoints"
