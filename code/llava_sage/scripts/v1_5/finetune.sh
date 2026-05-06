#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.1}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${REPO_ROOT}/checkpoints/assembled_llava_v15_from_mmproj_gate_20260311}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/dpo_train.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/root/train2014}"
VISION_TOWER="${VISION_TOWER:-${REPO_ROOT}/clip-vit-large-patch14-336}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${REPO_ROOT}/scripts/zero2_bf16.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/checkpoints/finetune_dpo_xformers_gate}"
REPORT_TO="${REPORT_TO:-none}"
TUNE_MM_MLP_ADAPTER="${TUNE_MM_MLP_ADAPTER:-False}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
MM_PROJECTOR_LR="${MM_PROJECTOR_LR:-2e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "Launching xformers finetune with:"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
echo "  DATA_PATH=${DATA_PATH}"
echo "  IMAGE_FOLDER=${IMAGE_FOLDER}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  TUNE_MM_MLP_ADAPTER=${TUNE_MM_MLP_ADAPTER}"

deepspeed llava/train/train_xformers.py \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --version v1 \
    --data_path "${DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --vision_tower "${VISION_TOWER}" \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter "${TUNE_MM_MLP_ADAPTER}" \
    --mm_projector_lr "${MM_PROJECTOR_LR}" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --learning_rate "${LEARNING_RATE}" \
    --weight_decay 0. \
    --warmup_ratio "${WARMUP_RATIO}" \
    --lr_scheduler_type "cosine" \
    --logging_steps "${LOGGING_STEPS}" \
    --tf32 True \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --gradient_checkpointing True \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --lazy_preprocess True \
    --report_to "${REPORT_TO}" \
    ${EXTRA_ARGS}
