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
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES
IFS=, read -r -a GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${NUM_GPUS:-${#GPU_LIST[@]}}"
MASTER_PORT="${MASTER_PORT:-29541}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${REPO_ROOT}/checkpoints/assembled_llava_v15_from_pretrain_meanreg_20260319_052641}"
PRETRAIN_MM_MLP_ADAPTER="${PRETRAIN_MM_MLP_ADAPTER:-}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/train_raw_mixed_masked_plus_removed_plus_vqa.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-data/images/coco/train2014}"
VISION_TOWER="${VISION_TOWER:-${REPO_ROOT}/clip-vit-large-patch14-336}"
PATCH_MASK_ANALYSIS_PATH="${PATCH_MASK_ANALYSIS_PATH:-${REPO_ROOT}/patch_mask_analysis_train_raw_qwenkeep_sam3_nonumbermask_compat.npz}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${REPO_ROOT}/scripts/zero1_bf16.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/checkpoints/finetune_stage2_mask_suppress_zero1_coco_direct_gatepatch_mixed_bs48_e3_stop1}"
REPORT_TO="${REPORT_TO:-none}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-32}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
MM_PROJECTOR_LR="${MM_PROJECTOR_LR:-}"
MASK_PATCH_LOSS_WEIGHT="${MASK_PATCH_LOSS_WEIGHT:-0.125}"
GATE_L1_LOSS_WEIGHT="${GATE_L1_LOSS_WEIGHT:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"
DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-True}"
MAX_STEPS="${MAX_STEPS:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "Launching mixed stage-2 xformers finetune with DeepSpeed ZeRO-1 + P2P:"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  NUM_GPUS=${NUM_GPUS}"
echo "  MASTER_PORT=${MASTER_PORT}"
echo "  MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
if [[ -n "${PRETRAIN_MM_MLP_ADAPTER}" ]]; then
  echo "  PRETRAIN_MM_MLP_ADAPTER=${PRETRAIN_MM_MLP_ADAPTER}"
else
  echo "  PRETRAIN_MM_MLP_ADAPTER=<disabled>"
fi
echo "  DATA_PATH=${DATA_PATH}"
echo "  IMAGE_FOLDER=${IMAGE_FOLDER}"
echo "  PATCH_MASK_ANALYSIS_PATH=${PATCH_MASK_ANALYSIS_PATH}"
echo "  DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}"
if [[ -n "${MAX_STEPS}" ]]; then
  echo "  MAX_STEPS=${MAX_STEPS}"
else
  echo "  MAX_STEPS=<disabled>"
fi
echo "  PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "  GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
if [[ -n "${MM_PROJECTOR_LR}" ]]; then
  echo "  MM_PROJECTOR_LR=${MM_PROJECTOR_LR}"
else
  echo "  MM_PROJECTOR_LR=<disabled>"
fi
echo "  MASK_PATCH_LOSS_WEIGHT=${MASK_PATCH_LOSS_WEIGHT}"
echo "  GATE_L1_LOSS_WEIGHT=${GATE_L1_LOSS_WEIGHT}"
echo "  DDP_FIND_UNUSED_PARAMETERS=${DDP_FIND_UNUSED_PARAMETERS}"
echo "  NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE}"
echo "  NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"

TRAIN_CMD=(
    deepspeed
    --master_port "${MASTER_PORT}"
    llava/train/train_xformers.py
    --deepspeed "${DEEPSPEED_CONFIG}"
    --model_name_or_path "${MODEL_NAME_OR_PATH}"
    --version v1
    --data_path "${DATA_PATH}"
    --image_folder "${IMAGE_FOLDER}"
    --patch_mask_analysis_path "${PATCH_MASK_ANALYSIS_PATH}"
    --vision_tower "${VISION_TOWER}"
    --mm_projector_type mlp2x_gelu
    --tune_mm_mlp_adapter False
    --mask_patch_loss_weight "${MASK_PATCH_LOSS_WEIGHT}"
    --gate_l1_loss_weight "${GATE_L1_LOSS_WEIGHT}"
    --mm_vision_select_layer -2
    --mm_use_im_start_end False
    --mm_use_im_patch_token False
    --image_aspect_ratio pad
    --group_by_modality_length True
    --bf16 True
    --output_dir "${OUTPUT_DIR}"
    --num_train_epochs "${NUM_TRAIN_EPOCHS}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --evaluation_strategy "no"
    --save_strategy "epoch"
    --save_total_limit "${SAVE_TOTAL_LIMIT}"
    --learning_rate "${LEARNING_RATE}"
    --weight_decay 0.
    --warmup_ratio "${WARMUP_RATIO}"
    --lr_scheduler_type "cosine"
    --logging_steps "${LOGGING_STEPS}"
    --tf32 True
    --model_max_length "${MODEL_MAX_LENGTH}"
    --gradient_checkpointing True
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
    --lazy_preprocess True
    --report_to "${REPORT_TO}"
    --ddp_find_unused_parameters "${DDP_FIND_UNUSED_PARAMETERS}"
)

if [[ -n "${MAX_STEPS}" ]]; then
  TRAIN_CMD+=(--max_steps "${MAX_STEPS}")
fi

if [[ -n "${PRETRAIN_MM_MLP_ADAPTER}" ]]; then
  TRAIN_CMD+=(--pretrain_mm_mlp_adapter "${PRETRAIN_MM_MLP_ADAPTER}")
fi

if [[ -n "${MM_PROJECTOR_LR}" ]]; then
  TRAIN_CMD+=(--mm_projector_lr "${MM_PROJECTOR_LR}")
fi

if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS_ARR=(${EXTRA_ARGS})
  TRAIN_CMD+=("${EXTRA_ARGS_ARR[@]}")
fi

"${TRAIN_CMD[@]}"
