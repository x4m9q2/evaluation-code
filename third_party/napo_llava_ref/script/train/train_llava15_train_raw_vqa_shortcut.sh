#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-train_raw_vqa_shortcut_dpo_${TIMESTAMP}}"

MODEL_PATH="${MODEL_PATH:-models/llava-v1.5-7b}"
VISION_TOWER="${VISION_TOWER:-models/clip-vit-large-patch14-336}"
DATA_DIR="${DATA_DIR:-third_party/napo_llava_ref/datasets/train_raw_vqa_shortcut_hf}"
OUTPUT_ROOT="${OUTPUT_ROOT:-third_party/napo_llava_ref/.ckpt}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}/checkpoints}"
LOGGING_DIR="${LOGGING_DIR:-${OUTPUT_ROOT}/${RUN_NAME}/log}"

NUM_EPOCHS="${NUM_EPOCHS:-4}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-5e-7}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
DPO_BETA="${DPO_BETA:-0.1}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-./script/zero2.json}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "RUN_NAME=${RUN_NAME}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "DATA_DIR=${DATA_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

deepspeed ./muffin/train/train_llava15.py \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --ddp_timeout 180000 \
    --model_name_or_path "${MODEL_PATH}" \
    --data_dir "${DATA_DIR}" \
    --image_folder not_used \
    --vision_tower "${VISION_TOWER}" \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --fully_tune True \
    --image_aspect_ratio pad \
    --bf16 True \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --evaluation_strategy no \
    --save_strategy epoch \
    --save_total_limit 2 \
    --data_source_names '' \
    --data_source_weights 1 \
    --learning_rate "${LEARNING_RATE}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
    --logging_steps "${LOGGING_STEPS}" \
    --logging_dir "${LOGGING_DIR}" \
    --tf32 True \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --task DPO \
    --report_to none \
    --run_name "${RUN_NAME}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --dpo_use_average False \
    --dpo_token_weighted False \
    --dpo_token_weight 1.0 \
    --dpo_beta "${DPO_BETA}" \
    ${EXTRA_ARGS}
