#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

RUN_NAME="${RUN_NAME:-gemma3_4b_pretrain_projector_sdpa}"
OUTPUT_DIR="${OUTPUT_DIR:-${PRETRAIN_NOGATE_CHECKPOINT}}"
LOG_FILE="${LOG_FILE:-${BUNDLE_ROOT}/logs/${RUN_NAME}.log}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${CODE_ROOT}/scripts/zero2_bf16.json}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
SAVE_STEPS="${SAVE_STEPS:-2500}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
REPORT_TO="${REPORT_TO:-none}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
DISABLE_FLASH_ATTN2="${DISABLE_FLASH_ATTN2:-True}"
MAX_STEPS="${MAX_STEPS:--1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${OUTPUT_DIR}" "$(dirname "${LOG_FILE}")"

check_path "${BASE_MODEL_ID}" "Gemma base model"
check_path "${DEEPSPEED_CONFIG}" "DeepSpeed config"
check_path "${PRETRAIN_DATA}" "pretraining JSON"
check_path "${PRETRAIN_IMAGE_FOLDER}" "pretraining image folder"

{
  echo "[run] ${RUN_NAME}"
  echo "[model] ${BASE_MODEL_ID}"
  echo "[data] ${PRETRAIN_DATA}"
  echo "[image_folder] ${PRETRAIN_IMAGE_FOLDER}"
  echo "[output] ${OUTPUT_DIR}"
  echo "[attn] ${ATTN_IMPLEMENTATION}"
  echo "[gate] disabled"
  echo "[batch] per_device=${PER_DEVICE_TRAIN_BATCH_SIZE} grad_accum=${GRADIENT_ACCUMULATION_STEPS}"
  date
} | tee "${LOG_FILE}"

cd "${GEMMA_DIR}"
"${PYTHON_BIN}" -m deepspeed.launcher.runner --include "localhost:${CUDA_VISIBLE_DEVICES}" src/train/train_sft.py \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --model_id "${BASE_MODEL_ID}" \
  --data_path "${PRETRAIN_DATA}" \
  --image_folder "${PRETRAIN_IMAGE_FOLDER}" \
  --use_dual_input_gate False \
  --gate_l1_loss_weight 0.0 \
  --mask_patch_loss_weight 0.0 \
  --use_liger False \
  --disable_flash_attn2 "${DISABLE_FLASH_ATTN2}" \
  --attn_implementation "${ATTN_IMPLEMENTATION}" \
  --lora_enable False \
  --freeze_projector False \
  --freeze_vision_tower True \
  --freeze_llm True \
  --bf16 True \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${RUN_NAME}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate "${LEARNING_RATE:-1e-3}" \
  --projector_lr "${PROJECTOR_LR:-1e-3}" \
  --weight_decay "${WEIGHT_DECAY:-0.0}" \
  --warmup_ratio "${WARMUP_RATIO:-0.03}" \
  --adam_beta2 0.95 \
  --lr_scheduler_type cosine \
  --logging_steps "${LOGGING_STEPS}" \
  --tf32 True \
  --max_seq_length "${MAX_SEQ_LENGTH:-2048}" \
  --gradient_checkpointing True \
  --report_to "${REPORT_TO}" \
  --lazy_preprocess True \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-4}" \
  ${EXTRA_ARGS} 2>&1 | tee -a "${LOG_FILE}"
