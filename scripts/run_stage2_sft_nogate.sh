#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

RUN_NAME="${RUN_NAME:-gemma3_4b_stage2_nogate_sdpa}"
MODEL_ID="${MODEL_ID:-${PRETRAIN_NOGATE_CHECKPOINT}}"
OUTPUT_DIR="${OUTPUT_DIR:-${STAGE2_NOGATE_CHECKPOINT}}"
LOG_FILE="${LOG_FILE:-${BUNDLE_ROOT}/logs/${RUN_NAME}.log}"
NUM_GPUS="${NUM_GPUS:-4}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
REPORT_TO="${REPORT_TO:-none}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
DISABLE_FLASH_ATTN2="${DISABLE_FLASH_ATTN2:-True}"
MAX_STEPS="${MAX_STEPS:--1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

unset NCCL_P2P_DISABLE

mkdir -p "${OUTPUT_DIR}" "$(dirname "${LOG_FILE}")"

{
  echo "[run] ${RUN_NAME}"
  echo "[model] ${MODEL_ID}"
  echo "[data] ${STAGE2_DATA}"
  echo "[image_folder] ${STAGE2_IMAGE_FOLDER}"
  echo "[output] ${OUTPUT_DIR}"
  echo "[attn] ${ATTN_IMPLEMENTATION}"
  echo "[gate] disabled"
  echo "[launcher] torch.distributed.run nproc_per_node=${NUM_GPUS}"
  echo "[batch] per_device=${PER_DEVICE_TRAIN_BATCH_SIZE} grad_accum=${GRADIENT_ACCUMULATION_STEPS}"
  date
} | tee "${LOG_FILE}"

cd "${GEMMA_DIR}"
"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${NUM_GPUS}" \
  src/train/train_sft.py \
  --model_id "${MODEL_ID}" \
  --data_path "${STAGE2_DATA}" \
  --image_folder "${STAGE2_IMAGE_FOLDER}" \
  --use_dual_input_gate False \
  --gate_l1_loss_weight 0.0 \
  --mask_patch_loss_weight 0.0 \
  --use_liger False \
  --disable_flash_attn2 "${DISABLE_FLASH_ATTN2}" \
  --attn_implementation "${ATTN_IMPLEMENTATION}" \
  --lora_enable False \
  --freeze_projector False \
  --freeze_vision_tower "${FREEZE_VISION_TOWER:-True}" \
  --freeze_llm False \
  --bf16 True \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${RUN_NAME}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate "${LEARNING_RATE:-2e-5}" \
  --projector_lr "${PROJECTOR_LR:-2e-5}" \
  --vision_lr "${VISION_LR:-2e-6}" \
  --weight_decay "${WEIGHT_DECAY:-0.0}" \
  --warmup_ratio "${WARMUP_RATIO:-0.03}" \
  --adam_beta2 0.95 \
  --lr_scheduler_type cosine \
  --logging_steps "${LOGGING_STEPS:-10}" \
  --tf32 True \
  --gradient_checkpointing True \
  --report_to "${REPORT_TO}" \
  --lazy_preprocess True \
  --save_strategy epoch \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-4}" \
  ${EXTRA_ARGS} 2>&1 | tee -a "${LOG_FILE}"
