#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

RUN_NAME="${RUN_NAME:-gemma3_4b_stage2_gate_l1_mask_sdpa}"
OUTPUT_DIR="${OUTPUT_DIR:-${STAGE2_CHECKPOINT}}"
LOG_FILE="${LOG_FILE:-${BUNDLE_ROOT}/logs/${RUN_NAME}.log}"
NUM_GPUS="${NUM_GPUS:-4}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
REPORT_TO="${REPORT_TO:-none}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
DISABLE_FLASH_ATTN2="${DISABLE_FLASH_ATTN2:-True}"
MASK_PATCH_LOSS_WEIGHT="${MASK_PATCH_LOSS_WEIGHT:-0.041666666666666664}"
MAX_STEPS="${MAX_STEPS:--1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

unset NCCL_P2P_DISABLE

mkdir -p "${OUTPUT_DIR}" "$(dirname "${LOG_FILE}")"

check_path "${PRETRAIN_CHECKPOINT}" "Gemma pretrain checkpoint"
check_path "${GATE_TEXT_MODEL_ID}" "gate text encoder"
check_path "${STAGE2_DATA}" "stage-2 JSON"
check_path "${STAGE2_IMAGE_FOLDER}" "stage-2 image root"
check_path "${PATCH_MASK_ANALYSIS_PATH}" "patch mask NPZ"

{
  echo "[run] ${RUN_NAME}"
  echo "[model] ${PRETRAIN_CHECKPOINT}"
  echo "[gate_text] ${GATE_TEXT_MODEL_ID}"
  echo "[data] ${STAGE2_DATA}"
  echo "[image_folder] ${STAGE2_IMAGE_FOLDER}"
  echo "[patch_mask] ${PATCH_MASK_ANALYSIS_PATH}"
  echo "[output] ${OUTPUT_DIR}"
  echo "[attn] ${ATTN_IMPLEMENTATION}"
  echo "[launcher] torch.distributed.run nproc_per_node=${NUM_GPUS}"
  echo "[batch] per_device=${PER_DEVICE_TRAIN_BATCH_SIZE} grad_accum=${GRADIENT_ACCUMULATION_STEPS}"
  echo "[loss] gate_l1=${GATE_L1_LOSS_WEIGHT:-0.01} mask_patch=${MASK_PATCH_LOSS_WEIGHT}"
  date
} | tee "${LOG_FILE}"

cd "${GEMMA_DIR}"
run_or_echo "${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${NUM_GPUS}" \
  src/train/train_sft.py \
  --model_id "${PRETRAIN_CHECKPOINT}" \
  --data_path "${STAGE2_DATA}" \
  --image_folder "${STAGE2_IMAGE_FOLDER}" \
  --patch_mask_analysis_path "${PATCH_MASK_ANALYSIS_PATH}" \
  --disable_number_mask_loss True \
  --use_dual_input_gate True \
  --gate_text_model_id "${GATE_TEXT_MODEL_ID}" \
  --freeze_gate_text_encoder True \
  --gate_l1_loss_weight "${GATE_L1_LOSS_WEIGHT:-0.01}" \
  --mask_patch_loss_weight "${MASK_PATCH_LOSS_WEIGHT}" \
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
  --gate_lr "${GATE_LR:-2e-5}" \
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
