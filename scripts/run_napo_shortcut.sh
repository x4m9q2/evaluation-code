#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

RUN_NAME="${RUN_NAME:-gemma3_4b_napo_shortcut_sdpa}"
OUTPUT_DIR="${OUTPUT_DIR:-${BUNDLE_ROOT}/checkpoints/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-${BUNDLE_ROOT}/logs/${RUN_NAME}.log}"
NUM_GPUS="${NUM_GPUS:-4}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
REPORT_TO="${REPORT_TO:-none}"
MAX_STEPS="${MAX_STEPS:--1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
NAPO_IMAGE_FOLDER="${NAPO_IMAGE_FOLDER:-${BUNDLE_ROOT}/data/images/coco/train2014}"

mkdir -p "${OUTPUT_DIR}" "$(dirname "${LOG_FILE}")"

check_path "${BASE_MODEL_ID}" "Gemma base model"
check_path "${NAPO_DATA}" "Gemma NaPO preference JSON"
check_path "${NAPO_IMAGE_FOLDER}" "NaPO image folder"

{
  echo "[run] ${RUN_NAME}"
  echo "[model] ${BASE_MODEL_ID}"
  echo "[data] ${NAPO_DATA}"
  echo "[image_folder] ${NAPO_IMAGE_FOLDER}"
  echo "[output] ${OUTPUT_DIR}"
  echo "[loss] type=${NAPO_LOSS_TYPE:-dyn_lq} beta=${BETA:-0.1} alpha=${NAPO_ALPHA:-0.5} dyn_q_avg=${NAPO_DYN_Q_USE_AVERAGE:-True}"
  echo "[gate] disabled by default"
  echo "[batch] ${NUM_GPUS} x ${PER_DEVICE_TRAIN_BATCH_SIZE} x ${GRADIENT_ACCUMULATION_STEPS}"
  date
} | tee "${LOG_FILE}"

cd "${GEMMA_DIR}"
"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${NUM_GPUS}" \
  src/train/train_dpo.py \
  --model_id "${BASE_MODEL_ID}" \
  --data_path "${NAPO_DATA}" \
  --image_folder "${NAPO_IMAGE_FOLDER}" \
  --use_dual_input_gate "${USE_DUAL_INPUT_GATE:-False}" \
  --napo_loss_type "${NAPO_LOSS_TYPE:-dyn_lq}" \
  --napo_alpha "${NAPO_ALPHA:-0.5}" \
  --napo_q "${NAPO_Q:-1.0}" \
  --napo_dyn_q_use_average "${NAPO_DYN_Q_USE_AVERAGE:-True}" \
  --disable_token_type_ids "${DISABLE_TOKEN_TYPE_IDS:-False}" \
  --disable_ref_model "${DISABLE_REF_MODEL:-False}" \
  --dpo_loss sigmoid \
  --precompute_ref_log_probs False \
  --beta "${BETA:-0.1}" \
  --use_liger False \
  --disable_flash_attn2 True \
  --attn_implementation "${ATTN_IMPLEMENTATION:-sdpa}" \
  --vision_attn_implementation "${VISION_ATTN_IMPLEMENTATION:-sdpa}" \
  --lora_enable False \
  --freeze_projector False \
  --freeze_vision_tower True \
  --freeze_llm False \
  --bf16 True \
  --output_dir "${OUTPUT_DIR}" \
  --remove_unused_columns False \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate "${LEARNING_RATE:-5e-7}" \
  --projector_lr "${PROJECTOR_LR:-5e-7}" \
  --vision_lr "${VISION_LR:-2e-7}" \
  --weight_decay "${WEIGHT_DECAY:-0.01}" \
  --warmup_ratio "${WARMUP_RATIO:-0.1}" \
  --adam_beta2 0.95 \
  --lr_scheduler_type cosine \
  --logging_steps "${LOGGING_STEPS:-1}" \
  --tf32 True \
  --gradient_checkpointing True \
  --report_to "${REPORT_TO}" \
  --lazy_preprocess True \
  --save_strategy epoch \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-4}" \
  ${EXTRA_ARGS} \
  "$@" 2>&1 | tee -a "${LOG_FILE}"
