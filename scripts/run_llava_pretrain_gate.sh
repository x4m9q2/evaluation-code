#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_llava.sh"

cd "${LLAVA_CODE_ROOT}"

DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${LLAVA_CODE_ROOT}/scripts/zero2_bf16.json}"
REPORT_TO="${REPORT_TO:-none}"
SAVE_STEPS="${SAVE_STEPS:-24000}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-32}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

check_path "${LLAVA_BASE_MODEL}" "LLaVA base model"
check_path "${LLAVA_VISION_TOWER}" "CLIP vision tower"
check_path "${LLAVA_PRETRAIN_DATA}" "pretraining JSON"
check_path "${LLAVA_PRETRAIN_IMAGE_ROOT}" "pretraining image root"

run_or_echo "${PYTHON_BIN}" -m deepspeed.launcher.runner llava/train/train_xformers.py \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --model_name_or_path "${LLAVA_BASE_MODEL}" \
  --version v1 \
  --data_path "${LLAVA_PRETRAIN_DATA}" \
  --image_folder "${LLAVA_PRETRAIN_IMAGE_ROOT}" \
  --vision_tower "${LLAVA_VISION_TOWER}" \
  --mm_projector_type mlp2x_gelu \
  --tune_mm_mlp_adapter True \
  --use_dual_input_gate True \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --bf16 True \
  --output_dir "${LLAVA_PRETRAIN_OUTPUT}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --evaluation_strategy no \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit 1 \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-4}" \
  --lazy_preprocess True \
  --report_to "${REPORT_TO}" \
  ${EXTRA_ARGS}
