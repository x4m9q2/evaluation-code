#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_llava.sh"

cd "${LLAVA_CODE_ROOT}"

DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${LLAVA_CODE_ROOT}/scripts/zero1_bf16.json}"
REPORT_TO="${REPORT_TO:-none}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-32}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
MM_PROJECTOR_LR="${MM_PROJECTOR_LR:-2e-5}"
MASK_PATCH_LOSS_WEIGHT="${MASK_PATCH_LOSS_WEIGHT:-0}"
GATE_L1_LOSS_WEIGHT="${GATE_L1_LOSS_WEIGHT:-0}"
SAVE_STEPS="${SAVE_STEPS:-1716}"
EVALUATION_STRATEGY="${EVALUATION_STRATEGY:-epoch}"
LR_SCHEDULER_TOTAL_STEPS_SCALE="${LR_SCHEDULER_TOTAL_STEPS_SCALE:-1.5}"
LLAVA_STAGE2_NOGATE_CHECKPOINT="${LLAVA_STAGE2_NOGATE_CHECKPOINT:-${BUNDLE_ROOT}/checkpoints/llava_stage2_nogate}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

check_path "${LLAVA_BASE_MODEL}" "LLaVA base model"
check_path "${LLAVA_VISION_TOWER}" "CLIP vision tower"
check_path "${LLAVA_PRETRAIN_PROJECTOR}" "pretrained mm_projector.bin"
check_path "${LLAVA_STAGE2_DATA}" "stage-2 JSON/JSONL"
check_path "${LLAVA_STAGE2_EVAL_DATA}" "stage-2 validation JSON/JSONL"
check_path "${LLAVA_STAGE2_IMAGE_ROOT}" "stage-2 image root"

run_or_echo "${PYTHON_BIN}" -m deepspeed.launcher.runner llava/train/train_xformers.py \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --model_name_or_path "${LLAVA_BASE_MODEL}" \
  --version v1 \
  --data_path "${LLAVA_STAGE2_DATA}" \
  --eval_data_path "${LLAVA_STAGE2_EVAL_DATA}" \
  --image_folder "${LLAVA_STAGE2_IMAGE_ROOT}" \
  --vision_tower "${LLAVA_VISION_TOWER}" \
  --pretrain_mm_mlp_adapter "${LLAVA_PRETRAIN_PROJECTOR}" \
  --mm_projector_type mlp2x_gelu \
  --tune_mm_mlp_adapter False \
  --use_dual_input_gate False \
  --mm_projector_lr "${MM_PROJECTOR_LR}" \
  --mask_patch_loss_weight "${MASK_PATCH_LOSS_WEIGHT}" \
  --gate_l1_loss_weight "${GATE_L1_LOSS_WEIGHT}" \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length True \
  --bf16 True \
  --output_dir "${LLAVA_STAGE2_NOGATE_CHECKPOINT}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --evaluation_strategy "${EVALUATION_STRATEGY}" \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit 3 \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --lr_scheduler_total_steps_scale "${LR_SCHEDULER_TOTAL_STEPS_SCALE}" \
  --logging_steps 10 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-4}" \
  --lazy_preprocess True \
  --report_to "${REPORT_TO}" \
  ${EXTRA_ARGS}
