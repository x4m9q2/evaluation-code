#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_llava.sh"

require_cuda_visible_devices_count 4 "NaPO LLaVA training"

NAPO_LLAVA_ROOT="${BUNDLE_ROOT}/third_party/napo_llava_ref"
export PYTHONPATH="${NAPO_LLAVA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
NAPO_LOCAL_CLIP_LINK="${BUNDLE_ROOT}/third_party/clip-vit-large-patch14-336"
if [[ ! -e "${NAPO_LOCAL_CLIP_LINK}" ]]; then
  ln -s "${LLAVA_VISION_TOWER}" "${NAPO_LOCAL_CLIP_LINK}"
fi

cd "${NAPO_LLAVA_ROOT}"

DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${NAPO_LLAVA_ROOT}/script/zero2.json}"
RUN_NAME="${RUN_NAME:-napo_llava_shortcut}"
OUTPUT_DIR="${OUTPUT_DIR:-${NAPO_LLAVA_OUTPUT_ROOT}/${RUN_NAME}/checkpoints}"
LOGGING_DIR="${LOGGING_DIR:-${NAPO_LLAVA_OUTPUT_ROOT}/${RUN_NAME}/log}"
MASTER_PORT="${MASTER_PORT:-29500}"

check_path "${LLAVA_BASE_MODEL}" "LLaVA base model"
check_path "${LLAVA_VISION_TOWER}" "CLIP vision tower"
check_path "${NAPO_LLAVA_DATA_DIR}" "NaPO preference dataset directory"

run_or_echo "${PYTHON_BIN}" -m deepspeed.launcher.runner --master_port "${MASTER_PORT}" ./muffin/train/train_llava15.py \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --ddp_timeout 180000 \
  --model_name_or_path "${LLAVA_BASE_MODEL}" \
  --data_dir "${NAPO_LLAVA_DATA_DIR}" \
  --image_folder not_used \
  --vision_tower "${LLAVA_VISION_TOWER}" \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --fully_tune True \
  --image_aspect_ratio pad \
  --bf16 True \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs "${NUM_EPOCHS:-3}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-2}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE:-2}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-4}" \
  --eval_strategy no \
  --save_strategy epoch \
  --save_total_limit 3 \
  --data_source_names '' \
  --data_source_weights 1 \
  --learning_rate "${LEARNING_RATE:-5e-7}" \
  --weight_decay "${WEIGHT_DECAY:-0.01}" \
  --warmup_ratio "${WARMUP_RATIO:-0.1}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE:-cosine}" \
  --logging_steps "${LOGGING_STEPS:-10}" \
  --logging_dir "${LOGGING_DIR}" \
  --tf32 True \
  --model_max_length "${MODEL_MAX_LENGTH:-2048}" \
  --gradient_checkpointing True \
  --lazy_preprocess True \
  --task DPO \
  --report_to none \
  --run_name "${RUN_NAME}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-8}" \
  --dpo_use_average False \
  --dpo_token_weighted False \
  --dpo_token_weight 1.0 \
  --dpo_beta "${DPO_BETA:-0.1}" \
  ${EXTRA_ARGS:-}
