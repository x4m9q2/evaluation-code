#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CODE_ROOT="${BUNDLE_ROOT}/code/gemma_gate"
GEMMA_DIR="${CODE_ROOT}/gemma"
PYTHONPATH="${CODE_ROOT}:${GEMMA_DIR}:${GEMMA_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONPATH

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

MODEL_ID="${BUNDLE_ROOT}/models/Gemma-3-4B-IT"
GATE_TEXT_MODEL_ID="${BUNDLE_ROOT}/models/siglip-so400m-patch14-384"
DATA_PATH="${BUNDLE_ROOT}/data/pretrain_llava_v1_5_mix665k_single_noocr_max200_imageonly_strict_noocr.json"
IMAGE_FOLDER="${BUNDLE_ROOT}/data/playground_data"
DEEPSPEED_CONFIG="${CODE_ROOT}/scripts/zero2_bf16.json"
OUTPUT_DIR="${BUNDLE_ROOT}/checkpoints/gemma3_4b_pretrain_gate_projector_l1p1_zero2_bs16_ga2_flashattn_save2500"
LOG_FILE="${BUNDLE_ROOT}/logs/$(basename "${OUTPUT_DIR}").log"
RUN_NAME="${RUN_NAME:-$(basename "${OUTPUT_DIR}")}"
PYTHON_BIN="${PYTHON_BIN:-python}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
DISABLE_FLASH_ATTN2="${DISABLE_FLASH_ATTN2:-False}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-auto}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
SAVE_STEPS="${SAVE_STEPS:-2500}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
REPORT_TO="${REPORT_TO:-none}"
MAX_STEPS="${MAX_STEPS:--1}"

mkdir -p "$(dirname "${LOG_FILE}")" "${OUTPUT_DIR}"

cd "${GEMMA_DIR}"
"${PYTHON_BIN}" -m deepspeed.launcher.runner --include localhost:${CUDA_VISIBLE_DEVICES} src/train/train_sft.py \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --model_id "${MODEL_ID}" \
  --data_path "${DATA_PATH}" \
  --image_folder "${IMAGE_FOLDER}" \
  --use_dual_input_gate True \
  --gate_text_model_id "${GATE_TEXT_MODEL_ID}" \
  --freeze_gate_text_encoder True \
  --gate_l1_loss_weight 0.1 \
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
  --learning_rate 1e-3 \
  --projector_lr 1e-3 \
  --gate_lr 1e-3 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --adam_beta2 0.95 \
  --lr_scheduler_type cosine \
  --logging_steps "${LOGGING_STEPS}" \
  --tf32 True \
  --max_seq_length 2048 \
  --gradient_checkpointing True \
  --report_to "${REPORT_TO}" \
  --lazy_preprocess True \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --dataloader_num_workers 4 2>&1 | tee "${LOG_FILE}"
