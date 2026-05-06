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

MODEL_ID="${BUNDLE_ROOT}/checkpoints/gemma3_4b_pretrain_gate_projector_l1_sdpa"
GATE_TEXT_MODEL_ID="${BUNDLE_ROOT}/models/siglip-so400m-patch14-384"
DATA_PATH="${BUNDLE_ROOT}/data/stage2/train_raw_mixed_qwenratio_oldbase_sam3_plus_vqa.json"
IMAGE_FOLDER="${BUNDLE_ROOT}/data/playground_data/coco/train2014"
PATCH_MASK_ANALYSIS_PATH="${BUNDLE_ROOT}/data/stage2/patch_mask_analysis_train_raw_qwenkeep_sam3_compat.npz"
DEEPSPEED_CONFIG="${CODE_ROOT}/scripts/zero1_bf16.json"
OUTPUT_DIR="${BUNDLE_ROOT}/checkpoints/gemma3_4b_qwenratio_sam3_gate_l1_mask_nonumber_zero1_bs8_ga4"
LOG_FILE="${BUNDLE_ROOT}/logs/$(basename "${OUTPUT_DIR}").log"
PYTHON_BIN="${PYTHON_BIN:-python}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
FREEZE_VISION_TOWER="${FREEZE_VISION_TOWER:-True}"
DISABLE_FLASH_ATTN2="${DISABLE_FLASH_ATTN2:-True}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-auto}"

mkdir -p "$(dirname "${LOG_FILE}")" "${OUTPUT_DIR}"

cd "${GEMMA_DIR}"
"${PYTHON_BIN}" -m deepspeed.launcher.runner --include localhost:${CUDA_VISIBLE_DEVICES} src/train/train_sft.py \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --model_id "${MODEL_ID}" \
  --data_path "${DATA_PATH}" \
  --image_folder "${IMAGE_FOLDER}" \
  --patch_mask_analysis_path "${PATCH_MASK_ANALYSIS_PATH}" \
  --disable_number_mask_loss True \
  --use_dual_input_gate True \
  --gate_text_model_id "${GATE_TEXT_MODEL_ID}" \
  --freeze_gate_text_encoder True \
  --gate_l1_loss_weight 0.01 \
  --mask_patch_loss_weight 0.125 \
  --use_liger False \
  --disable_flash_attn2 "${DISABLE_FLASH_ATTN2}" \
  --attn_implementation "${ATTN_IMPLEMENTATION}" \
  --lora_enable False \
  --freeze_projector False \
  --freeze_vision_tower "${FREEZE_VISION_TOWER}" \
  --freeze_llm False \
  --bf16 True \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs 3 \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate 2e-5 \
  --projector_lr 2e-5 \
  --vision_lr 2e-6 \
  --gate_lr 2e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --adam_beta2 0.95 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --tf32 True \
  --gradient_checkpointing True \
  --report_to none \
  --lazy_preprocess True \
  --save_strategy epoch \
  --save_total_limit 3 \
  --dataloader_num_workers 4 2>&1 | tee "${LOG_FILE}"
