#!/bin/bash
set -e

# Disable all proxy variables to avoid wandb connectivity issues.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}."
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')

VISION_TOWER=${VISION_TOWER:-/path/to/local_scratch/LLaVA/clip-vit-large-patch14-336}
IMAGE_FOLDER=${IMAGE_FOLDER:-./playground/data}
REPORT_TO=${REPORT_TO:-none}
OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints/llava-v1.5-13b-pretrain}
EXTRA_ARGS=${EXTRA_ARGS:-}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-./scripts/zero2_bf16.json}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
WANDB_PROJECT=${WANDB_PROJECT:-llava-v1_5-pretrain}
WANDB_NAME=${WANDB_NAME:-$(basename "${OUTPUT_DIR}")}
export WANDB_PROJECT WANDB_NAME

deepspeed --num_gpus "${NUM_GPUS}" llava/train/train_xformers.py \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --model_name_or_path llava-v1.5-7b \
    --version v1 \
    --data_path ./playground/data/llava_v1_5_mix665k_single_noocr_max200_imageonly_strict_noocr.json \
    --image_folder "${IMAGE_FOLDER}" \
    --vision_tower "${VISION_TOWER}" \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio "${WARMUP_RATIO}" \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to "${REPORT_TO}" \
    ${EXTRA_ARGS}
