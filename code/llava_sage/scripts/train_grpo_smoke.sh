#!/bin/bash
set -euo pipefail

export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

cd /path/to/local_scratch/LLaVA

PROMPT_VERSION="v1"
MODEL_PATH="/path/to/local_scratch/LLaVA/checkpoints/llava_merged_train_merged_1gpu_wandb_from0_0317_1"
DATA_PATH="./grpo_train.json"
IMAGE_FOLDER="/path/to/local_scratch/sam3"
VISION_TOWER="./clip-vit-large-patch14-336"
OUTPUT_DIR="./checkpoints/grpo-debug-ori-only"
DEEPSPEED_CONFIG="./scripts/zero2.json"

TRAIN_LANGUAGE_MODEL="False"
TRAIN_VISION_TOWER="False"
TRAIN_MM_PROJECTOR="True"
TRAIN_GATE="True"
TRAIN_LM_HEAD="False"
LORA_ENABLE="False"

export PYTHONPATH="/path/to/local_scratch/LLaVA:${PYTHONPATH:-}"

deepspeed llava/train/train_grpo.py \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --model_name_or_path ${MODEL_PATH} \
    --version ${PROMPT_VERSION} \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --vision_tower ${VISION_TOWER} \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --max_steps 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 3 \
    --save_total_limit 2 \
    --learning_rate 5e-7 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to wandb \
    --tune_language_model ${TRAIN_LANGUAGE_MODEL} \
    --tune_vision_tower ${TRAIN_VISION_TOWER} \
    --tune_mm_projector ${TRAIN_MM_PROJECTOR} \
    --tune_gate ${TRAIN_GATE} \
    --tune_lm_head ${TRAIN_LM_HEAD} \
    --lora_enable ${LORA_ENABLE} \
    --grpo_group_size 6 \
    --grpo_max_new_tokens 8 \
    --grpo_do_sample True \
    --grpo_temperature 1.0 \
    --grpo_top_p 0.95 \
    --grpo_lambda_ori 0.0 \
    --grpo_lambda_as 0.0 \
    --grpo_lambda_sep 0.0 \
    --grpo_lambda_cross 0.0 \
    --grpo_empty_penalty 1.0 \
    --grpo_length_penalty 0.5 \
    --grpo_repeat_penalty 0.5 \
    --grpo_max_answer_words 4 \
    --grpo_kl_coef 0.05