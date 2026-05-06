#!/bin/bash
set -euo pipefail

export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

cd /path/to/sage_repro_bundle

PROMPT_VERSION="v1"
MODEL_PATH="/path/to/sage_repro_bundle/checkpoints/llava_merged_train_raw_1gpu_wandb_from0_0311_013658"
DATA_PATH="./grpo_train.json"
IMAGE_FOLDER="data/images/coco/train2014"
VISION_TOWER="./clip-vit-large-patch14-336"
EVAL_DATA_PATH="./grpo_val.json"
OUTPUT_DIR="./checkpoints/llava-grpo-anti"
DEEPSPEED_CONFIG="./scripts/zero2.json"

TRAIN_LANGUAGE_MODEL="False"
TRAIN_VISION_TOWER="False"
TRAIN_MM_PROJECTOR="True"
TRAIN_GATE="True"
TRAIN_LM_HEAD="True"
LORA_ENABLE="False"

export PYTHONPATH="/path/to/local_scratch/LLaVA:${PYTHONPATH:-}"

deepspeed llava/train/train_grpo_anti.py \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --model_name_or_path ${MODEL_PATH} \
    --version ${PROMPT_VERSION} \
    --data_path ${DATA_PATH} \
    --eval_data_path ${EVAL_DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --vision_tower ${VISION_TOWER} \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 200 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
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
    --grpo_update_epochs 4 \
    --grpo_adaptive_group_reuse True \
    --grpo_low_var_threshold 0.15 \
    --grpo_high_var_threshold 0.60 \
    --grpo_eval_max_groups 200 \
    --grpo_max_new_tokens 16 \
    --grpo_do_sample True \
    --grpo_temperature 1.0 \
    --grpo_top_p 0.95 \
    --grpo_reward_match_as 1.0 \
    --grpo_reward_other 0.2 \
    --grpo_reward_shortcut -1.0
