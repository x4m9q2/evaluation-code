#!/bin/bash
set -euo pipefail

export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

cd /path/to/sage_repro_bundle

PROMPT_VERSION="v1"
MODEL_PATH="/path/to/sage_repro_bundle/checkpoints/assembled_llava_v15_from_mmproj_gate_20260311"
DATA_PATH="./grpo_train.json"
IMAGE_FOLDER="/root"
VISION_TOWER="./clip-vit-large-patch14-336"
OUTPUT_DIR="./checkpoints/llava_grpo_anti_smoke"

TRAIN_LANGUAGE_MODEL="False"
TRAIN_VISION_TOWER="False"
TRAIN_MM_PROJECTOR="True"
TRAIN_GATE="True"
TRAIN_LM_HEAD="True"
LORA_ENABLE="False"

export PYTHONPATH="/path/to/local_scratch/LLaVA:${PYTHONPATH:-}"

torchrun --nproc_per_node=4 --master_port=29501 llava/train/train_grpo_anti.py \
    --model_name_or_path ${MODEL_PATH} \
    --version ${PROMPT_VERSION} \
    --data_path ${DATA_PATH} \
    --eval_data_path ./grpo_val.json \
    --image_folder ${IMAGE_FOLDER} \
    --vision_tower ${VISION_TOWER} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --max_steps 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 2 \
    --save_strategy "steps" \
    --save_steps 1 \
    --save_total_limit 2 \
    --learning_rate 1e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
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
    --grpo_group_size 4 \
    --grpo_update_epochs 4 \
    --grpo_adaptive_group_reuse True \
    --grpo_low_var_threshold 0.15 \
    --grpo_high_var_threshold 0.60 \
    --grpo_eval_max_groups 8 \
    --grpo_max_new_tokens 8 \
    --grpo_min_new_tokens 1 \
    --grpo_do_sample True \
    --grpo_temperature 1.0 \
    --grpo_top_p 0.95 \
    --grpo_reward_match_as 1.0 \
    --grpo_reward_other 0.2 \
    --grpo_reward_shortcut -1.0 \
    --grpo_clip_epsilon 0.2 \
    --grpo_kl_coef 0.1