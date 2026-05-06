#!/bin/bash

MODEL_NAME="google/gemma-3-4b-it"

export PYTHONPATH=.:src:$PYTHONPATH

deepspeed src/train/train_grpo.py \
    --optim adamw_bnb_8bit \
    --max_completion_length 256 \
    --max_prompt_length 512 \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path /path/to/your/training/data.json \
    --image_folder /path/to/your/image/folder \
    --disable_flash_attn2 True \
    --lora_enable False \
    --freeze_projector False \
    --freeze_vision_tower False \
    --freeze_llm False \
    --bf16 True \
    --output_dir output/test \
    --num_train_epochs 1 \
    --num_generations 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --projector_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --dataloader_num_workers 4