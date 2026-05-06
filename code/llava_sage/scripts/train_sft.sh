cd /path/to/local_scratch/LLaVA

CUDA_VISIBLE_DEVICES=0 \
REPORT_TO=wandb \
WANDB_PROJECT=llava-train-raw \
WANDB_NAME=train_raw_1gpu_wandb_from0_0311_013658 \
OUTPUT_DIR=./checkpoints/train_raw_1gpu_wandb_from0_0311_013658 \
EXTRA_ARGS="--model_name_or_path /path/to/local_scratch/LLaVA/checkpoints/assembled_llava_v15_from_mmproj_gate_20260311 \
--data_path /path/to/local_scratch/LLaVA/playground/data/train_raw_llava_train2017.json \
--image_folder /path/to/local_scratch/LLaVA/playground/data/coco \
--per_device_train_batch_size 8 \
--gradient_checkpointing False \
--logging_steps 10" \
bash scripts/v1_5/pretrain.sh