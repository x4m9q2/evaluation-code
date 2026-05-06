#!/bin/bash
set -euo pipefail

cd /path/to/local_scratch/LLaVA/checkpoints

ln -sfn merged_train_merged_1gpu_wandb_from0_0317 \
        llava_merged_train_merged_1gpu_wandb_from0_0317

echo "Symlink ready:"
echo "/path/to/local_scratch/LLaVA/checkpoints/merged_train_merged_1gpu_wandb_from0_0317"