# #!/bin/bash

MODEL_NAME="google/gemma-3-4b-it"

export PYTHONPATH=.:src:$PYTHONPATH

python src/merge_lora_weights.py \
    --model-path outputs/test_lora \
    --model-base $MODEL_NAME  \
    --save-model-path outputs/merge_test \
    --safe-serialization