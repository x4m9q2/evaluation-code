#!/bin/bash
set -euo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

REPO_ROOT=${REPO_ROOT:-/path/to/sage_repro_bundle}
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NPROC_PER_NODE=${NPROC_PER_NODE:-$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')}

DATASET=${DATASET:-vg}
VG_JSONL=${VG_JSONL:-${REPO_ROOT}/data2/vg/train.jsonl}
GQA_JSONL=${GQA_JSONL:-${REPO_ROOT}/data2/GQA/train.jsonl}
FORCE_REBUILD_TRAIN_JSON=${FORCE_REBUILD_TRAIN_JSON:-0}

case "${DATASET}" in
  vg)
    MASTER_PORT=${MASTER_PORT:-29542}
    TRAIN_JSON=${TRAIN_JSON:-${REPO_ROOT}/playground/data/data2_vg_train_llava.json}
    OUTPUT_PREFIX=finetune_data2_vg_from_meanreg_p2p_ddp
    ;;
  gqa)
    MASTER_PORT=${MASTER_PORT:-29543}
    TRAIN_JSON=${TRAIN_JSON:-${REPO_ROOT}/playground/data/data2_gqa_train_llava.json}
    OUTPUT_PREFIX=finetune_data2_gqa_from_meanreg_p2p_ddp
    ;;
  *)
    echo "Unsupported DATASET=${DATASET}; expected vg or gqa" >&2
    exit 1
    ;;
esac

# The converted samples use absolute image paths, so "/" is sufficient here.
IMAGE_FOLDER=${IMAGE_FOLDER:-/}
VISION_TOWER=${VISION_TOWER:-${REPO_ROOT}/clip-vit-large-patch14-336}
BASE_MODEL=${BASE_MODEL:-${REPO_ROOT}/llava-v1.5-7b}
PRETRAIN_CKPT_DIR=${PRETRAIN_CKPT_DIR:-${REPO_ROOT}/checkpoints/llava_pretrain_4gpu_xformers_aggressive_p2p_bs32_gate_pretain_meanreg_20260319_052641}
PRETRAIN_ADAPTER=${PRETRAIN_ADAPTER:-${PRETRAIN_CKPT_DIR}/mm_projector.bin}

REPORT_TO=${REPORT_TO:-none}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/checkpoints/${OUTPUT_PREFIX}_${RUN_TAG}}

NUM_EPOCHS=${NUM_EPOCHS:-3}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-2}
SAVE_STEPS=${SAVE_STEPS:-500}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-2}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
LOGGING_STEPS=${LOGGING_STEPS:-10}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-2048}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-True}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
IMAGE_ASPECT_RATIO=${IMAGE_ASPECT_RATIO:-pad}
GROUP_BY_MODALITY_LENGTH=${GROUP_BY_MODALITY_LENGTH:-True}
EXTRA_ARGS=${EXTRA_ARGS:-}

export PRINT_TRAINABLE_PARAMS=${PRINT_TRAINABLE_PARAMS:-0}
export SAVE_INIT_TRAINABLES=${SAVE_INIT_TRAINABLES:-0}
export SAVE_INIT_TO_DISK=${SAVE_INIT_TO_DISK:-0}
export DELTA_SCOPE=${DELTA_SCOPE:-all_trainable}
export DELTA_MODE=${DELTA_MODE:-fingerprint}
export DEBUG_DELTA_AFTER_TRAIN=${DEBUG_DELTA_AFTER_TRAIN:-0}
export DEBUG_DELTA_PRINT_LIMIT=${DEBUG_DELTA_PRINT_LIMIT:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

if [ ! -f "${PRETRAIN_ADAPTER}" ]; then
  echo "Missing pretrain projector: ${PRETRAIN_ADAPTER}" >&2
  exit 1
fi

if [ ! -f "${TRAIN_JSON}" ] || [ "${FORCE_REBUILD_TRAIN_JSON}" = "1" ]; then
  python "${REPO_ROOT}/scripts/data2/convert_data2_train_to_llava.py" \
    --dataset "${DATASET}" \
    --vg-input "${VG_JSONL}" \
    --gqa-input "${GQA_JSONL}" \
    --output "${TRAIN_JSON}"
fi

torchrun --master_port "${MASTER_PORT}" --nproc_per_node="${NPROC_PER_NODE}" llava/train/train_xformers.py \
    --model_name_or_path "${BASE_MODEL}" \
    --version v1 \
    --data_path "${TRAIN_JSON}" \
    --image_folder "${IMAGE_FOLDER}" \
    --vision_tower "${VISION_TOWER}" \
    --pretrain_mm_mlp_adapter "${PRETRAIN_ADAPTER}" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio "${IMAGE_ASPECT_RATIO}" \
    --group_by_modality_length "${GROUP_BY_MODALITY_LENGTH}" \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --learning_rate "${LEARNING_RATE}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --lr_scheduler_type "cosine" \
    --logging_steps "${LOGGING_STEPS}" \
    --tf32 True \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --lazy_preprocess True \
    --report_to "${REPORT_TO}" \
    ${EXTRA_ARGS}
