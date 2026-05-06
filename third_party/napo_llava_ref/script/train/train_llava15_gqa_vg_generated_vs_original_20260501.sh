#!/bin/bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export MUFFIN_LOGP_NUM_WORKERS="${MUFFIN_LOGP_NUM_WORKERS:-4}"

MODEL_PATH="${MODEL_PATH:-models/llava-v1.5-7b}"
VISION_TOWER="${VISION_TOWER:-models/clip-vit-large-patch14-336}"
OUTPUT_ROOT="${OUTPUT_ROOT:-third_party/napo_llava_ref/.ckpt}"

NUM_EPOCHS="${NUM_EPOCHS:-3}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-16}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-5e-7}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
DPO_BETA="${DPO_BETA:-0.1}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-./script/zero2.json}"
MASTER_PORT="${MASTER_PORT:-29500}"
COMMON_EXTRA_ARGS="${COMMON_EXTRA_ARGS:-}"

run_one() {
    local name="$1"
    local data_dir="$2"
    local run_name="$3"
    local output_dir="${OUTPUT_ROOT}/${run_name}/checkpoints"
    local logging_dir="${OUTPUT_ROOT}/${run_name}/log"

    echo "===== START ${name} $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
    echo "RUN_NAME=${run_name}"
    echo "MODEL_PATH=${MODEL_PATH}"
    echo "DATA_DIR=${data_dir}"
    echo "OUTPUT_DIR=${output_dir}"

    deepspeed --master_port "${MASTER_PORT}" ./muffin/train/train_llava15.py \
        --deepspeed "${DEEPSPEED_CONFIG}" \
        --ddp_timeout 180000 \
        --model_name_or_path "${MODEL_PATH}" \
        --data_dir "${data_dir}" \
        --image_folder not_used \
        --vision_tower "${VISION_TOWER}" \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --fully_tune True \
        --image_aspect_ratio pad \
        --bf16 True \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --output_dir "${output_dir}" \
        --num_train_epochs "${NUM_EPOCHS}" \
        --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
        --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
        --evaluation_strategy no \
        --save_strategy epoch \
        --save_total_limit 3 \
        --save_only_model True \
        --data_source_names '' \
        --data_source_weights 1 \
        --learning_rate "${LEARNING_RATE}" \
        --weight_decay "${WEIGHT_DECAY}" \
        --warmup_ratio "${WARMUP_RATIO}" \
        --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
        --logging_steps "${LOGGING_STEPS}" \
        --logging_dir "${logging_dir}" \
        --tf32 True \
        --model_max_length "${MODEL_MAX_LENGTH}" \
        --gradient_checkpointing True \
        --lazy_preprocess True \
        --task DPO \
        --report_to none \
        --run_name "${run_name}" \
        --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
        --dpo_use_average False \
        --dpo_token_weighted False \
        --dpo_token_weight 1.0 \
        --dpo_beta "${DPO_BETA}" \
        ${COMMON_EXTRA_ARGS}

    local status=$?
    echo "===== END ${name} status=${status} $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
    return "${status}"
}

TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
GQA_DATA_DIR="${GQA_DATA_DIR:-third_party/napo_llava_ref/datasets/gqa_area001_max0p5_genans_pos_origans_neg_hf}"
VG_DATA_DIR="${VG_DATA_DIR:-third_party/napo_llava_ref/datasets/vg_area001_max0p5_genans_pos_origans_neg_hf}"
GQA_RUN_NAME="${GQA_RUN_NAME:-napo_gqa_area001_max0p5_genans_pos_origans_neg_3epoch_${TIMESTAMP}}"
VG_RUN_NAME="${VG_RUN_NAME:-napo_vg_area001_max0p5_genans_pos_origans_neg_3epoch_${TIMESTAMP}}"

run_one "gqa" "${GQA_DATA_DIR}" "${GQA_RUN_NAME}"
gqa_status=$?

run_one "vg" "${VG_DATA_DIR}" "${VG_RUN_NAME}"
vg_status=$?

echo "SUMMARY gqa_status=${gqa_status} vg_status=${vg_status}"
if [[ "${gqa_status}" -ne 0 || "${vg_status}" -ne 0 ]]; then
    exit 1
fi
