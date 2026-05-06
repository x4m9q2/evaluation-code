#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
CODE_ROOT="${BUNDLE_ROOT}/code/gemma_gate"
GEMMA_DIR="${CODE_ROOT}/gemma"

export PYTHONPATH="${CODE_ROOT}:${GEMMA_DIR}:${GEMMA_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

PYTHON_BIN="${PYTHON_BIN:-${GEMMA_DIR}/.venv/bin/python}"
RUN_NAME="${RUN_NAME:-gemma3_4b_napo_shortcut_dpo_sdpa_$(date +%Y%m%d_%H%M%S)}"
REPORT_TO="${REPORT_TO:-none}"

# NaPO shortcut training uses the original Gemma model by default. Override MODEL_ID
# explicitly if you want to start from a specific non-gate checkpoint.
MODEL_ID="${BUNDLE_ROOT}/models/Gemma-3-4B-IT"
USE_DUAL_INPUT_GATE="${USE_DUAL_INPUT_GATE:-False}"
GATE_TEXT_MODEL_ID="${BUNDLE_ROOT}/models/siglip-so400m-patch14-384"
DATA_PATH="${BUNDLE_ROOT}/data/napo/train_raw_pos_neg_shortcut.json"
IMAGE_FOLDER="${BUNDLE_ROOT}/data/playground_data/coco/train2014"
OUTPUT_DIR="${BUNDLE_ROOT}/checkpoints/${RUN_NAME}"
LOG_FILE="${BUNDLE_ROOT}/logs/${RUN_NAME}.log"

NUM_GPUS="${NUM_GPUS:-4}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"

export WANDB_PROJECT="${WANDB_PROJECT:-gemma-gate-napo-shortcut}"
export WANDB_NAME="${WANDB_NAME:-${RUN_NAME}}"

mkdir -p "${OUTPUT_DIR}" "$(dirname "${LOG_FILE}")"

{
  echo "[run] ${RUN_NAME}"
  echo "[code] ${CODE_ROOT}"
  echo "[model] ${MODEL_ID}"
  echo "[data] ${DATA_PATH}"
  echo "[image] ${IMAGE_FOLDER}"
  echo "[output] ${OUTPUT_DIR}"
  echo "[python] ${PYTHON_BIN}"
  echo "[batch] ${NUM_GPUS} GPUs x per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} x gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}"
  echo "[loss] napo_loss_type=${NAPO_LOSS_TYPE:-dyn_lq} beta=${BETA:-0.1} alpha=${NAPO_ALPHA:-0.5} dyn_q_use_average=${NAPO_DYN_Q_USE_AVERAGE:-True}"
  echo "[gate] use_dual_input_gate=${USE_DUAL_INPUT_GATE}"
  if [[ "${USE_DUAL_INPUT_GATE}" == "True" || "${USE_DUAL_INPUT_GATE}" == "true" || "${USE_DUAL_INPUT_GATE}" == "1" ]]; then
    echo "[gate] gate_text_model_id=${GATE_TEXT_MODEL_ID}"
  else
    echo "[gate] disabled: no SigLIP text encoder or gate LR will be used"
  fi
  echo "[attn] language=${ATTN_IMPLEMENTATION:-eager} vision=${VISION_ATTN_IMPLEMENTATION:-sdpa}"
  echo "[nccl] P2P default, IB disabled"
  echo "[report_to] ${REPORT_TO}"
  echo "[wandb] project=${WANDB_PROJECT} name=${WANDB_NAME}"
  date
} | tee "${LOG_FILE}"

cd "${GEMMA_DIR}"
EXTRA_GATE_ARGS=()
if [[ "${USE_DUAL_INPUT_GATE}" == "True" || "${USE_DUAL_INPUT_GATE}" == "true" || "${USE_DUAL_INPUT_GATE}" == "1" ]]; then
  EXTRA_GATE_ARGS+=("--gate_text_model_id" "${GATE_TEXT_MODEL_ID}")
  EXTRA_GATE_ARGS+=("--freeze_gate_text_encoder" "True")
  EXTRA_GATE_ARGS+=("--gate_lr" "${GATE_LR:-5e-7}")
fi
"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${NUM_GPUS}" \
  src/train/train_dpo.py \
  --model_id "${MODEL_ID}" \
  --data_path "${DATA_PATH}" \
  --image_folder "${IMAGE_FOLDER}" \
  --use_dual_input_gate "${USE_DUAL_INPUT_GATE}" \
  "${EXTRA_GATE_ARGS[@]}" \
  --napo_loss_type "${NAPO_LOSS_TYPE:-dyn_lq}" \
  --napo_alpha "${NAPO_ALPHA:-0.5}" \
  --napo_q "${NAPO_Q:-1.0}" \
  --napo_dyn_q_use_average "${NAPO_DYN_Q_USE_AVERAGE:-True}" \
  --disable_token_type_ids "${DISABLE_TOKEN_TYPE_IDS:-False}" \
  --disable_ref_model "${DISABLE_REF_MODEL:-False}" \
  --dpo_loss sigmoid \
  --precompute_ref_log_probs False \
  --beta "${BETA:-0.1}" \
  --use_liger False \
  --disable_flash_attn2 True \
  --attn_implementation "${ATTN_IMPLEMENTATION:-eager}" \
  --vision_attn_implementation "${VISION_ATTN_IMPLEMENTATION:-sdpa}" \
  --lora_enable False \
  --freeze_projector False \
  --freeze_vision_tower True \
  --freeze_llm False \
  --bf16 True \
  --output_dir "${OUTPUT_DIR}" \
  --remove_unused_columns False \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate "${LEARNING_RATE:-5e-7}" \
  --projector_lr "${PROJECTOR_LR:-5e-7}" \
  --vision_lr "${VISION_LR:-2e-7}" \
  --weight_decay "${WEIGHT_DECAY:-0.01}" \
  --warmup_ratio "${WARMUP_RATIO:-0.1}" \
  --adam_beta2 0.95 \
  --lr_scheduler_type cosine \
  --logging_steps "${LOGGING_STEPS:-1}" \
  --tf32 True \
  --gradient_checkpointing True \
  --report_to "${REPORT_TO}" \
  --lazy_preprocess True \
  --save_strategy epoch \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-4}" \
  "$@" 2>&1 | tee -a "${LOG_FILE}"
