#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

CAUSALMM_DIR="${BUNDLE_ROOT}/code/causalmm_gemma3/gemma3"
SAGE_AS_ROOT="${SAGE_AS_ROOT:-${BUNDLE_ROOT}/data/sage_as}"
CMSV_DATASET="${CMSV_DATASET:-vqa}"
MODEL_PATH="${MODEL_PATH:-${BASE_MODEL_ID}}"
OUT_DIR="${OUT_DIR:-${BUNDLE_ROOT}/outputs/cmsv_causalmm/gemma/${CMSV_DATASET}}"
NUM_SHARDS="${NUM_SHARDS:-4}"
IMAGE_POLICY="${IMAGE_POLICY:-original}"
LIMIT="${LIMIT:-}"

case "${CMSV_DATASET}" in
  vqa|vqa_v2_cmsv|vqa-v2-cmsv)
    DATASET_ARG="vqa"
    DATA_PATH="${DATA_PATH:-${SAGE_AS_ROOT}/data/vqa_v2_cmsv/test.json}"
    IMAGE_FOLDER="${IMAGE_FOLDER:-${BUNDLE_ROOT}/data/images/coco/train2014}"
    ;;
  gqa|gqa_cmsv|gqa-cmsv)
    DATASET_ARG="gqa"
    DATA_PATH="${DATA_PATH:-${SAGE_AS_ROOT}/data/gqa_cmsv/test.jsonl}"
    IMAGE_FOLDER="${IMAGE_FOLDER:-${BUNDLE_ROOT}/data/images/gqa/images}"
    ;;
  vg|vg_cmsv|vg-cmsv)
    DATASET_ARG="vg"
    DATA_PATH="${DATA_PATH:-${SAGE_AS_ROOT}/data/vg_cmsv/test.jsonl}"
    IMAGE_FOLDER="${IMAGE_FOLDER:-${BUNDLE_ROOT}/data/images/vg}"
    ;;
  *)
    echo "[error] CMSV_DATASET must be one of: vqa, gqa, vg. Got: ${CMSV_DATASET}" >&2
    exit 2
    ;;
esac

mkdir -p "${OUT_DIR}"

QUESTION_FILE="${QUESTION_FILE:-${OUT_DIR}/${DATASET_ARG}_cmsv_questions.jsonl}"
ANSWER_FILE="${ANSWER_FILE:-${OUT_DIR}/${DATASET_ARG}_cmsv_answers.json}"
OUTPUT_FILE="${OUTPUT_FILE:-${OUT_DIR}/gemma3_causalmm_${DATASET_ARG}_cmsv_predictions.json}"

check_path "${MODEL_PATH}" "Gemma model for CausalMM"
check_path "${DATA_PATH}" "CMSV test split"
check_path "${IMAGE_FOLDER}" "CMSV image folder"

CONVERT_CMD=(
  "${PYTHON_BIN}" "${BUNDLE_ROOT}/scripts/tools/convert_cmsv_test_to_causalmm_inputs.py"
  --input "${DATA_PATH}"
  --dataset "${DATASET_ARG}"
  --question-output "${QUESTION_FILE}"
  --answer-output "${ANSWER_FILE}"
  --image-policy "${IMAGE_POLICY}"
)
if [[ -n "${LIMIT}" ]]; then
  CONVERT_CMD+=(--limit "${LIMIT}")
fi
run_or_echo "${CONVERT_CMD[@]}"

SHARD_DIR="${OUT_DIR}/shards"
mkdir -p "${SHARD_DIR}"
run_or_echo "${PYTHON_BIN}" "${BUNDLE_ROOT}/scripts/tools/split_jsonl_shards.py" \
  --input "${QUESTION_FILE}" \
  --output-dir "${SHARD_DIR}" \
  --num-shards "${NUM_SHARDS}"

cd "${CAUSALMM_DIR}"
SHARD_OUTPUTS=()
for SHARD_ID in $(seq 0 $((NUM_SHARDS - 1))); do
  SHARD_QUESTION_FILE="${SHARD_DIR}/${DATASET_ARG}_cmsv_questions.shard${SHARD_ID}of${NUM_SHARDS}.jsonl"
  SHARD_OUTPUT_FILE="${SHARD_DIR}/gemma3_causalmm_${DATASET_ARG}.shard${SHARD_ID}of${NUM_SHARDS}.json"
  SHARD_OUTPUTS+=("${SHARD_OUTPUT_FILE}.jsonl")
  run_or_echo "${PYTHON_BIN}" eval_test_raw_gemma3_causalmm.py \
    --model-path "${MODEL_PATH}" \
    --question-file "${SHARD_QUESTION_FILE}" \
    --answer-file "${ANSWER_FILE}" \
    --image-folder "${IMAGE_FOLDER}" \
    --output-file "${SHARD_OUTPUT_FILE}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-16}" \
    --gamma "${GAMMA:-1.0}" \
    --epsilon "${EPSILON:-0.1}" \
    --temperature "${TEMPERATURE:-0.0}" \
    --top-p "${TOP_P:-1.0}" \
    --batch-size "${BATCH_SIZE:-1}" \
    --resume \
    --cf-mode "${CF_MODE:-language}" \
    --attention-method "${ATTENTION_METHOD:-reverse_and_normalize}" \
    --vision-method "${VISION_METHOD:-shuffle}" \
    --dtype "${DTYPE:-bfloat16}" \
    "$@"
done

run_or_echo "${PYTHON_BIN}" merge_eval_test_raw_shards.py \
  "${SHARD_OUTPUTS[@]}" \
  --question-file "${QUESTION_FILE}" \
  --output-file "${OUTPUT_FILE}"
