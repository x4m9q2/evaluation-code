#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_llava.sh"

CAUSALMM_ROOT="${BUNDLE_ROOT}/code/evaluation/causalmm_llava/llava-1.5"
CAUSALMM_EXP_ROOT="${CAUSALMM_ROOT}/experiments"
SAGE_AS_ROOT="${SAGE_AS_ROOT:-${BUNDLE_ROOT}/data/sage_as}"
CMSV_DATASET="${CMSV_DATASET:-${LLAVA_EVAL_DATASET:-vqa}}"
MODEL_PATH="${MODEL_PATH:-${LLAVA_BASE_MODEL}}"
MODEL_BASE="${MODEL_BASE:-}"
OUT_DIR="${OUT_DIR:-${OUTPUT_ROOT}/cmsv_causalmm/llava/${CMSV_DATASET}}"
IMAGE_POLICY="${IMAGE_POLICY:-original}"
LIMIT="${LIMIT:-}"
GAMMA="${GAMMA:-1.0}"
EPSILON="${EPSILON:-0.6}"
SEED="${SEED:-55}"

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
OUTPUT_FILE="${OUTPUT_FILE:-${OUT_DIR}/llava_causalmm_${DATASET_ARG}_cmsv_predictions.jsonl}"

check_path "${MODEL_PATH}" "LLaVA model for CausalMM"
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

CMD=(
  "${PYTHON_BIN}" "${CAUSALMM_EXP_ROOT}/eval/object_hallucination_vqa_llava.py"
  --model-path "${MODEL_PATH}"
  --question-file "${QUESTION_FILE}"
  --image-folder "${IMAGE_FOLDER}"
  --answers-file "${OUTPUT_FILE}"
  --gamma "${GAMMA}"
  --epsilon "${EPSILON}"
  --seed "${SEED}"
)
if [[ -n "${MODEL_BASE}" ]]; then
  CMD+=(--model-base "${MODEL_BASE}")
fi

(
  cd "${CAUSALMM_ROOT}"
  export PYTHONPATH="${CAUSALMM_ROOT}:${CAUSALMM_EXP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  run_or_echo "${CMD[@]}" "$@"
)
