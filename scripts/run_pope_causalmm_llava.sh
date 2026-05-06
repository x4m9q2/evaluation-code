#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_llava.sh"

CAUSALMM_ROOT="${BUNDLE_ROOT}/code/evaluation/causalmm_llava/llava-1.5"
CAUSALMM_EXP_ROOT="${CAUSALMM_ROOT}/experiments"
MODEL_PATH="${MODEL_PATH:-${LLAVA_BASE_MODEL}}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${POPE_IMAGE_ROOT}}"
QUESTION_FILE="${QUESTION_FILE:-${CAUSALMM_EXP_ROOT}/data/POPE/coco/coco_pope_random.json}"
OUT_DIR="${OUT_DIR:-${OUTPUT_ROOT}/pope_causalmm_llava}"
ANSWER_FILE="${ANSWER_FILE:-${OUT_DIR}/llava15_coco_pope_random_answers.jsonl}"
GAMMA="${GAMMA:-1.0}"
EPSILON="${EPSILON:-0.6}"
SEED="${SEED:-55}"

mkdir -p "${OUT_DIR}"

check_path "${MODEL_PATH}" "LLaVA base model for CausalMM"
check_path "${QUESTION_FILE}" "CausalMM POPE question file"
check_path "${IMAGE_FOLDER}" "CausalMM POPE image folder"

(
  cd "${CAUSALMM_ROOT}"
  export PYTHONPATH="${CAUSALMM_ROOT}:${CAUSALMM_EXP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  run_or_echo "${PYTHON_BIN}" "${CAUSALMM_EXP_ROOT}/eval/object_hallucination_vqa_llava.py" \
    --model-path "${MODEL_PATH}" \
    --question-file "${QUESTION_FILE}" \
    --image-folder "${IMAGE_FOLDER}" \
    --answers-file "${ANSWER_FILE}" \
    --gamma "${GAMMA}" \
    --epsilon "${EPSILON}" \
    --seed "${SEED}" \
    "$@"
)
