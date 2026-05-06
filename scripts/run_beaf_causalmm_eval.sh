#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

BEAF_DIR="${BUNDLE_ROOT}/code/beaf_causalmm/gemma3"
MODEL_PATH="${MODEL_PATH:-${BASE_MODEL_ID}}"
OUT_DIR="${OUT_DIR:-${BUNDLE_ROOT}/outputs/beaf_causalmm}"
OUTPUT_FILE="${OUTPUT_FILE:-${OUT_DIR}/gemma3_causalmm_test_raw.json}"
QUESTION_FILE="${QUESTION_FILE:-${OUT_DIR}/test_raw_llava.jsonl}"
ANSWER_FILE="${ANSWER_FILE:-${TEST_RAW_WITH_SHORTCUT}}"
NUM_SHARDS="${NUM_SHARDS:-4}"

mkdir -p "${OUT_DIR}"

if [[ ! -f "${QUESTION_FILE}" ]]; then
  "${PYTHON_BIN}" "${BUNDLE_ROOT}/scripts/tools/convert_test_raw_to_llava_jsonl.py" \
    --input "${ANSWER_FILE}" \
    --output "${QUESTION_FILE}"
fi

SHARD_DIR="${OUT_DIR}/shards"
mkdir -p "${SHARD_DIR}"
"${PYTHON_BIN}" "${BUNDLE_ROOT}/scripts/tools/split_jsonl_shards.py" \
  --input "${QUESTION_FILE}" \
  --output-dir "${SHARD_DIR}" \
  --num-shards "${NUM_SHARDS}"

cd "${BEAF_DIR}"
SHARD_OUTPUTS=()
for SHARD_ID in $(seq 0 $((NUM_SHARDS - 1))); do
  SHARD_QUESTION_FILE="${SHARD_DIR}/test_raw_llava.shard${SHARD_ID}of${NUM_SHARDS}.jsonl"
  SHARD_OUTPUT_FILE="${SHARD_DIR}/gemma3_causalmm_test_raw.shard${SHARD_ID}of${NUM_SHARDS}.json"
  SHARD_OUTPUTS+=("${SHARD_OUTPUT_FILE}.jsonl")
  "${PYTHON_BIN}" eval_test_raw_gemma3_causalmm.py \
    --model-path "${MODEL_PATH}" \
    --question-file "${SHARD_QUESTION_FILE}" \
    --answer-file "${ANSWER_FILE}" \
    --image-folder "${STAGE2_IMAGE_FOLDER}" \
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

"${PYTHON_BIN}" merge_eval_test_raw_shards.py \
  "${SHARD_OUTPUTS[@]}" \
  --question-file "${QUESTION_FILE}" \
  --output-file "${OUTPUT_FILE}"
