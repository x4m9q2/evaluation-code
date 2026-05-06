#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MODEL_PATH="${STAGE2_CHECKPOINT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BUNDLE_ROOT}/outputs/infer_test_raw}"
NUM_GPUS="${NUM_GPUS:-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
OVERWRITE="${OVERWRITE:-0}"
NO_SHORT_ANSWER_PROMPT="${NO_SHORT_ANSWER_PROMPT:-1}"

MODEL_TAG="$(basename "${MODEL_PATH}")"
OUT_DIR="${OUTPUT_ROOT}/${MODEL_TAG}"
mkdir -p "${OUT_DIR}"

pids=()
for idx in $(seq 0 $((NUM_GPUS - 1))); do
  cmd=(
    "${PYTHON_BIN}" "${CODE_ROOT}/scripts2/eval_test_raw_gemma3.py"
    --model-path "${MODEL_PATH}"
    --data-path "${TEST_RAW_WITH_SHORTCUT}"
    --image-folder "${STAGE2_IMAGE_FOLDER}"
    --output-root "${OUTPUT_ROOT}"
    --batch-size "${BATCH_SIZE}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --temperature "${TEMPERATURE:-0.0}"
    --gpu "${idx}"
    --num-chunks "${NUM_GPUS}"
    --chunk-idx "${idx}"
    --gate-text-model-id "${GATE_TEXT_MODEL_ID}"
  )
  if [[ "${OVERWRITE}" == "1" ]]; then
    cmd+=(--overwrite)
  fi
  if [[ "${NO_SHORT_ANSWER_PROMPT}" == "1" ]]; then
    cmd+=(--no-short-answer-prompt)
  fi

  echo "[launch] chunk ${idx}/${NUM_GPUS} on GPU ${idx}"
  "${cmd[@]}" > "${OUT_DIR}/chunk${idx}of${NUM_GPUS}.log" 2>&1 &
  pids+=($!)
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

"${PYTHON_BIN}" "${BUNDLE_ROOT}/scripts/tools/merge_infer_chunks.py" \
  --source-data "${TEST_RAW_WITH_SHORTCUT}" \
  --chunk-dir "${OUT_DIR}" \
  --num-chunks "${NUM_GPUS}" \
  --output "${OUT_DIR}/test_raw_with_shortcut_answer.merged.json"

echo "${OUT_DIR}/test_raw_with_shortcut_answer.merged.json"
