#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CODE_ROOT="${BUNDLE_ROOT}/code/gemma_gate"
PYTHON="${PYTHON:-python}"
MODEL_PATH="${BUNDLE_ROOT}/models/Gemma-3-4B-IT"
DATA_PATH="${BUNDLE_ROOT}/data/eval/test_raw_with_shortcut_answer.json"
IMAGE_FOLDER="${BUNDLE_ROOT}/data/playground_data/coco/train2014"
OUTPUT_ROOT="${BUNDLE_ROOT}/outputs/infer_result"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
XVERIFY_ROOT="${CODE_ROOT}/x_verify"
XVERIFY_MODEL_PATH="${XVERIFY_ROOT}/xVerify-0.5B-I"
XVERIFY_PYTHON="${PYTHON}"
XVERIFY_GPU="${XVERIFY_GPU:-0}"
XVERIFY_BATCH_SIZE="${XVERIFY_BATCH_SIZE:-32}"

export PYTHONPATH="${CODE_ROOT}:${CODE_ROOT}/gemma:${CODE_ROOT}/gemma/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON}" "${CODE_ROOT}/scripts2/eval_test_raw_gemma3.py" \
  --model-path "${MODEL_PATH}" \
  --data-path "${DATA_PATH}" \
  --image-folder "${IMAGE_FOLDER}" \
  --output-root "${OUTPUT_ROOT}" \
  --batch-size "${BATCH_SIZE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --gpu "${GPU}" \
  --run-xverify \
  --xverify-root "${XVERIFY_ROOT}" \
  --xverify-model-path "${XVERIFY_MODEL_PATH}" \
  --xverify-python "${XVERIFY_PYTHON}" \
  --xverify-gpu "${XVERIFY_GPU}" \
  --xverify-batch-size "${XVERIFY_BATCH_SIZE}" \
  --overwrite
