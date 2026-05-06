#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_llava.sh"

SHORTCUT_CODE_DIR="${SHORTCUT_CODE_DIR:-${BUNDLE_ROOT}/code/shortcut_pipeline}"
INPUT_JSON="${INPUT_JSON:-${BUNDLE_ROOT}/data/napo/shortcut_generated_vqa/train.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${BUNDLE_ROOT}/data/napo_llava/train_raw_pos_neg_shortcut_hf}"
IMAGE_ROOT="${IMAGE_ROOT:-${BUNDLE_ROOT}/data/images/coco/train2014}"

echo "Build LLaVA NaPO HF dataset"
echo "Input:  ${INPUT_JSON}"
echo "Image:  ${IMAGE_ROOT}"
echo "Output: ${OUTPUT_DIR}"
echo

check_path "${INPUT_JSON}" "shortcut NaPO train JSON"
check_path "${IMAGE_ROOT}" "COCO train2014 image root"

CMD=(
  "${PYTHON_BIN}" "${SHORTCUT_CODE_DIR}/build_shortcut_napo_llava_dataset.py"
  --input-json "${INPUT_JSON}"
  --output-dir "${OUTPUT_DIR}"
  --image-root "${IMAGE_ROOT}"
  --overwrite
)

run_or_echo "${CMD[@]}"
