#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_llava.sh"

SHORTCUT_CODE_DIR="${SHORTCUT_CODE_DIR:-${BUNDLE_ROOT}/code/shortcut_pipeline}"
SAGE_AS_ROOT="${SAGE_AS_ROOT:-${BUNDLE_ROOT}/data/sage_as}"
OUTPUT_DIR="${OUTPUT_DIR:-${SAGE_AS_ROOT}/data/vqa_v2_cmsv}"
ANNOTATIONS_JSON="${ANNOTATIONS_JSON:-${BUNDLE_ROOT}/data/detect-shortcuts/data/vqa2/v2_mscoco_train2014_annotations.json}"
HF_PROXY="${HF_PROXY:-}"

ATTACH_SHORTCUT_ANSWER="${ATTACH_SHORTCUT_ANSWER:-0}"

echo "Download official VQA v2-CMSV splits"
echo "Output:      ${OUTPUT_DIR}"
echo "Annotations: ${ANNOTATIONS_JSON}"
echo "Proxy:       ${HF_PROXY}"
echo "Attach shortcut_answer: ${ATTACH_SHORTCUT_ANSWER}"
echo

if [[ "${ATTACH_SHORTCUT_ANSWER}" == "1" ]]; then
  check_path "${ANNOTATIONS_JSON}" "official VQA v2 train annotations"
fi

CMD=(
  "${PYTHON_BIN}" "${SHORTCUT_CODE_DIR}/download_and_attach_vqa_v2_cmsv.py"
  --output-dir "${OUTPUT_DIR}"
  --http-proxy "${HF_PROXY}"
)

if [[ "${ATTACH_SHORTCUT_ANSWER}" == "1" ]]; then
  CMD+=(--annotations-json "${ANNOTATIONS_JSON}" --attach-shortcut-answer)
fi

run_or_echo "${CMD[@]}"
