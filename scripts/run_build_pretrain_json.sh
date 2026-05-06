#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_llava.sh"

RAW_LLAVA_MIX="${RAW_LLAVA_MIX:-${BUNDLE_ROOT}/data/llava_stage1/llava_v1_5_mix665k.json}"
STRICT_JSON="${STRICT_JSON:-${BUNDLE_ROOT}/data/pretrain_llava_v1_5_mix665k_single_noocr_max200_imageonly_strict_noocr.json}"
STRICT_STATS="${STRICT_STATS:-${BUNDLE_ROOT}/data/pretrain_llava_v1_5_mix665k_single_noocr_max200_imageonly_strict_noocr.stats.json}"
STRICT_DROPLIST="${STRICT_DROPLIST:-${BUNDLE_ROOT}/code/data_tools/llava_v1_5_strict_noocr_drop_hashes.txt}"

mkdir -p "$(dirname "${STRICT_JSON}")"

echo "Build LLaVA pretraining JSON with relative paths."
echo "Input raw mix: ${RAW_LLAVA_MIX}"
echo "Output final:  ${STRICT_JSON}"
echo

check_path "${RAW_LLAVA_MIX}" "raw llava_v1_5_mix665k.json"

run_or_echo "${PYTHON_BIN}" "${BUNDLE_ROOT}/code/data_tools/build_llava_pretrain_json.py" \
  --input "${RAW_LLAVA_MIX}" \
  --output "${STRICT_JSON}" \
  --stats-output "${STRICT_STATS}" \
  --max-answer-chars 200 \
  --mode aggressive \
  --drop-image-prefix ocr_vqa/ \
  --drop-hashes "${STRICT_DROPLIST}"
