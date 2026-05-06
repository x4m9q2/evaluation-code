#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_llava.sh"

TASK="${1:-help}"
shift || true

print_help() {
  cat <<EOF
Usage: $0 <task>

Tasks:
  help                 Show this message.
  vqa-generate         Generate VQA train_raw SAM3 union masks.
  vqa-filter           Run Qwen visual-cue filtering for VQA and merge shard outputs.
  vqa-build            Build the final VQA mixed JSON + compat NPZ package.
  gqa-vg-generate      Generate GQA/VG sampled10000 SAM3 union masks.
  gqa-vg-filter        Run Qwen visual-cue filtering for GQA/VG and merge shard outputs.
  gqa-vg-build         Build GQA/VG qwenkeep+nonumbermask packages.

Common env overrides:
  PYTHON_BIN, SAM3_CHECKPOINT, MASK_GPUS, NUM_SHARDS, BATCH_SIZE, RESOLUTION, SCORE_THRESH, RUN=1
EOF
}

case "${TASK}" in
  help|-h|--help)
    print_help
    ;;
  vqa-generate)
    VQA_QA_JSONL="${VQA_QA_JSONL:-${BUNDLE_ROOT}/data/stage2/train_raw_llava.jsonl}"
    VQA_MAPPING_JSON="${VQA_MAPPING_JSON:-${BUNDLE_ROOT}/data/stage2/merged_output_rule_mapping.json}"
    VQA_MASK_OUTPUT="${VQA_MASK_OUTPUT:-${BUNDLE_ROOT}/outputs/sam3_train_raw_llava_union_masks}"
    check_path "${SAM3_CHECKPOINT}" "SAM3 checkpoint"
    check_path "${VQA_QA_JSONL}" "VQA QA JSONL"
    check_path "${VQA_MAPPING_JSON}" "VQA rule mapping"
    check_path "${VQA_TRAIN2014_IMAGE_ROOT}" "COCO train2014 image root"
    run_or_echo "${PYTHON_BIN}" "${BUNDLE_ROOT}/code/sam3/scripts/generate_union_masks_from_mapping.py" \
      --qa-jsonl "${VQA_QA_JSONL}" \
      --mapping-json "${VQA_MAPPING_JSON}" \
      --image-root "${VQA_TRAIN2014_IMAGE_ROOT}" \
      --output-dir "${VQA_MASK_OUTPUT}" \
      --batch-size "${BATCH_SIZE:-32}" \
      --resolution "${RESOLUTION:-1008}" \
      --score-thresh "${SCORE_THRESH:-0.5}" \
      --checkpoint-path "${SAM3_CHECKPOINT}" \
      --device "${DEVICE:-cuda}" \
      --no-load-from-hf \
      --num-shards "${NUM_SHARDS:-4}" \
      --shard-index "${SHARD_INDEX:-0}"
    ;;
  vqa-filter)
    run_or_echo bash "${BUNDLE_ROOT}/scripts/run_qwen_visual_cue_filter.sh" vqa
    ;;
  vqa-build)
    run_or_echo "${PYTHON_BIN}" "${BUNDLE_ROOT}/code/llava_sage/scripts2/build_qwenkeep_stage2_package.py" "$@"
    ;;
  gqa-vg-generate)
    check_path "${SAM3_CHECKPOINT}" "SAM3 checkpoint"
    run_or_echo bash "${BUNDLE_ROOT}/code/data_tools/gqa_vg_sam3_masks_sampled10000_20260430/run_generate_masks.sh" "$@"
    ;;
  gqa-vg-filter)
    run_or_echo bash "${BUNDLE_ROOT}/scripts/run_qwen_visual_cue_filter.sh" "${1:-all}"
    ;;
  gqa-vg-build)
    run_or_echo "${PYTHON_BIN}" "${BUNDLE_ROOT}/code/data_tools/gqa_vg_sam3_masks_sampled10000_20260430/build_qwenkeep_packages.py"
    ;;
  *)
    echo "[error] unknown task: ${TASK}" >&2
    print_help >&2
    exit 2
    ;;
esac
