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
  PYTHON_BIN, SAM3_CHECKPOINT, MASK_GPUS, NUM_SHARDS, BATCH_SIZE, RESOLUTION, SCORE_THRESH
EOF
}

case "${TASK}" in
  help|-h|--help)
    print_help
    ;;
  vqa-generate)
    VQA_QA_JSONL="${VQA_QA_JSONL:-${BUNDLE_ROOT}/data/shortcut_pipeline/cross_modality_qa_questions.jsonl}"
    VQA_MAPPING_JSON="${VQA_MAPPING_JSON:-${BUNDLE_ROOT}/data/shortcut_pipeline/cross_modality_qa_mapping.json}"
    VQA_MASK_OUTPUT="${VQA_MASK_OUTPUT:-${BUNDLE_ROOT}/outputs/sam3_vqa_cmsv_union_masks}"
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
    if [[ -z "${VQA_FILTER_INPUT_ROOT:-}" ]]; then
      if [[ -d "${BUNDLE_ROOT}/outputs/sam3_vqa_cmsv_union_masks/shard_meta" ]]; then
        export VQA_FILTER_INPUT_ROOT="${BUNDLE_ROOT}/outputs/sam3_vqa_cmsv_union_masks"
      else
        export VQA_FILTER_INPUT_ROOT="${BUNDLE_ROOT}/data/shortcut_pipeline/union_mask"
      fi
    fi
    run_or_echo bash "${BUNDLE_ROOT}/scripts/run_qwen_visual_cue_filter.sh" vqa
    ;;
  vqa-build)
    VQA_GENERATED_TRAIN_JSON="${VQA_GENERATED_TRAIN_JSON:-${BUNDLE_ROOT}/data/shortcut_pipeline/vqa_v2_cmsv/train.json}"
    VQA_KEEP_JSON="${VQA_KEEP_JSON:-${BUNDLE_ROOT}/analysis/visual_cue_question_filter_qwen35_strict4/full_all_shards/merged/keep.json}"
    VQA_REMOVE_JSON="${VQA_REMOVE_JSON:-${BUNDLE_ROOT}/analysis/visual_cue_question_filter_qwen35_strict4/full_all_shards/merged/remove.json}"
    VQA_ORIGINAL_SPLIT_TRAIN="${VQA_ORIGINAL_SPLIT_TRAIN:-${BUNDLE_ROOT}/data/shortcut_pipeline/vqa_v2_cmsv/train.json}"
    VQA_ORIGINAL_SPLIT_VAL="${VQA_ORIGINAL_SPLIT_VAL:-${BUNDLE_ROOT}/data/shortcut_pipeline/vqa_v2_cmsv/val.json}"
    VQA_ORIGINAL_SPLIT_TEST="${VQA_ORIGINAL_SPLIT_TEST:-${BUNDLE_ROOT}/data/shortcut_pipeline/vqa_v2_cmsv/test.json}"
    if [[ -z "${VQA_MASK_DIR:-}" ]]; then
      if [[ -d "${BUNDLE_ROOT}/outputs/sam3_vqa_cmsv_union_masks/masks" ]]; then
        VQA_MASK_DIR="${BUNDLE_ROOT}/outputs/sam3_vqa_cmsv_union_masks/masks"
      else
        VQA_MASK_DIR="${BUNDLE_ROOT}/data/shortcut_pipeline/union_mask/masks"
      fi
    fi
    VQA_OUTPUT_JSON="${VQA_OUTPUT_JSON:-${BUNDLE_ROOT}/data/stage2/train_raw_mixed_qwenratio_oldbase_sam3_plus_vqa_nonumbermask.json}"
    VQA_OUTPUT_NPZ="${VQA_OUTPUT_NPZ:-${BUNDLE_ROOT}/data/stage2/patch_mask_analysis_train_raw_qwenkeep_sam3_nonumbermask_compat.npz}"
    VQA_SUMMARY_JSON="${VQA_SUMMARY_JSON:-${BUNDLE_ROOT}/data/stage2/train_raw_mixed_qwenratio_oldbase_sam3_plus_vqa_nonumbermask.summary.json}"
    check_path "${VQA_GENERATED_TRAIN_JSON}" "VQA generated-train JSON"
    check_path "${VQA_KEEP_JSON}" "Qwen keep JSON"
    check_path "${VQA_REMOVE_JSON}" "Qwen remove JSON"
    check_path "${VQA_ORIGINAL_SPLIT_TRAIN}" "VQA original-source train split"
    check_path "${VQA_ORIGINAL_SPLIT_VAL}" "VQA original-source val split"
    check_path "${VQA_ORIGINAL_SPLIT_TEST}" "VQA original-source test split"
    check_path "${VQA_MASK_DIR}" "VQA SAM3 mask directory"
    run_or_echo "${PYTHON_BIN}" "${BUNDLE_ROOT}/code/llava_sage/scripts2/build_qwenkeep_stage2_package.py" \
      --generated-train "${VQA_GENERATED_TRAIN_JSON}" \
      --keep-json "${VQA_KEEP_JSON}" \
      --remove-json "${VQA_REMOVE_JSON}" \
      --original-split "${VQA_ORIGINAL_SPLIT_TRAIN}" \
      --original-split "${VQA_ORIGINAL_SPLIT_VAL}" \
      --original-split "${VQA_ORIGINAL_SPLIT_TEST}" \
      --mask-dir "${VQA_MASK_DIR}" \
      --output-json "${VQA_OUTPUT_JSON}" \
      --output-mask-npz "${VQA_OUTPUT_NPZ}" \
      --summary-json "${VQA_SUMMARY_JSON}" \
      "$@"
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
