#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_llava.sh"

INSTANCES_JSON="${INSTANCES_JSON:-${BUNDLE_ROOT}/annotations/instances_train2014.json}"
VQA_QUESTIONS_JSON="${VQA_QUESTIONS_JSON:-${BUNDLE_ROOT}/data/detect-shortcuts/data/vqa2/v2_OpenEnded_mscoco_train2014_questions.json}"
VQA_ANNOTATIONS_JSON="${VQA_ANNOTATIONS_JSON:-${BUNDLE_ROOT}/data/detect-shortcuts/data/vqa2/v2_mscoco_train2014_annotations.json}"
SHORTCUT_CODE_DIR="${SHORTCUT_CODE_DIR:-${BUNDLE_ROOT}/code/shortcut_pipeline}"
SHORTCUT_PIPELINE_DIR="${SHORTCUT_PIPELINE_DIR:-${BUNDLE_ROOT}/data/shortcut_pipeline}"
SHORTCUT_GMINER="${SHORTCUT_GMINER:-${SHORTCUT_CODE_DIR}/bin/GMiner}"
SHORTCUT_MATCHER_BIN="${SHORTCUT_MATCHER_BIN:-${SHORTCUT_CODE_DIR}/bin/cuda}"
SAM3_BPE_PATH="${SAM3_BPE_PATH:-${BUNDLE_ROOT}/code/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz}"
SAM3_DEVICE="${SAM3_DEVICE:-cuda}"
SAM3_BATCH_SIZE="${SAM3_BATCH_SIZE:-32}"
SAM3_RESOLUTION="${SAM3_RESOLUTION:-1008}"
SAM3_SCORE_THRESH="${SAM3_SCORE_THRESH:-0.5}"
STAGE2_NUM_SHARDS="${STAGE2_NUM_SHARDS:-1}"
STAGE2_SHARD_INDEX="${STAGE2_SHARD_INDEX:-0}"
SAM3_SHARD_CUDA_VISIBLE_DEVICES="${SAM3_SHARD_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}"

SUPPORT="${SUPPORT:-0.02}"
MAX_LENGTH="${MAX_LENGTH:-4}"
MIN_CONF="${MIN_CONF:-0.3}"
MOST_COMMON_ANSWERS="${MOST_COMMON_ANSWERS:-200}"
MATCHER_GPUS="${MATCHER_GPUS:-${CUDA_VISIBLE_DEVICES}}"
MATCHER_BATCH_SIZE="${MATCHER_BATCH_SIZE:-262144}"
STAGE1_LIMIT="${STAGE1_LIMIT:-0}"
STAGE2_LIMIT="${STAGE2_LIMIT:--1}"

INSTANCES_JSON="$(resolve_bundle_path "${INSTANCES_JSON}")"
VQA_QUESTIONS_JSON="$(resolve_bundle_path "${VQA_QUESTIONS_JSON}")"
VQA_ANNOTATIONS_JSON="$(resolve_bundle_path "${VQA_ANNOTATIONS_JSON}")"
SHORTCUT_CODE_DIR="$(resolve_bundle_path "${SHORTCUT_CODE_DIR}")"
SHORTCUT_PIPELINE_DIR="$(resolve_bundle_path "${SHORTCUT_PIPELINE_DIR}")"
SHORTCUT_GMINER="$(resolve_bundle_path "${SHORTCUT_GMINER}")"
SHORTCUT_MATCHER_BIN="$(resolve_bundle_path "${SHORTCUT_MATCHER_BIN}")"
SAM3_BPE_PATH="$(resolve_bundle_path "${SAM3_BPE_PATH}")"
VQA_TRAIN2014_IMAGE_ROOT="$(resolve_bundle_path "${VQA_TRAIN2014_IMAGE_ROOT}")"
SAM3_CHECKPOINT="$(resolve_bundle_path "${SAM3_CHECKPOINT}")"

echo "Stage 1 shortcut pipeline"
echo "Output root: ${SHORTCUT_PIPELINE_DIR}"
echo

run_mask_shards() {
  local mask_script="$1"
  local qa_jsonl="$2"
  local mapping_json="$3"
  local union_mask_root="$4"
  local image_root="$5"
  local checkpoint_path="$6"
  shift 6
  local -a checkpoint_args=("$@")

  if [[ "${STAGE2_NUM_SHARDS}" -le 1 ]]; then
    local -a single_cmd=(
      "${PYTHON_BIN}" "${mask_script}"
      --qa-jsonl "${qa_jsonl}"
      --mapping-json "${mapping_json}"
      --output-dir "${union_mask_root}"
      --image-root "${image_root}"
      --batch-size "${SAM3_BATCH_SIZE}"
      --resolution "${SAM3_RESOLUTION}"
      --score-thresh "${SAM3_SCORE_THRESH}"
      --device "${SAM3_DEVICE}"
      --num-shards 1
      --shard-index "${STAGE2_SHARD_INDEX}"
    )
    if [[ "${#checkpoint_args[@]}" -gt 0 ]]; then
      single_cmd+=("${checkpoint_args[@]}")
    fi
    run_or_echo "${single_cmd[@]}"
    return
  fi

  if [[ "${SAM3_DEVICE}" != cuda* ]]; then
    echo "[error] multi-GPU mask generation requires SAM3_DEVICE to be a CUDA device, got: ${SAM3_DEVICE}" >&2
    return 2
  fi

  local -a shard_gpus=()
  IFS=',' read -r -a shard_gpus <<< "${SAM3_SHARD_CUDA_VISIBLE_DEVICES}"
  if [[ "${#shard_gpus[@]}" -lt "${STAGE2_NUM_SHARDS}" ]]; then
    echo "[error] STAGE2_NUM_SHARDS=${STAGE2_NUM_SHARDS} but only ${#shard_gpus[@]} GPUs were provided in SAM3_SHARD_CUDA_VISIBLE_DEVICES=${SAM3_SHARD_CUDA_VISIBLE_DEVICES}" >&2
    return 2
  fi

  echo "Stage 2 SAM3 mask sharding"
  echo "Shards: ${STAGE2_NUM_SHARDS}"
  echo "GPUs:   ${SAM3_SHARD_CUDA_VISIBLE_DEVICES}"
  echo

  local -a pids=()
  local shard=0
  local shard_gpu=""
  for ((shard = 0; shard < STAGE2_NUM_SHARDS; shard++)); do
    shard_gpu="${shard_gpus[$shard]}"
    local -a shard_cmd=(
      env
      CUDA_VISIBLE_DEVICES="${shard_gpu}"
      "${PYTHON_BIN}" "${mask_script}"
      --qa-jsonl "${qa_jsonl}"
      --mapping-json "${mapping_json}"
      --output-dir "${union_mask_root}"
      --image-root "${image_root}"
      --batch-size "${SAM3_BATCH_SIZE}"
      --resolution "${SAM3_RESOLUTION}"
      --score-thresh "${SAM3_SCORE_THRESH}"
      --device cuda
      --num-shards "${STAGE2_NUM_SHARDS}"
      --shard-index "${shard}"
    )
    if [[ "${#checkpoint_args[@]}" -gt 0 ]]; then
      shard_cmd+=("${checkpoint_args[@]}")
    fi

    echo "[run/bg][shard ${shard}] ${shard_cmd[*]}"
    "${shard_cmd[@]}" &
    pids+=("$!")
  done

  local failed=0
  for ((shard = 0; shard < ${#pids[@]}; shard++)); do
    if ! wait "${pids[$shard]}"; then
      echo "[error] SAM3 mask shard ${shard} failed" >&2
      failed=1
    fi
  done

  if [[ "${failed}" != "0" ]]; then
    return 1
  fi
}

check_sam3_runtime() {
  local -a check_cmd=(
    "${PYTHON_BIN}" -c
    "import ftfy, iopath, pycocotools, timm; from sam3 import build_sam3_image_model"
  )
  run_or_echo "${check_cmd[@]}"
}

sam3_ckpt_args() {
  local checkpoint_path="$1"
  local -n out_args_ref="$2"
  out_args_ref=()
  if [[ -n "${checkpoint_path}" && -f "${checkpoint_path}" ]]; then
    out_args_ref+=(--checkpoint-path "${checkpoint_path}" --no-load-from-hf)
  else
    echo "[warn] SAM3 checkpoint not found at ${checkpoint_path}; falling back to Hugging Face download" >&2
  fi
}

check_path "${INSTANCES_JSON}" "COCO instances_train2014"
check_path "${VQA_QUESTIONS_JSON}" "VQA train questions"
check_path "${VQA_ANNOTATIONS_JSON}" "VQA train annotations"
check_path "${SHORTCUT_GMINER}" "GMiner binary"
check_path "${SHORTCUT_MATCHER_BIN}" "shortcut matcher binary"

CMD=(
  "${PYTHON_BIN}" "${SHORTCUT_CODE_DIR}/run_full_pipeline.py"
  --instances-json "${INSTANCES_JSON}"
  --questions-json "${VQA_QUESTIONS_JSON}"
  --annotations-json "${VQA_ANNOTATIONS_JSON}"
  --gminer-path "${SHORTCUT_GMINER}"
  --matcher-binary "${SHORTCUT_MATCHER_BIN}"
  --support "${SUPPORT}"
  --max-length "${MAX_LENGTH}"
  --min-conf "${MIN_CONF}"
  --most-common-answers "${MOST_COMMON_ANSWERS}"
  --matcher-gpus "${MATCHER_GPUS}"
  --matcher-batch-size "${MATCHER_BATCH_SIZE}"
  --work-dir "${SHORTCUT_PIPELINE_DIR}"
)

if [[ "${STAGE1_LIMIT}" -gt 0 ]]; then
  CMD+=(--limit "${STAGE1_LIMIT}")
fi

run_or_echo "${CMD[@]}"

STAGE2_MERGED_JSON="${STAGE2_MERGED_JSON:-${SHORTCUT_PIPELINE_DIR}/gqa_merged_output_with_answer_type.json}"
if [[ -z "${STAGE2_QUESTIONS_JSON:-}" ]]; then
  if [[ "${STAGE1_LIMIT}" -gt 0 ]]; then
    STAGE2_QUESTIONS_JSON="${SHORTCUT_PIPELINE_DIR}/train_questions.json"
  else
    STAGE2_QUESTIONS_JSON="${VQA_QUESTIONS_JSON}"
  fi
fi
STAGE2_INPUT_JSON="${STAGE2_INPUT_JSON:-${SHORTCUT_PIPELINE_DIR}/cross_modality_qa_input.json}"
STAGE2_QA_JSONL="${STAGE2_QA_JSONL:-${SHORTCUT_PIPELINE_DIR}/cross_modality_qa_questions.jsonl}"
STAGE2_MAPPING_JSON="${STAGE2_MAPPING_JSON:-${SHORTCUT_PIPELINE_DIR}/cross_modality_qa_mapping.json}"
STAGE2_UNION_MASK_ROOT="${STAGE2_UNION_MASK_ROOT:-${SHORTCUT_PIPELINE_DIR}/union_mask}"
STAGE2_MASK_ROOT="${STAGE2_MASK_ROOT:-${SHORTCUT_PIPELINE_DIR}/output_mask}"

STAGE2_MERGED_JSON="$(resolve_bundle_path "${STAGE2_MERGED_JSON}")"
STAGE2_QUESTIONS_JSON="$(resolve_bundle_path "${STAGE2_QUESTIONS_JSON}")"
STAGE2_INPUT_JSON="$(resolve_bundle_path "${STAGE2_INPUT_JSON}")"
STAGE2_QA_JSONL="$(resolve_bundle_path "${STAGE2_QA_JSONL}")"
STAGE2_MAPPING_JSON="$(resolve_bundle_path "${STAGE2_MAPPING_JSON}")"
STAGE2_UNION_MASK_ROOT="$(resolve_bundle_path "${STAGE2_UNION_MASK_ROOT}")"
STAGE2_MASK_ROOT="$(resolve_bundle_path "${STAGE2_MASK_ROOT}")"

echo
echo "Stage 1 post-step: prepare stage-2 masks"
echo "Merged: ${STAGE2_MERGED_JSON}"
echo "Ques:   ${STAGE2_QUESTIONS_JSON}"
echo "Input:  ${STAGE2_INPUT_JSON}"
echo "QA:     ${STAGE2_QA_JSONL}"
echo "Map:    ${STAGE2_MAPPING_JSON}"
echo "Union:  ${STAGE2_UNION_MASK_ROOT}"
echo "Masks:  ${STAGE2_MASK_ROOT}"
echo

check_path "${VQA_TRAIN2014_IMAGE_ROOT}" "COCO train2014 image root"
check_path "${SAM3_BPE_PATH}" "SAM3 tokenizer BPE"

STAGE2_PREPARE_CMD=(
  "${PYTHON_BIN}" "${SHORTCUT_CODE_DIR}/prepare_stage2_inputs.py"
  --merged-json "${STAGE2_MERGED_JSON}"
  --questions-json "${STAGE2_QUESTIONS_JSON}"
  --output-json "${STAGE2_INPUT_JSON}"
  --qa-jsonl "${STAGE2_QA_JSONL}"
  --mapping-json "${STAGE2_MAPPING_JSON}"
  --limit "${STAGE2_LIMIT}"
)
run_or_echo "${STAGE2_PREPARE_CMD[@]}"

echo "Stage 2 preflight: verify SAM3 runtime imports"
check_sam3_runtime

declare -a SAM3_CKPT_ARGS=()
sam3_ckpt_args "${SAM3_CHECKPOINT}" SAM3_CKPT_ARGS

run_mask_shards \
  "${BUNDLE_ROOT}/code/sam3/scripts/generate_union_masks_from_mapping.py" \
  "${STAGE2_QA_JSONL}" \
  "${STAGE2_MAPPING_JSON}" \
  "${STAGE2_UNION_MASK_ROOT}" \
  "${VQA_TRAIN2014_IMAGE_ROOT}" \
  "${SAM3_CHECKPOINT}" \
  "${SAM3_CKPT_ARGS[@]}"

STAGE2_APPLY_CMD=(
  "${PYTHON_BIN}" "${SHORTCUT_CODE_DIR}/apply_union_masks_to_images.py"
  --qa-jsonl "${STAGE2_QA_JSONL}"
  --mask-dir "${STAGE2_UNION_MASK_ROOT}/masks"
  --image-root "${VQA_TRAIN2014_IMAGE_ROOT}"
  --output-dir "${STAGE2_MASK_ROOT}"
)
run_or_echo "${STAGE2_APPLY_CMD[@]}"
