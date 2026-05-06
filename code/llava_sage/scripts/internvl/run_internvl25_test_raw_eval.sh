#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/internvl/run_internvl25_test_raw_eval.sh [options]

Options:
  --model-path PATH         Local InternVL2.5-8B directory. Default: /root/models/InternVL2_5-8B
  --run-name NAME           Output run name. Default: internvl25_8b_test_raw_YYYYmmdd_HHMMSS
  --gpus IDS                Comma-separated GPU ids for inference chunks. Default: 0,1,2,3
  --xverify-gpu ID          GPU id for xVerify. Default: 0
  --batch-size N            InternVL inference batch size per GPU. Default: 8
  --num-workers N           Dataloader workers per GPU. Default: 4
  --max-new-tokens N        Generation max_new_tokens. Default: 128
  --num-beams N             Generation num_beams. Default: 1
  --temperature V           Generation temperature. Default: 0
  --input-size N            InternVL image size. Default: 448
  --max-num N               InternVL max dynamic tiles. Default: 12
  --use-flash-attn          Enable use_flash_attn when loading InternVL
  --prompt-suffix TEXT      Optional suffix appended after each question
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUNDLE_ROOT="${REPO_ROOT}/eval_accuracy_shortcut_bundle_20260402"

MODEL_PATH="/root/models/InternVL2_5-8B"
RUN_NAME="internvl25_8b_test_raw_$(date +%Y%m%d_%H%M%S)"
GPUS="0,1,2,3"
XVERIFY_GPU="0"
BATCH_SIZE=8
NUM_WORKERS=4
MAX_NEW_TOKENS=128
NUM_BEAMS=1
TEMPERATURE=0
INPUT_SIZE=448
MAX_NUM=12
USE_FLASH_ATTN=0
PROMPT_SUFFIX=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --xverify-gpu) XVERIFY_GPU="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --num-beams) NUM_BEAMS="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --input-size) INPUT_SIZE="$2"; shift 2 ;;
    --max-num) MAX_NUM="$2"; shift 2 ;;
    --use-flash-attn) USE_FLASH_ATTN=1; shift ;;
    --prompt-suffix) PROMPT_SUFFIX="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model path not found: ${MODEL_PATH}" >&2
  exit 1
fi

RUN_ROOT="${BUNDLE_ROOT}/outputs/${RUN_NAME}"
CHUNK_DIR="${RUN_ROOT}/infer_chunks"
LOG_DIR="${RUN_ROOT}/logs"
PRED_PATH="${RUN_ROOT}/predictions_${RUN_NAME}.jsonl"
XVERIFY_INPUT="${RUN_ROOT}/predictions_${RUN_NAME}.xverify_input.json"
SHORTCUT_INPUT="${RUN_ROOT}/predictions_${RUN_NAME}.shortcut_xverify_input.json"
ACC_OUT="${RUN_ROOT}/accuracy_xverify"
SHORTCUT_OUT="${RUN_ROOT}/shortcut_xverify"

mkdir -p "${CHUNK_DIR}" "${LOG_DIR}" "${ACC_OUT}" "${SHORTCUT_OUT}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
NUM_CHUNKS="${#GPU_ARRAY[@]}"
if [[ "${NUM_CHUNKS}" -lt 1 ]]; then
  echo "No GPUs provided via --gpus" >&2
  exit 1
fi

echo "[1/5] Running InternVL inference on ${NUM_CHUNKS} chunk(s)..."
PIDS=()
for IDX in "${!GPU_ARRAY[@]}"; do
  GPU_ID="${GPU_ARRAY[$IDX]}"
  ANSWERS_FILE="${CHUNK_DIR}/chunk${IDX}.jsonl"
  LOG_FILE="${LOG_DIR}/infer_chunk${IDX}.log"
  EXTRA_ARGS=()
  if [[ "${USE_FLASH_ATTN}" == "1" ]]; then
    EXTRA_ARGS+=(--use-flash-attn)
  fi
  if [[ -n "${PROMPT_SUFFIX}" ]]; then
    EXTRA_ARGS+=(--prompt-suffix "${PROMPT_SUFFIX}")
  fi
  CUDA_VISIBLE_DEVICES="${GPU_ID}" OMP_NUM_THREADS=1 PYTHONPATH="${REPO_ROOT}" \
    python "${REPO_ROOT}/scripts/internvl/eval_test_raw_internvl.py" \
      --model-path "${MODEL_PATH}" \
      --question-file "${REPO_ROOT}/test_raw.json" \
      --image-folder /root/train2014 \
      --answers-file "${ANSWERS_FILE}" \
      --num-chunks "${NUM_CHUNKS}" \
      --chunk-idx "${IDX}" \
      --batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --num-beams "${NUM_BEAMS}" \
      --temperature "${TEMPERATURE}" \
      --input-size "${INPUT_SIZE}" \
      --max-num "${MAX_NUM}" \
      "${EXTRA_ARGS[@]}" \
      >"${LOG_FILE}" 2>&1 &
  PIDS+=("$!")
done

for PID in "${PIDS[@]}"; do
  wait "${PID}"
done

echo "[2/5] Merging chunk predictions..."
python "${BUNDLE_ROOT}/scripts/merge_prediction_chunks.py" \
  --chunk-dir "${CHUNK_DIR}" \
  --out "${PRED_PATH}"

echo "[3/5] Building xVerify inputs..."
python "${REPO_ROOT}/build_xverify_shortcut_data.py" \
  --pred-path "${PRED_PATH}" \
  --vqa-path "${REPO_ROOT}/test_raw.json" \
  --output-path "${XVERIFY_INPUT}"
python "${REPO_ROOT}/build_xverify_shortcut_data.py" \
  --pred-path "${PRED_PATH}" \
  --vqa-path "${REPO_ROOT}/vqa_train2014.json" \
  --output-path "${SHORTCUT_INPUT}"

echo "[4/5] Running xVerify accuracy..."
CUDA_VISIBLE_DEVICES="${XVERIFY_GPU}" PYTHONPATH="${BUNDLE_ROOT}/xverify_runtime" \
  python "${BUNDLE_ROOT}/xverify_runtime/run_local_xverify.py" \
    --data-path "${XVERIFY_INPUT}" \
    --output-path "${ACC_OUT}" \
    --model-path "${REPO_ROOT}/x_verify/xVerify-0.5B-I" \
    --batch-size 32

echo "[5/5] Running xVerify shortcut-rate..."
CUDA_VISIBLE_DEVICES="${XVERIFY_GPU}" PYTHONPATH="${BUNDLE_ROOT}/xverify_runtime" \
  python "${BUNDLE_ROOT}/xverify_runtime/run_local_xverify.py" \
    --data-path "${SHORTCUT_INPUT}" \
    --output-path "${SHORTCUT_OUT}" \
    --model-path "${REPO_ROOT}/x_verify/xVerify-0.5B-I" \
    --batch-size 32

echo
echo "Accuracy summary:"
python "${BUNDLE_ROOT}/scripts/print_xverify_metrics.py" "${ACC_OUT}"
echo
echo "Shortcut summary:"
python "${BUNDLE_ROOT}/scripts/print_xverify_metrics.py" "${SHORTCUT_OUT}"
