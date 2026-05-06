#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/internvl/run_internvl25_vqav2_testdev.sh [options]

Options:
  --model-path PATH         Local InternVL2.5-8B directory. Default: /root/models/InternVL2_5-8B
  --run-name NAME           Run/output name. Default: internvl25_8b_vqav2_testdev_YYYYmmdd_HHMMSS
  --ckpt NAME               Name used in VQAv2 answer/upload output paths. Default: run-name
  --gpus IDS                Comma-separated GPU ids for inference chunks. Default: 0,1,2,3
  --batch-size N            Inference batch size per GPU. Default: 8
  --num-workers N           Dataloader workers per GPU. Default: 4
  --max-new-tokens N        Generation max_new_tokens. Default: 32
  --num-beams N             Generation num_beams. Default: 1
  --temperature V           Generation temperature. Default: 0
  --input-size N            InternVL image size. Default: 448
  --max-num N               InternVL max dynamic tiles. Default: 12
  --question-file PATH      VQAv2 test-dev jsonl. Default: playground/data/eval/vqav2/llava_vqav2_mscoco_test-dev2015.jsonl
  --image-folder PATH       VQAv2 test2015 image directory. Default: playground/data/eval/vqav2/test2015
  --use-flash-attn          Enable use_flash_attn when loading InternVL
  --prompt-suffix TEXT      Optional suffix appended after each question
  --submit-evalai           Submit the generated JSON to EvalAI
  --wait                    Poll EvalAI until the submission finishes
  --token-file PATH         EvalAI token file. Default: ~/.evalai/token.json
  --public                  Submit as public on EvalAI
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUNDLE_ROOT="${REPO_ROOT}/vqav2_eval_bundle_20260403"

MODEL_PATH="/root/models/InternVL2_5-8B"
RUN_NAME="internvl25_8b_vqav2_testdev_$(date +%Y%m%d_%H%M%S)"
CKPT=""
GPUS="0,1,2,3"
BATCH_SIZE=8
NUM_WORKERS=4
MAX_NEW_TOKENS=32
NUM_BEAMS=1
TEMPERATURE=0
INPUT_SIZE=448
MAX_NUM=12
QUESTION_FILE="${REPO_ROOT}/playground/data/eval/vqav2/llava_vqav2_mscoco_test-dev2015.jsonl"
IMAGE_FOLDER="${REPO_ROOT}/playground/data/eval/vqav2/test2015"
USE_FLASH_ATTN=0
PROMPT_SUFFIX=""
SUBMIT_EVALAI=0
WAIT_FOR_EVALAI=0
TOKEN_FILE="${HOME}/.evalai/token.json"
IS_PUBLIC=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --ckpt) CKPT="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --num-beams) NUM_BEAMS="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --input-size) INPUT_SIZE="$2"; shift 2 ;;
    --max-num) MAX_NUM="$2"; shift 2 ;;
    --question-file) QUESTION_FILE="$2"; shift 2 ;;
    --image-folder) IMAGE_FOLDER="$2"; shift 2 ;;
    --use-flash-attn) USE_FLASH_ATTN=1; shift ;;
    --prompt-suffix) PROMPT_SUFFIX="$2"; shift 2 ;;
    --submit-evalai) SUBMIT_EVALAI=1; shift ;;
    --wait) WAIT_FOR_EVALAI=1; shift ;;
    --token-file) TOKEN_FILE="$2"; shift 2 ;;
    --public) IS_PUBLIC=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${CKPT}" ]]; then
  CKPT="${RUN_NAME}"
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model path not found: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ ! -f "${QUESTION_FILE}" ]]; then
  echo "Question file not found: ${QUESTION_FILE}" >&2
  exit 1
fi

if [[ ! -d "${IMAGE_FOLDER}" ]]; then
  echo "Image folder not found: ${IMAGE_FOLDER}" >&2
  exit 1
fi

RUN_ROOT="${BUNDLE_ROOT}/outputs/${RUN_NAME}"
CHUNK_DIR="${RUN_ROOT}/infer_chunks"
LOG_DIR="${RUN_ROOT}/logs"
MERGED_JSONL="${REPO_ROOT}/playground/data/eval/vqav2/answers/llava_vqav2_mscoco_test-dev2015/${CKPT}/merge.jsonl"
UPLOAD_JSON="${REPO_ROOT}/playground/data/eval/vqav2/answers_upload/llava_vqav2_mscoco_test-dev2015/${CKPT}.json"
SUBMIT_RESP="${RUN_ROOT}/evalai_submit_response.json"
POLL_LOG="${RUN_ROOT}/evalai_poll.log"

mkdir -p "${CHUNK_DIR}" "${LOG_DIR}" "$(dirname "${MERGED_JSONL}")" "$(dirname "${UPLOAD_JSON}")"

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
NUM_CHUNKS="${#GPU_ARRAY[@]}"
if [[ "${NUM_CHUNKS}" -lt 1 ]]; then
  echo "No GPUs provided via --gpus" >&2
  exit 1
fi

echo "[1/4] Running InternVL inference on ${NUM_CHUNKS} chunk(s)..."
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
    python "${REPO_ROOT}/scripts/internvl/eval_vqav2_internvl.py" \
      --model-path "${MODEL_PATH}" \
      --question-file "${QUESTION_FILE}" \
      --image-folder "${IMAGE_FOLDER}" \
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

echo "[2/4] Merging chunk predictions..."
python "${REPO_ROOT}/eval_accuracy_shortcut_bundle_20260402/scripts/merge_prediction_chunks.py" \
  --chunk-dir "${CHUNK_DIR}" \
  --out "${MERGED_JSONL}"

echo "[3/4] Building EvalAI upload JSON..."
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" python "${REPO_ROOT}/scripts/convert_vqav2_for_submission.py" \
  --dir "${REPO_ROOT}/playground/data/eval/vqav2" \
  --split llava_vqav2_mscoco_test-dev2015 \
  --ckpt "${CKPT}"

echo "Merged predictions: ${MERGED_JSONL}"
echo "EvalAI upload file: ${UPLOAD_JSON}"

if [[ "${SUBMIT_EVALAI}" != "1" ]]; then
  echo "[4/4] Skipping EvalAI submission."
  exit 0
fi

if [[ ! -f "${TOKEN_FILE}" ]]; then
  echo "EvalAI token file not found: ${TOKEN_FILE}" >&2
  exit 1
fi

echo "[4/4] Submitting to EvalAI..."
SUBMIT_ARGS=(
  python "${BUNDLE_ROOT}/submit_evalai.py"
  --file "${UPLOAD_JSON}"
  --token-file "${TOKEN_FILE}"
)
if [[ "${IS_PUBLIC}" == "1" ]]; then
  SUBMIT_ARGS+=(--public)
fi
"${SUBMIT_ARGS[@]}" | tee "${SUBMIT_RESP}"

if [[ "${WAIT_FOR_EVALAI}" != "1" ]]; then
  exit 0
fi

SUBMISSION_ID="$(
  python - "${SUBMIT_RESP}" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
for key in ("id", "submission", "submission_id"):
    value = data.get(key)
    if isinstance(value, int):
        print(value)
        break
else:
    raise SystemExit("Could not find submission id in EvalAI response")
PY
)"

echo "Polling EvalAI submission ${SUBMISSION_ID}..."
python "${BUNDLE_ROOT}/poll_evalai_submission.py" \
  --submission-id "${SUBMISSION_ID}" \
  --token-file "${TOKEN_FILE}" \
  --wait | tee "${POLL_LOG}"
