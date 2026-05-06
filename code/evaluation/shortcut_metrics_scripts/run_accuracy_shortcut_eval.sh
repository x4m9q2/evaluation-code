#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_accuracy_shortcut_eval.sh --model-path /abs/checkpoint --run-name my_eval [options]

Required:
  --model-path PATH         LLaVA checkpoint to evaluate
  --run-name NAME           Output subdirectory name under outputs/

Optional:
  --gpus IDS                Comma-separated GPU ids for inference chunks, default: 0,1,2,3
  --xverify-gpu ID          GPU id for xVerify, default: 0
  --infer-batch-size N      Inference batch size, default: 16
  --infer-workers N         Inference dataloader workers, default: 4
  --xverify-batch-size N    xVerify batch size, default: 32
  --conv-mode MODE          Conversation mode, default: llava_v1
  --max-new-tokens N        Generation max_new_tokens, default: 128
  --temperature V           Generation temperature, default: 0
  --num-beams N             Generation num_beams, default: 1
  --question-file PATH      JSONL question file, default: data/test_raw_eval.jsonl
  --image-folder PATH       Image folder, default: assets/train2014
  --test-raw-json PATH      Anti-shortcut answers, default: data/test_raw.json
  --vqa-json PATH           Original VQA answers, default: data/vqa_train2014.json
  --xverify-model PATH      xVerify local model dir, default: assets/xVerify-0.5B-I
  --python BIN              Python executable, default: python
EOF
}

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH=""
RUN_NAME=""
GPUS="0,1,2,3"
XVERIFY_GPU="0"
INFER_BATCH_SIZE=16
INFER_WORKERS=4
XVERIFY_BATCH_SIZE=32
CONV_MODE="llava_v1"
MAX_NEW_TOKENS=128
TEMPERATURE=0
NUM_BEAMS=1
PYTHON_BIN="python"
QUESTION_FILE="$BUNDLE_ROOT/data/test_raw_eval.jsonl"
IMAGE_FOLDER="$BUNDLE_ROOT/assets/train2014"
TEST_RAW_JSON="$BUNDLE_ROOT/data/test_raw.json"
VQA_JSON="$BUNDLE_ROOT/data/vqa_train2014.json"
XVERIFY_MODEL="$BUNDLE_ROOT/assets/xVerify-0.5B-I"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --xverify-gpu) XVERIFY_GPU="$2"; shift 2 ;;
    --infer-batch-size) INFER_BATCH_SIZE="$2"; shift 2 ;;
    --infer-workers) INFER_WORKERS="$2"; shift 2 ;;
    --xverify-batch-size) XVERIFY_BATCH_SIZE="$2"; shift 2 ;;
    --conv-mode) CONV_MODE="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --num-beams) NUM_BEAMS="$2"; shift 2 ;;
    --question-file) QUESTION_FILE="$2"; shift 2 ;;
    --image-folder) IMAGE_FOLDER="$2"; shift 2 ;;
    --test-raw-json) TEST_RAW_JSON="$2"; shift 2 ;;
    --vqa-json) VQA_JSON="$2"; shift 2 ;;
    --xverify-model) XVERIFY_MODEL="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$MODEL_PATH" || -z "$RUN_NAME" ]]; then
  usage
  exit 1
fi

mkdir -p "$BUNDLE_ROOT/outputs/$RUN_NAME" "$BUNDLE_ROOT/tmp"
RUN_ROOT="$BUNDLE_ROOT/outputs/$RUN_NAME"
CHUNK_DIR="$RUN_ROOT/infer_chunks"
LOG_DIR="$RUN_ROOT/logs"
PRED_PATH="$RUN_ROOT/predictions_${RUN_NAME}.jsonl"
XVERIFY_INPUT="$RUN_ROOT/predictions_${RUN_NAME}.xverify_input.json"
SHORTCUT_INPUT="$RUN_ROOT/predictions_${RUN_NAME}.shortcut_xverify_input.json"
ACC_OUT="$RUN_ROOT/accuracy_xverify"
SHORTCUT_OUT="$RUN_ROOT/shortcut_xverify"

mkdir -p "$CHUNK_DIR" "$LOG_DIR" "$ACC_OUT" "$SHORTCUT_OUT"

if [[ ! -f "$QUESTION_FILE" ]]; then
  "$PYTHON_BIN" "$BUNDLE_ROOT/scripts/build_test_raw_eval.py" --src "$TEST_RAW_JSON" --out "$QUESTION_FILE"
fi

IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
NUM_CHUNKS="${#GPU_ARRAY[@]}"
if [[ "$NUM_CHUNKS" -lt 1 ]]; then
  echo "No GPUs provided via --gpus" >&2
  exit 1
fi

echo "[1/5] Running LLaVA inference on $NUM_CHUNKS chunk(s)..."
PIDS=()
for IDX in "${!GPU_ARRAY[@]}"; do
  GPU_ID="${GPU_ARRAY[$IDX]}"
  ANSWERS_FILE="$CHUNK_DIR/chunk${IDX}.jsonl"
  LOG_FILE="$LOG_DIR/infer_chunk${IDX}.log"
  CUDA_VISIBLE_DEVICES="$GPU_ID" OMP_NUM_THREADS=1 PYTHONPATH="$BUNDLE_ROOT" \
    "$PYTHON_BIN" -m llava.eval.model_vqa_loader \
      --model-path "$MODEL_PATH" \
      --image-folder "$IMAGE_FOLDER" \
      --question-file "$QUESTION_FILE" \
      --answers-file "$ANSWERS_FILE" \
      --conv-mode "$CONV_MODE" \
      --temperature "$TEMPERATURE" \
      --num_beams "$NUM_BEAMS" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --batch-size "$INFER_BATCH_SIZE" \
      --num-workers "$INFER_WORKERS" \
      --num-chunks "$NUM_CHUNKS" \
      --chunk-idx "$IDX" \
      >"$LOG_FILE" 2>&1 &
  PIDS+=("$!")
done

for PID in "${PIDS[@]}"; do
  wait "$PID"
done

echo "[2/5] Merging chunk predictions..."
"$PYTHON_BIN" "$BUNDLE_ROOT/scripts/merge_prediction_chunks.py" \
  --chunk-dir "$CHUNK_DIR" \
  --out "$PRED_PATH"

echo "[3/5] Building xVerify inputs..."
"$PYTHON_BIN" "$BUNDLE_ROOT/scripts/build_xverify_shortcut_data.py" \
  --pred-path "$PRED_PATH" \
  --vqa-path "$TEST_RAW_JSON" \
  --output-path "$XVERIFY_INPUT"
"$PYTHON_BIN" "$BUNDLE_ROOT/scripts/build_xverify_shortcut_data.py" \
  --pred-path "$PRED_PATH" \
  --vqa-path "$VQA_JSON" \
  --output-path "$SHORTCUT_INPUT"

echo "[4/5] Running xVerify accuracy..."
CUDA_VISIBLE_DEVICES="$XVERIFY_GPU" PYTHONPATH="$BUNDLE_ROOT/xverify_runtime" \
  "$PYTHON_BIN" "$BUNDLE_ROOT/xverify_runtime/run_local_xverify.py" \
    --data-path "$XVERIFY_INPUT" \
    --output-path "$ACC_OUT" \
    --model-path "$XVERIFY_MODEL" \
    --batch-size "$XVERIFY_BATCH_SIZE"

echo "[5/5] Running xVerify shortcut-rate..."
CUDA_VISIBLE_DEVICES="$XVERIFY_GPU" PYTHONPATH="$BUNDLE_ROOT/xverify_runtime" \
  "$PYTHON_BIN" "$BUNDLE_ROOT/xverify_runtime/run_local_xverify.py" \
    --data-path "$SHORTCUT_INPUT" \
    --output-path "$SHORTCUT_OUT" \
    --model-path "$XVERIFY_MODEL" \
    --batch-size "$XVERIFY_BATCH_SIZE"

echo
echo "Accuracy summary:"
"$PYTHON_BIN" "$BUNDLE_ROOT/scripts/print_xverify_metrics.py" "$ACC_OUT"
echo
echo "Shortcut summary:"
"$PYTHON_BIN" "$BUNDLE_ROOT/scripts/print_xverify_metrics.py" "$SHORTCUT_OUT"
