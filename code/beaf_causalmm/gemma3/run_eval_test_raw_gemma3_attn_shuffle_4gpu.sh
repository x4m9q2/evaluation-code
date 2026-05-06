#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$SCRIPT_DIR"

MODEL_PATH="${BUNDLE_ROOT}/models/Gemma-3-4B-IT"
QUESTION_FILE="${BUNDLE_ROOT}/outputs/beaf_causalmm/test_raw_llava.jsonl"
ANSWER_FILE="${BUNDLE_ROOT}/data/eval/test_raw_with_shortcut_answer.json"
IMAGE_FOLDER="${BUNDLE_ROOT}/data/playground_data/coco/train2014"
OUT_DIR="${BUNDLE_ROOT}/outputs/beaf_causalmm/gemma3_attn_shuffle_4gpu"
BASE_NAME="${BASE_NAME:-gemma3_attn_shuffle_test_raw_results}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
GAMMA="${GAMMA:-1.0}"
EPSILON="${EPSILON:-0.1}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1.0}"
ATTENTION_LAYER="${ATTENTION_LAYER:--1}"
DTYPE="${DTYPE:-bfloat16}"
SEED="${SEED:-42}"
LIMIT="${LIMIT:-}"
RESUME="${RESUME:-0}"
LOG_TO_FILE="${LOG_TO_FILE:-0}"

mkdir -p "$OUT_DIR"

pids=()
for gpu in 0 1 2 3; do
  output_file="$OUT_DIR/${BASE_NAME}.shard${gpu}.json"
  cmd=(
    python eval_test_raw_gemma3_attn_shuffle.py
    --model-path "$MODEL_PATH"
    --question-file "$QUESTION_FILE"
    --answer-file "$ANSWER_FILE"
    --image-folder "$IMAGE_FOLDER"
    --output-file "$output_file"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --gamma "$GAMMA"
    --epsilon "$EPSILON"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --seed "$SEED"
    --attention-layer "$ATTENTION_LAYER"
    --dtype "$DTYPE"
    --num-shards 4
    --shard-index "$gpu"
    --device-map auto
    --progress-position "$gpu"
  )

  if [[ -n "$LIMIT" ]]; then
    cmd+=(--limit "$LIMIT")
  fi
  if [[ "$RESUME" == "1" ]]; then
    cmd+=(--resume)
  fi

  echo "Launching shard $gpu on GPU $gpu -> $output_file"
  if [[ "$LOG_TO_FILE" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "$OUT_DIR/${BASE_NAME}.shard${gpu}.log" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" &
  fi
  pids+=($!)
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

python "$SCRIPT_DIR/merge_eval_test_raw_shards.py" \
  --question-file "$QUESTION_FILE" \
  --output-file "$OUT_DIR/${BASE_NAME}.json" \
  "$OUT_DIR/${BASE_NAME}.shard0.json.jsonl" \
  "$OUT_DIR/${BASE_NAME}.shard1.json.jsonl" \
  "$OUT_DIR/${BASE_NAME}.shard2.json.jsonl" \
  "$OUT_DIR/${BASE_NAME}.shard3.json.jsonl"

echo "Merged result: $OUT_DIR/${BASE_NAME}.json"
