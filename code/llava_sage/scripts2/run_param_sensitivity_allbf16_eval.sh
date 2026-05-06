#!/usr/bin/env bash
set -euo pipefail

ROOT="/path/to/sage_repro_bundle"
OUT_ROOT="${OUT_ROOT:-/path/to/sage_repro_bundle/infer_result_param_sensitivity_epoch2_20260505_allbf16}"
DATA_PATH="${DATA_PATH:-/path/to/sage_repro_bundle/test_data/test_raw_with_shortcut_answer.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/root/train2014}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
XVERIFY_BATCH_SIZE="${XVERIFY_BATCH_SIZE:-32}"
EXPECTED_TOTAL="${EXPECTED_TOTAL:-12199}"

cd "$ROOT"

TAGS=(
  mask_div2_l1div2
  mask_div2_l1base
  mask_div2_l1x2
  mask_base_l1div2
  mask_base_l1base
  mask_base_l1x2
  mask_x2_l1div2
  mask_x2_l1base
  mask_x2_l1x2
)

declare -A CKPTS=(
  [mask_div2_l1div2]="/path/to/sage_repro_bundle/checkpoints/finetune_stage2_qwenratio_oldbase_sam3_qwenkeepmask_maskloss_div2_l1div2_nonumber_full_bs32_20260503_0942/checkpoint-3432"
  [mask_div2_l1base]="/path/to/sage_repro_bundle/checkpoints/finetune_stage2_qwenratio_oldbase_sam3_qwenkeepmask_maskloss_div2_nonumber_full_bs32_20260502_0236/checkpoint-3432"
  [mask_div2_l1x2]="/path/to/sage_repro_bundle/checkpoints/finetune_stage2_qwenratio_oldbase_sam3_qwenkeepmask_maskloss_div2_l1x2_nonumber_fullsched_bs32_20260504_0822/checkpoint-3432"
  [mask_base_l1div2]="/path/to/sage_repro_bundle/checkpoints/finetune_stage2_qwenratio_oldbase_sam3_qwenkeepmask_maskloss_base_l1div2_nonumber_full_bs32_20260503_1837/checkpoint-3432"
  [mask_base_l1base]="/path/to/sage_repro_bundle/checkpoints/finetune_stage2_qwenratio_oldbase_sam3_qwenkeepmask_nonumbermaskloss_full_bs32_20260425_175907/checkpoint-3432"
  [mask_base_l1x2]="/path/to/sage_repro_bundle/checkpoints/finetune_stage2_qwenratio_oldbase_sam3_qwenkeepmask_maskloss_base_l1x2_nonumber_full_bs32_20260430_030058/checkpoint-3432"
  [mask_x2_l1div2]="/path/to/sage_repro_bundle/checkpoints/finetune_stage2_qwenratio_oldbase_sam3_qwenkeepmask_maskloss_x2_l1div2_nonumber_full_bs32_20260503_0115/checkpoint-3432"
  [mask_x2_l1base]="/path/to/sage_repro_bundle/checkpoints/finetune_stage2_qwenratio_oldbase_sam3_qwenkeepmask_maskloss_x2_l1base_nonumber_fullsched_bs32_20260504_0911/checkpoint-3432"
  [mask_x2_l1x2]="/path/to/sage_repro_bundle/checkpoints/finetune_stage2_qwenratio_oldbase_sam3_qwenkeepmask_maskloss_x2_l1x2_nonumber_full_bs32_20260502_1356/checkpoint-3432"
)

json_path() {
  local tag="$1"
  echo "$OUT_ROOT/$tag/checkpoint-3432/test_raw_with_shortcut_answer.json"
}

metrics_path() {
  local tag="$1"
  echo "$OUT_ROOT/$tag/checkpoint-3432/test_raw_with_shortcut_answer.xverify_metrics.json"
}

valid_json() {
  local path="$1"
  python - "$path" "$EXPECTED_TOTAL" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
if not path.exists():
    raise SystemExit(1)
try:
    data = json.load(path.open())
except Exception:
    raise SystemExit(1)
if len(data) != expected:
    raise SystemExit(1)
qids = [x.get("question_id") for x in data]
empty = sum(1 for x in data if not str(x.get("model_pred") or "").strip())
if empty or len(qids) != len(set(qids)):
    raise SystemExit(1)
PY
}

run_infer() {
  local tag="$1"
  local gpu="$2"
  local out_dir="$OUT_ROOT/$tag"
  mkdir -p "$out_dir"
  echo "[infer:start] $tag gpu=$gpu"
  python scripts2/batch_infer.py \
    --model-path "${CKPTS[$tag]}" \
    --data-path "$DATA_PATH" \
    --has-gate true \
    --image-folder "$IMAGE_FOLDER" \
    --output-root "$out_dir" \
    --gpu "$gpu" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --temperature 0 \
    --num-beams 1 \
    --max-new-tokens 128 \
    --torch-dtype bfloat16 \
    --overwrite > "$out_dir/infer_bf16.log" 2>&1
  valid_json "$(json_path "$tag")"
  echo "[infer:done] $tag"
}

run_xverify() {
  local tag="$1"
  local gpu="$2"
  local input
  input="$(json_path "$tag")"
  echo "[xverify:start] $tag gpu=$gpu"
  python scripts2/eval_shortcut_metrics.py \
    --input-path "$input" \
    --gpu "$gpu" \
    --batch-size "$XVERIFY_BATCH_SIZE" \
    --overwrite > "$OUT_ROOT/$tag/checkpoint-3432/xverify.log" 2>&1
  test -f "$(metrics_path "$tag")"
  echo "[xverify:done] $tag"
}

launch_missing_infer() {
  local -a batch=()
  local tag
  for tag in "${TAGS[@]}"; do
    if valid_json "$(json_path "$tag")"; then
      echo "[infer:skip] $tag already complete"
    else
      batch+=("$tag")
    fi
  done

  local i=0
  while [ "$i" -lt "${#batch[@]}" ]; do
    local launched=0
    local pids=()
    while [ "$launched" -lt 4 ] && [ "$i" -lt "${#batch[@]}" ]; do
      run_infer "${batch[$i]}" "$launched" &
      pids+=("$!")
      i=$((i + 1))
      launched=$((launched + 1))
    done
    local pid
    for pid in "${pids[@]}"; do
      wait "$pid"
    done
  done
}

launch_missing_xverify() {
  local -a batch=()
  local tag
  for tag in "${TAGS[@]}"; do
    valid_json "$(json_path "$tag")"
    if [ -f "$(metrics_path "$tag")" ]; then
      echo "[xverify:skip] $tag already complete"
    else
      batch+=("$tag")
    fi
  done

  local i=0
  while [ "$i" -lt "${#batch[@]}" ]; do
    local launched=0
    local pids=()
    while [ "$launched" -lt 4 ] && [ "$i" -lt "${#batch[@]}" ]; do
      run_xverify "${batch[$i]}" "$launched" &
      pids+=("$!")
      i=$((i + 1))
      launched=$((launched + 1))
    done
    local pid
    for pid in "${pids[@]}"; do
      wait "$pid"
    done
  done
}

summarize() {
  python - "$OUT_ROOT" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
tags = [
    "mask_div2_l1div2",
    "mask_div2_l1base",
    "mask_div2_l1x2",
    "mask_base_l1div2",
    "mask_base_l1base",
    "mask_base_l1x2",
    "mask_x2_l1div2",
    "mask_x2_l1base",
    "mask_x2_l1x2",
]
label = {
    "mask_div2_l1div2": ("mask /2", "l1 /2"),
    "mask_div2_l1base": ("mask /2", "l1 base"),
    "mask_div2_l1x2": ("mask /2", "l1 x2"),
    "mask_base_l1div2": ("mask base", "l1 /2"),
    "mask_base_l1base": ("mask base", "l1 base"),
    "mask_base_l1x2": ("mask base", "l1 x2"),
    "mask_x2_l1div2": ("mask x2", "l1 /2"),
    "mask_x2_l1base": ("mask x2", "l1 base"),
    "mask_x2_l1x2": ("mask x2", "l1 x2"),
}
rows = []
for tag in tags:
    mp = root / tag / "checkpoint-3432" / "test_raw_with_shortcut_answer.xverify_metrics.json"
    jp = root / tag / "checkpoint-3432" / "test_raw_with_shortcut_answer.json"
    data = json.load(jp.open())
    metrics = json.load(mp.open())
    empty = sum(1 for x in data if not str(x.get("model_pred") or "").strip())
    acc = metrics["accuracy"]["stat_info"]
    sr = metrics["shortcut_rate"]["stat_info"]
    acc_by_type = metrics["accuracy"].get("by_answer_type", {})
    sr_by_type = metrics["shortcut_rate"].get("by_answer_type", {})
    rows.append((tag, label[tag][0], label[tag][1], acc, sr, acc_by_type, sr_by_type, len(data), empty))

def val(stat):
    return 100.0 * float(stat.get("accuracy", stat.get("Accuracy")))

def stat_num(stat):
    correct = stat.get("correct_num", stat.get("Correct_num", "?"))
    total = stat.get("total", stat.get("valid_num", stat.get("Valid_num", "?")))
    return f'{correct}/{total}'

print("# Overall Acc/SR (bf16, checkpoint-3432)")
print("| mask loss | l1 /2 | l1 base | l1 x2 |")
print("|---|---:|---:|---:|")
grid = { (mask, l1): (acc, sr) for _, mask, l1, acc, sr, *_ in rows }
for mask in ["mask /2", "mask base", "mask x2"]:
    cells = []
    for l1 in ["l1 /2", "l1 base", "l1 x2"]:
        acc, sr = grid[(mask, l1)]
        cells.append(f"{val(acc):.2f} / {val(sr):.2f}")
    print(f"| {mask} | " + " | ".join(cells) + " |")

print()
print("# Detailed Overall")
print("| tag | mask loss | l1 loss | Acc | SR | Acc count | SR count | total | empty |")
print("|---|---|---|---:|---:|---:|---:|---:|---:|")
for tag, mask, l1, acc, sr, acc_by_type, sr_by_type, total, empty in rows:
    print(f"| {tag} | {mask} | {l1} | {val(acc):.2f} | {val(sr):.2f} | {stat_num(acc)} | {stat_num(sr)} | {total} | {empty} |")

answer_types = ["yes/no", "number", "other"]

def norm_type(k):
    s = str(k).lower().strip()
    s = re.sub(r"[_ -]+", "/", s)
    if s in {"yes/no", "yesno", "yes/no question"}:
        return "yes/no"
    if s in {"number", "num"}:
        return "number"
    if s in {"other", "others"}:
        return "other"
    return s

print()
print("# By Answer Type")
print("| tag | yes/no Acc/SR | number Acc/SR | other Acc/SR |")
print("|---|---:|---:|---:|")
for tag, mask, l1, acc, sr, acc_by_type, sr_by_type, total, empty in rows:
    by_type = {}
    for k, v in acc_by_type.items():
        by_type.setdefault(norm_type(k), {})["accuracy"] = v
    for k, v in sr_by_type.items():
        by_type.setdefault(norm_type(k), {})["shortcut_rate"] = v
    cells = []
    for t in answer_types:
        a = by_type.get(t, {}).get("accuracy")
        s = by_type.get(t, {}).get("shortcut_rate")
        if a is None or s is None:
            cells.append("NA")
        else:
            cells.append(f"{val(a):.2f} / {val(s):.2f}")
    print(f"| {tag} | " + " | ".join(cells) + " |")
PY
}

echo "[root] $OUT_ROOT"
launch_missing_infer
launch_missing_xverify
summarize | tee "$OUT_ROOT/summary_bf16_epoch2.md"
echo "[done] summary: $OUT_ROOT/summary_bf16_epoch2.md"
