#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/path/to/sage_repro_bundle}"
MODEL_PATH="${MODEL_PATH:-/path/to/sage_repro_bundle/checkpoints/finetune_train_raw_plus_vqav2_20260402_052645}"
DATA_PATH="${DATA_PATH:-/path/to/sage_repro_bundle/test_data/train_raw_mask_answerable_with_shortcut_answer.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-data/images/coco/train2014}"
PATCH_MASK_ANALYSIS_PATH="${PATCH_MASK_ANALYSIS_PATH:-/path/to/sage_repro_bundle/patch_mask_analysis_output_mask_coco_seg_direct_llava_pad336_patch14.npz}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-/path/to/sage_repro_bundle/infer_result_mask_answerable_gate_suppress}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
NUM_CHUNKS="${NUM_CHUNKS:-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
HAS_GATE="${HAS_GATE:-true}"
RATIOS="${RATIOS:-0,0.25,0.5,0.75,1.0}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

IFS=',' read -r -a RATIO_ARRAY <<< "${RATIOS}"
for RATIO in "${RATIO_ARRAY[@]}"; do
    SAFE_RATIO="${RATIO//./p}"
    OUTPUT_ROOT="${OUTPUT_ROOT_BASE}_r${SAFE_RATIO}"
    echo "[start] ratio=${RATIO} output_root=${OUTPUT_ROOT}"
    .venv_unifolm/bin/python "${REPO_ROOT}/scripts2/batch_infer.py" \
        --model-path "${MODEL_PATH}" \
        --data-path "${DATA_PATH}" \
        --has-gate "${HAS_GATE}" \
        --image-folder "${IMAGE_FOLDER}" \
        --output-root "${OUTPUT_ROOT}" \
        --gpu "${GPU_LIST}" \
        --num-chunks "${NUM_CHUNKS}" \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" \
        --patch-mask-analysis-path "${PATCH_MASK_ANALYSIS_PATH}" \
        --gate-patch-suppress-ratio "${RATIO}" \
        --overwrite
    echo "[done] ratio=${RATIO}"
done
