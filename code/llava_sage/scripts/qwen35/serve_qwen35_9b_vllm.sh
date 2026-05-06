#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-.venv_qwen35_vllm}"
MODEL_DIR="${MODEL_DIR:-models/Qwen3.5-9B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-9B}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"

if [[ ! -x "${VENV_DIR}/bin/vllm" ]]; then
  echo "vLLM is not installed in ${VENV_DIR}. Run scripts/qwen35/setup_qwen35_vllm_env.sh first." >&2
  exit 1
fi

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  echo "Model files were not found under ${MODEL_DIR}. Run scripts/qwen35/download_qwen35_9b_modelscope.sh first." >&2
  exit 1
fi

source "${SCRIPT_DIR}/_common.sh"
qwen35_prepend_runtime_libs "${VENV_DIR}"

export CUDA_VISIBLE_DEVICES
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

exec "${VENV_DIR}/bin/vllm" serve "${MODEL_DIR}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --reasoning-parser qwen3 \
  "$@"
