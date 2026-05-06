#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_llava.sh"

cd "${BUNDLE_ROOT}"

ASSEMBLE_ADAPTER_PATH="${ASSEMBLE_ADAPTER_PATH:-${LLAVA_PRETRAIN_PROJECTOR}}"
ASSEMBLE_OUTPUT_PATH="${ASSEMBLE_OUTPUT_PATH:-${BUNDLE_ROOT}/checkpoints/llava_pretrain_gate_assembled}"
ASSEMBLE_MAX_SHARD_SIZE="${ASSEMBLE_MAX_SHARD_SIZE:-5GB}"
ASSEMBLE_FORCE_GATE="${ASSEMBLE_FORCE_GATE:-auto}"
ASSEMBLE_VISION_TOWER_CONFIG_PATH="${ASSEMBLE_VISION_TOWER_CONFIG_PATH:-}"
if [[ -z "${ASSEMBLE_VISION_TOWER_CONFIG_PATH}" ]]; then
  if command -v realpath >/dev/null 2>&1; then
    ASSEMBLE_VISION_TOWER_CONFIG_PATH="$(realpath --relative-to="${ASSEMBLE_OUTPUT_PATH}" "${LLAVA_VISION_TOWER}" 2>/dev/null || true)"
  fi
  ASSEMBLE_VISION_TOWER_CONFIG_PATH="${ASSEMBLE_VISION_TOWER_CONFIG_PATH:-${LLAVA_VISION_TOWER}}"
fi

check_path "${LLAVA_BASE_MODEL}" "LLaVA base model"
check_path "${LLAVA_VISION_TOWER}" "CLIP vision tower"
check_path "${ASSEMBLE_ADAPTER_PATH}" "projector/gate adapter"

run_or_echo "${PYTHON_BIN}" "${BUNDLE_ROOT}/code/data_tools/assemble_llava_base.py" \
  --base-model-path "${LLAVA_BASE_MODEL}" \
  --adapter-path "${ASSEMBLE_ADAPTER_PATH}" \
  --output-path "${ASSEMBLE_OUTPUT_PATH}" \
  --vision-tower-path "${LLAVA_VISION_TOWER}" \
  --vision-tower-config-path "${ASSEMBLE_VISION_TOWER_CONFIG_PATH}" \
  --max-shard-size "${ASSEMBLE_MAX_SHARD_SIZE}" \
  --force-gate "${ASSEMBLE_FORCE_GATE}"
