#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_llava.sh"

echo "Pipeline mode: ${PIPELINE_MODE:-check}"
echo "All steps execute by default. Override paths and training parameters with environment variables."
echo

bash "${BUNDLE_ROOT}/scripts/run_build_pretrain_json.sh"
bash "${BUNDLE_ROOT}/scripts/run_llava_pretrain_gate.sh"
bash "${BUNDLE_ROOT}/scripts/run_assemble_llava_checkpoint.sh"
bash "${BUNDLE_ROOT}/scripts/run_llava_stage2_mask_sft.sh"
bash "${BUNDLE_ROOT}/scripts/run_llava_stage2_mask_sft_nogate.sh"
bash "${BUNDLE_ROOT}/scripts/run_pope_eval.sh"
bash "${BUNDLE_ROOT}/scripts/run_beaf_eval.sh"
bash "${BUNDLE_ROOT}/scripts/run_cmsv_causalmm_llava.sh"
bash "${BUNDLE_ROOT}/scripts/run_napo_llava.sh"
bash "${BUNDLE_ROOT}/scripts/run_mask_generation_and_filtering.sh"
