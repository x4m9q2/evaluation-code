#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[warn] scripts/run_beaf_causalmm_eval.sh is a legacy alias." >&2
echo "[warn] CausalMM is evaluated on CMSV test splits, not on BEAF. Use scripts/run_cmsv_causalmm_gemma.sh." >&2

exec bash "${SCRIPT_DIR}/run_cmsv_causalmm_gemma.sh" "$@"
