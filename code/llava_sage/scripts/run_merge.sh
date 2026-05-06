#!/bin/bash
set -euo pipefail

cd /path/to/local_scratch/LLaVA

export PYTHONPATH=/path/to/local_scratch/LLaVA:${PYTHONPATH:-}

python /path/to/local_scratch/eval/merge_llava_weights.py