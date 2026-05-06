# SAGE Repro Bundle

This directory collects the SAGE reproduction code, scripts, comparison baselines, and selected JSON/NPZ metadata used in the current project.

It is organized so training/evaluation scripts use paths relative to this bundle. Large image files and model weights are not embedded; place them into the empty placeholder directories below before running.

- LLaVA-v1.5 base weights: `models/llava-v1.5-7b/`
- CLIP vision tower: `models/clip-vit-large-patch14-336/`
- xVerify weights for Acc/SR evaluation: `models/xVerify-0.5B-I/`
- SAM3 checkpoint: `models/sam3_ckpt/sam3.pt`
- Gemma 3 model weights: `models/Gemma-3-4B-IT/`
- SigLIP text encoder weights for the gate: `models/siglip-so400m-patch14-384/`
- Gemma image root: `data/playground_data/`
- LLaVA/SAGE image root: `data/images/`
- Optional Gemma-local xVerify mirror: `code/gemma_gate/x_verify/xVerify-0.5B-I/`

## Structure

- `code/gemma_gate/`: main Gemma 3 gate/SAGE training and evaluation implementation.
- `code/beaf_causalmm/`: BEAF/CausalMM-style comparison implementation for Gemma 3.
- `code/napo_gemma_debug/`: Gemma NaPO adaptation/debug scripts.
- `code/llava_sage/`: LLaVA-v1.5 SAGE implementation, including pretraining, mask-supervised finetuning, inference, POPE/BEAF hooks, and Acc/SR scripts.
- `third_party/napo_llava_ref/`: LLaVA NaPO comparison reference snapshot kept separate from the core SAGE code.
- `code/data_tools/`: shortcut, mask-generation, mask-filtering, and dataset-construction utilities.
- `code/evaluation/`: POPE, BEAF/CausalMM, xVerify, and shortcut metric code.
- `data/`: JSON/NPZ training and evaluation data, plus empty image placeholders.
- `models/`: empty model-weight placeholders.
- `scripts/`: bundle-level run scripts with relative data paths.
- `docs/`: manifest and RoPE notes.
- `checkpoints/`, `logs/`, `outputs/`: default local output directories.

Use the top-level `scripts/` directory for reproducible runs.

For the offline preprocessing chain that builds `mask -> Qwen filter -> final training JSON/NPZ`
artifacts for VQA, GQA, and VG, see
`docs/MASK_QWEN_PACKAGING_PIPELINE.md`.

For the LLaVA/SAGE submission code path, start with:

```bash
bash scripts/run_all_llava_sage_pipeline.sh
```

Top-level scripts are dry-run by default. Set `DRY_RUN=0` to execute long jobs. See `docs/LLAVA_SAGE_REPRO.md` for details.

## Training Parameter Stability

The LLaVA stage-2 SAGE scripts intentionally keep the original stable training
batch structure: ZeRO-1 bf16, `PER_DEVICE_TRAIN_BATCH_SIZE=32`,
`GRADIENT_ACCUMULATION_STEPS=1`, and 4 GPUs, giving an effective batch size of
128. Do not change the DeepSpeed stage, per-device batch size, gradient
accumulation, precision flags, or learning-rate schedule casually. Matching only
the effective batch size is not necessarily equivalent; unverified combinations
such as smaller micro-batches with larger gradient accumulation or ZeRO-2 may
trigger numerical instability or NaN losses.

If these parameters must be changed for memory reasons, first run a short smoke
test with frequent logging and verify that training loss, validation loss, and
saved checkpoints are normal before launching a full run.

For this bundle, use 20-step smoke tests instead of 1-step probes. One step only
checks that the process starts; 20 steps are enough to expose common NaN/Inf,
data-loading, optimizer, scheduler, and checkpoint-save problems while still
remaining cheap.

The filtered LLaVA pretraining JSON can be regenerated with:

```bash
bash scripts/run_build_pretrain_json.sh
```

Place the raw LLaVA mix file at `data/llava_stage1/llava_v1_5_mix665k.json`
before running that script. The raw file and intermediate generated JSON are
not included in the compressed release archive.

After LLaVA pretraining, assemble the saved `mm_projector.bin` adapter into a
directly loadable LLaVA checkpoint with:

```bash
DRY_RUN=0 bash scripts/run_assemble_llava_checkpoint.sh
```

The assembler injects projector weights and, when present, SAGE gate weights
into `models/llava-v1.5-7b/`. It writes the CLIP vision tower path as a relative
path in the saved config by default.

## Environment

Use one Python environment for both Gemma and LLaVA bundle scripts. The local
validated environment is named `.venv_gemma`; it uses Python 3.10, PyTorch
2.6.0+cu124, Transformers 4.51.3, and DeepSpeed 0.16.7.

Create it from the bundle root with:

```bash
python3 -m venv .venv_gemma
source .venv_gemma/bin/activate
pip install -r code/gemma_gate/gemma/requirements-py310.txt
```

Or create an equivalent conda environment:

```bash
conda env create -f environment.yml
conda activate sage-repro
```

`code/gemma_gate/requirements-extra.txt` currently contains `flash-attn`. Treat
it as an optional acceleration/backend dependency; install it only if your CUDA
toolchain can build or load the matching wheel. The validated smoke tests in
this bundle used the default SDPA/eager-compatible paths and do not require
Flash Attention 2 to start.

If you are using a pre-existing environment, verify the launcher stack before a
long run:

```bash
$PWD/.venv_gemma/bin/python -c "import torch, transformers, deepspeed; print(torch.__version__, transformers.__version__, deepspeed.__version__)"
```

Top-level Gemma scripts already launch with `${PYTHON_BIN}`. LLaVA and NaPO
LLaVA scripts also launch DeepSpeed through `${PYTHON_BIN} -m
deepspeed.launcher.runner`, so set `PYTHON_BIN` explicitly when using the
packaged environment:

```bash
export PYTHON_BIN=$PWD/.venv_gemma/bin/python
```

For direct ad-hoc commands outside the wrappers, also put the environment first
on `PATH` so `python`, `pip`, and `deepspeed` agree:

```bash
export PATH=$PWD/.venv_gemma/bin:$PATH
```

Alternatively install into your active environment from:

```bash
pip install -r code/gemma_gate/gemma/requirements.txt
```

Install `code/gemma_gate/requirements-extra.txt` only when you explicitly want
Flash Attention 2 support. The scripts default to SDPA/eager-compatible
attention paths. Override `ATTN_IMPLEMENTATION=eager` if your current
environment has SDPA issues.

## Pretraining

Default data:

- `data/pretrain_llava_v1_5_mix665k_single_noocr_max200_imageonly_strict_noocr.json`
- `data/playground_data/` as image root

Run:

```bash
cd /path/to/sage_repro_bundle
bash scripts/run_pretrain_gate.sh
```

Useful overrides:

```bash
PER_DEVICE_TRAIN_BATCH_SIZE=16 GRADIENT_ACCUMULATION_STEPS=2 SAVE_STEPS=2500 REPORT_TO=wandb bash scripts/run_pretrain_gate.sh
```

20-step gate smoke:

```bash
PYTHON_BIN=$PWD/.venv_gemma/bin/python CUDA_VISIBLE_DEVICES=0,1,2,3 \
RUN_NAME=smoke20_gemma_pretrain_gate_main_equiv \
OUTPUT_DIR=$PWD/checkpoints/smoke20_gemma_pretrain_gate_main_equiv \
LOG_FILE=$PWD/logs/smoke20_gemma_pretrain_gate_main_equiv.log \
MAX_STEPS=20 SAVE_STEPS=20 DATALOADER_NUM_WORKERS=0 REPORT_TO=none \
bash scripts/run_pretrain_gate.sh
```

Non-gated pretraining keeps the same data, optimizer, scheduler, batch, and
freeze settings, but disables the question-guided gate and L1 loss:

```bash
bash scripts/run_pretrain_nogate.sh
```

20-step non-gated smoke:

```bash
PYTHON_BIN=$PWD/.venv_gemma/bin/python CUDA_VISIBLE_DEVICES=0,1,2,3 \
RUN_NAME=smoke20_gemma_pretrain_nogate_main_equiv \
OUTPUT_DIR=$PWD/checkpoints/smoke20_gemma_pretrain_nogate_main_equiv \
LOG_FILE=$PWD/logs/smoke20_gemma_pretrain_nogate_main_equiv.log \
MAX_STEPS=20 SAVE_STEPS=20 DATALOADER_NUM_WORKERS=0 REPORT_TO=none \
bash scripts/run_pretrain_nogate.sh
```

## Stage-2 SFT

Stage-2 starts from `checkpoints/gemma3_4b_pretrain_gate_projector_l1_sdpa`.

```bash
bash scripts/run_stage2_sft_gate.sh
```

Default data:

- `data/stage2/train_raw_mixed_qwenratio_oldbase_sam3_plus_vqa.json`
- `data/stage2/patch_mask_analysis_train_raw_qwenkeep_sam3_compat.npz`
- `data/playground_data/coco/train2014`

20-step gate smoke:

```bash
PYTHON_BIN=$PWD/.venv_gemma/bin/python CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 \
RUN_NAME=smoke20_gemma_stage2_gate_main_equiv \
OUTPUT_DIR=$PWD/checkpoints/smoke20_gemma_stage2_gate_main_equiv \
LOG_FILE=$PWD/logs/smoke20_gemma_stage2_gate_main_equiv.log \
MAX_STEPS=20 DATALOADER_NUM_WORKERS=0 REPORT_TO=none \
EXTRA_ARGS='--save_strategy steps --save_steps 20' \
bash scripts/run_stage2_sft_gate.sh
```

Non-gated stage-2 SFT keeps the same data, optimizer, scheduler, and batch
settings, but disables the gate, L1 loss, and mask-patch loss:

```bash
bash scripts/run_stage2_sft_nogate.sh
```

20-step non-gated smoke, using a smoke non-gated pretraining checkpoint:

```bash
PYTHON_BIN=$PWD/.venv_gemma/bin/python CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 \
RUN_NAME=smoke20_gemma_stage2_nogate_main_equiv \
MODEL_ID=$PWD/checkpoints/smoke20_gemma_pretrain_nogate_main_equiv/checkpoint-20 \
OUTPUT_DIR=$PWD/checkpoints/smoke20_gemma_stage2_nogate_main_equiv \
LOG_FILE=$PWD/logs/smoke20_gemma_stage2_nogate_main_equiv.log \
MAX_STEPS=20 DATALOADER_NUM_WORKERS=0 REPORT_TO=none \
EXTRA_ARGS='--save_strategy steps --save_steps 20' \
bash scripts/run_stage2_sft_nogate.sh
```

## Evaluation

Run inference on `test_raw_with_shortcut_answer.json` without the extra short-answer system prompt:

```bash
BATCH_SIZE=16 \
bash scripts/run_eval_test_raw.sh
```

4-GPU sharded inference:

```bash
BATCH_SIZE=16 \
bash scripts/run_eval_test_raw_4gpu.sh
```

The merged prediction file is written under `outputs/infer_test_raw/<model_name>/`.

Then run xVerify Acc/SR:

```bash
INPUT_PATH=/path/to/merged_prediction.json \
bash scripts/run_xverify_metrics.sh
```

## NaPO Comparison

The Gemma NaPO entry uses the original Gemma model by default and disables the gate:

```bash
PER_DEVICE_TRAIN_BATCH_SIZE=16 \
NUM_TRAIN_EPOCHS=3 \
bash scripts/run_napo_shortcut.sh
```

20-step NaPO smoke:

```bash
PYTHON_BIN=$PWD/.venv_gemma/bin/python CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 \
RUN_NAME=smoke20_gemma_napo_main_equiv \
OUTPUT_DIR=$PWD/checkpoints/smoke20_gemma_napo_main_equiv \
LOG_FILE=$PWD/logs/smoke20_gemma_napo_main_equiv.log \
MAX_STEPS=20 DATALOADER_NUM_WORKERS=0 REPORT_TO=none \
EXTRA_ARGS='--save_strategy steps --save_steps 20' \
bash scripts/run_napo_shortcut.sh
```

Default data:

- `data/napo/train_raw_pos_neg_shortcut.json`
- `data/playground_data/coco/train2014`

The implemented loss path is NaPO-style `dyn_lq` with online reference-model log-probs. The LLaVA NaPO comparison code is kept under `third_party/napo_llava_ref/` as a reference snapshot rather than mixed into the core SAGE code path.

## BEAF / CausalMM Comparison

Run Gemma 3 CausalMM-style counterfactual decoding:

```bash
bash scripts/run_beaf_causalmm_eval.sh
```

This writes outputs under `outputs/beaf_causalmm/`.

Small end-to-end smoke:

```bash
PYTHON_BIN=$PWD/.venv_gemma/bin/python CUDA_VISIBLE_DEVICES=0 NUM_SHARDS=1 \
MAX_NEW_TOKENS=8 BATCH_SIZE=1 \
OUT_DIR=$PWD/outputs/beaf_causalmm_smoke2 \
OUTPUT_FILE=$PWD/outputs/beaf_causalmm_smoke2/gemma3_causalmm_test_raw_smoke2.json \
bash scripts/run_beaf_causalmm_eval.sh --limit 2
```

There is no separate file named `BEAF` in the recovered Gemma code. The included comparison code is the Gemma 3 CausalMM/BEAF-style counterfactual decoding bundle that was previously used for the BEAF/CausalMM comparison path.

For benchmark licensing hygiene, `data/beaf/` is shipped as a placeholder only.
Download the official BEAF Q/A JSON and image release yourself, then place them
under `data/beaf/beaf_qna.json` and `data/beaf/images/`.

## RoPE

See `docs/ROPE.md`. In short, this bundle relies on Hugging Face Gemma 3 RoPE and keeps patched forward logic aligned through `cache_position`; there is no separate custom RoPE implementation to enable.
