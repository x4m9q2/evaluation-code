# LLaVA/SAGE Reproducibility Guide

This bundle contains the code needed to reproduce the LLaVA-v1.5 based SAGE experiments, the NaPO comparison wrappers, mask generation/filtering utilities, and BEAF/POPE/Acc-SR evaluation.

All top-level scripts under `scripts/` resolve paths relative to the bundle root. Do not run the copied historical scripts directly unless you have checked their default paths.

## Included Code

- `code/llava_sage/`: LLaVA-v1.5 code with SAGE gate, patch-mask supervision, pretraining, stage-2 finetuning, inference, and shortcut metrics scripts.
- `third_party/napo_llava_ref/`: NaPO LLaVA reference snapshot, including DPO/NaPO training scripts and utilities. Large datasets/checkpoints are excluded.
- `code/data_tools/`: shortcut mining, mask analysis, mask filtering, SAM3 mask package construction, and data conversion utilities.
- `code/evaluation/pope_beaf_gate/`: POPE evaluation and gate-attention visualization scripts used in the LLaVA/SAGE experiments.
- `code/evaluation/causalmm_llava/`: CausalMM/BEAF-style LLaVA comparison code copied from the inspected `CausalMM-main` tree.
- `code/evaluation/x_verify/`: xVerify wrapper/source files for Acc/SR evaluation. The xVerify model weights are not included.
- `code/evaluation/shortcut_metrics_scripts/`: compact Acc/SR metric wrappers from the shortcut evaluation bundle.

## External Assets

The package intentionally excludes model weights, checkpoints, raw images, and large generated outputs. Place assets into these relative locations before running:

- LLaVA-v1.5-7B base: `models/llava-v1.5-7b/`
- CLIP vision tower: `models/clip-vit-large-patch14-336/`
- xVerify weights: `models/xVerify-0.5B-I/`
- SAM3 checkpoint: `models/sam3_ckpt/sam3.pt`
- COCO train2014 images: `data/images/coco/train2014/`
- COCO val2014 images for POPE: `data/pope/val2014/`
- POPE annotation files: `data/pope/coco/`
- BEAF images and Q/A file: `data/beaf/images/`, `data/beaf/beaf_qna.json`
  These are not bundled; obtain them from the official BEAF release.
- GQA/VG images for mask generation: `data/images/gqa/images/`, `data/images/vg/VG_100K/`, `data/images/vg/VG_100K_2/`

The submitted dataset metadata is copied under `data/sage_as/`. The full released benchmark data is hosted separately as documented in its Croissant metadata.

The working copy used during packaging previously contained local symlinks under
`code/data_tools/gqa_vg_sam3_masks_sampled10000_20260430/vg_images/`. Those links
pointed to local VG images and are intentionally excluded from the release archive.
Use the relative image directories above instead.

## One-Command Entrypoint

The wrapper scripts are dry-run by default. They print checks and commands without launching long jobs.

```bash
cd /path/to/sage_repro_bundle
bash scripts/run_all_llava_sage_pipeline.sh
```

To actually execute a script, set `DRY_RUN=0`:

```bash
DRY_RUN=0 bash scripts/run_llava_stage2_mask_sft.sh
```

You can override paths and hyperparameters through environment variables without editing scripts.

## Environment

Use the same Python environment for Gemma, LLaVA, NaPO, and evaluation wrappers
unless you are deliberately debugging dependency differences. The validated
bundle environment is `.venv_gemma` at the bundle root.

Create it with:

```bash
python3 -m venv .venv_gemma
source .venv_gemma/bin/activate
pip install -r code/gemma_gate/gemma/requirements-py310.txt
```

An equivalent conda entrypoint is also provided:

```bash
conda env create -f environment.yml
conda activate sage-repro
```

The tested stack is:

- Python 3.10
- PyTorch 2.6.0 with CUDA 12.4 runtime
- Transformers 4.51.3
- DeepSpeed 0.16.7
- optional flash-attn 2.8.3 when the local CUDA toolchain can build/use it

`code/gemma_gate/requirements-extra.txt` currently installs only
`flash-attn==2.8.3`. Install it only when you need Flash Attention 2 support and
the CUDA/PyTorch ABI matches. The default top-level smoke runs use
SDPA/eager-compatible paths and do not require it.

Check the environment with:

```bash
$PWD/.venv_gemma/bin/python -c "import torch, transformers, deepspeed; print(torch.__version__, transformers.__version__, deepspeed.__version__)"
```

For all top-level scripts, set:

```bash
export PYTHON_BIN=$PWD/.venv_gemma/bin/python
```

The packaged LLaVA and NaPO LLaVA launch scripts invoke DeepSpeed as
`${PYTHON_BIN} -m deepspeed.launcher.runner`, so the selected Python environment
controls both the training process and the launcher. For manual commands that
call `deepspeed` directly, additionally run:

```bash
export PATH=$PWD/.venv_gemma/bin:$PATH
```

Known stable distributed settings for the LLaVA stage-2 SAGE path are:

- 4 GPUs with `CUDA_VISIBLE_DEVICES=0,1,2,3`
- `NCCL_IB_DISABLE=1`
- `NCCL_P2P_DISABLE=0`
- bf16 training
- ZeRO-1 for stage-2 SFT and ZeRO-2 for pretraining

Do not treat matching effective batch size as equivalent to matching the tested
micro-batch, accumulation, ZeRO stage, and precision settings. Re-run a smoke
test after any change.

## Pretraining

To regenerate the filtered LLaVA pretraining JSON from the raw
`llava_v1_5_mix665k.json`, run:

```bash
bash scripts/run_build_pretrain_json.sh
```

This wrapper runs `code/data_tools/build_llava_pretrain_json.py`, which splits
multi-turn LLaVA data into single-turn image-grounded samples, removes
OCR-related samples, keeps answers up to 200 characters, removes OCR_VQA rows,
applies the strict no-OCR historical drop-list, and removes likely truncated GPT
replies with `--mode aggressive`.

The final default output is
`data/pretrain_llava_v1_5_mix665k_single_noocr_max200_imageonly_strict_noocr.json`.

The raw `data/llava_stage1/llava_v1_5_mix665k.json` and the intermediate
`data/llava_stage1/llava_v1_5_mix665k_single_noocr_max200_imageonly.json` are
not included in the compressed release archive. Place the raw file at the
relative path above before rerunning this build step.

```bash
bash scripts/run_llava_pretrain_gate.sh
```

Important defaults:

- `LLAVA_BASE_MODEL=models/llava-v1.5-7b`
- `LLAVA_VISION_TOWER=models/clip-vit-large-patch14-336`
- `LLAVA_PRETRAIN_DATA=data/pretrain_llava_v1_5_mix665k_single_noocr_max200_imageonly_strict_noocr.json`
- `LLAVA_PRETRAIN_IMAGE_ROOT=data/images`
- output: `checkpoints/llava_pretrain_gate`

Example execution:

```bash
DRY_RUN=0 PYTHON_BIN=$PWD/.venv_gemma/bin/python CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash scripts/run_llava_pretrain_gate.sh
```

The pretraining job saves adapter weights as `mm_projector.bin`. To assemble a
directly loadable LLaVA checkpoint by injecting those projector/gate weights
into the LLaVA-v1.5 base model, run:

```bash
DRY_RUN=0 bash scripts/run_assemble_llava_checkpoint.sh
```

Important defaults:

- `ASSEMBLE_ADAPTER_PATH=checkpoints/llava_pretrain_gate/mm_projector.bin`
- `ASSEMBLE_OUTPUT_PATH=checkpoints/llava_pretrain_gate_assembled`
- `ASSEMBLE_FORCE_GATE=auto`
- `ASSEMBLE_VISION_TOWER_CONFIG_PATH` defaults to a relative path from the assembled checkpoint to `models/clip-vit-large-patch14-336`

`ASSEMBLE_FORCE_GATE=auto` enables the SAGE gate only when the adapter file
contains `gate` weights. For no-gate adapters, the script writes
`use_dual_input_gate=false` and excludes randomly initialized gate weights from
the assembled checkpoint.

The no-gate pretraining baseline uses the same defaults but forces the
dual-input gate off and sets gate L1 regularization to zero:

```bash
DRY_RUN=0 PYTHON_BIN=$PWD/.venv_gemma/bin/python CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash scripts/run_llava_pretrain_nogate.sh
```

Its default output is `checkpoints/llava_pretrain_nogate`.

## Mask-Supervised Stage-2 Finetuning

```bash
bash scripts/run_llava_stage2_mask_sft.sh
```

Important defaults:

- `SAGE_AS_DATASET=vqa`
- `LLAVA_STAGE2_DATA=data/sage_as/data/vqa/train.json`
- `LLAVA_STAGE2_EVAL_DATA=data/sage_as/data/vqa/val.json`
- `LLAVA_PATCH_MASK_NPZ=data/sage_as/masks/vqa_masks.npz`
- `LLAVA_PRETRAIN_PROJECTOR=checkpoints/llava_pretrain_gate/mm_projector.bin`
- `MASK_PATCH_LOSS_WEIGHT=0.125`
- `GATE_L1_LOSS_WEIGHT=0.01`
- `DEEPSPEED_CONFIG=code/llava_sage/scripts/zero1_bf16.json`
- `PER_DEVICE_TRAIN_BATCH_SIZE=32`
- `GRADIENT_ACCUMULATION_STEPS=1`
- `NUM_TRAIN_EPOCHS=2`
- `LR_SCHEDULER_TOTAL_STEPS_SCALE=1.5`
- output: `checkpoints/llava_stage2_sage`

Example parameter sensitivity run:

```bash
DRY_RUN=0 PYTHON_BIN=$PWD/.venv_gemma/bin/python \
  MASK_PATCH_LOSS_WEIGHT=0.25 GATE_L1_LOSS_WEIGHT=0.02 \
  LLAVA_STAGE2_CHECKPOINT=checkpoints/sage_mask_x2_l1_x2 \
  bash scripts/run_llava_stage2_mask_sft.sh
```

## Acc/SR Evaluation

```bash
bash scripts/run_llava_eval_acc_sr.sh
```

Important defaults:

- `MODEL_PATH=checkpoints/llava_stage2_sage`
- `TEST_RAW_WITH_SHORTCUT=data/eval/test_raw_with_shortcut_answer.json`
- `TEST_IMAGE_ROOT=data/images/coco/train2014`
- `XVERIFY_MODEL=models/xVerify-0.5B-I`
- output: `outputs/llava_infer/`

Example:

```bash
DRY_RUN=0 MODEL_PATH=checkpoints/llava_stage2_sage HAS_GATE=auto \
  bash scripts/run_llava_eval_acc_sr.sh
```

## POPE Evaluation

```bash
bash scripts/run_pope_eval.sh
```

Required data:

- `data/pope/llava_pope_test.jsonl`
- `data/pope/coco/coco_pope_*.json`
- `data/pope/val2014/`

## BEAF Evaluation

```bash
bash scripts/run_beaf_eval.sh
```

Required data:

- `data/beaf/beaf_qna.json`
- `data/beaf/images/`

This bundle does not redistribute the official BEAF benchmark files. See
`data/beaf/README.md` for the expected directory layout and source links.

## NaPO LLaVA Training

```bash
DRY_RUN=0 PYTHON_BIN=$PWD/.venv_gemma/bin/python CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash scripts/run_napo_llava.sh
```

Default preference dataset directory:

- `data/napo_llava/train_raw_pos_neg_shortcut_hf`

The expected preference construction is: positive answer from `generated_answer`, negative answer from `original_answer`.

The wrapper points to `third_party/napo_llava_ref/`, which is intentionally kept
outside `code/` so the main SAGE implementation remains separated from the
third-party NaPO baseline snapshot.

## Mask Generation and Filtering

```bash
bash scripts/run_mask_generation_and_filtering.sh
```

This wrapper documents and checks the SAM3/mask tooling. Full SAM3 mask generation requires an external SAM3 code environment and checkpoint. The copied scripts preserve the exact filtering logic:

- Non-number mask release: keep QA rows, remove only number-answer rows from the released mask NPZ.
- Active patch filtering: remove mask rows with `active_patch_frac == 0` or `active_patch_frac > threshold`, while retaining QA rows.

Relevant implementation files:

- `code/data_tools/gqa_vg_sam3_masks_sampled10000_20260430/build_nonumber_mask_packages.py`
- `code/data_tools/gqa_vg_sam3_masks_sampled10000_20260430/build_area_filtered_train_packages.py`
- `code/llava_sage/scripts2/build_mixed_mask_training_package.py`

## Notes

- Top-level wrappers are the canonical reproducibility interface.
- Copied historical scripts are included for traceability and may contain old absolute-path defaults from the original workspace.
- Outputs are written under `outputs/`, `logs/`, and `checkpoints/`.
- The package excludes raw images, base model weights, trained checkpoints, xVerify weights, SAM3 weights, and Python virtual environments.
