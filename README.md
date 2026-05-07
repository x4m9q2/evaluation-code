# VQA-CMSV / SAGE Anonymous Reproduction Code

This repository is the anonymous code bundle for the VQA-CMSV benchmark
generation pipeline and the SAGE experiments. The recommended entry points are
the top-level scripts under `scripts/`. They resolve paths relative to the
repository root and are intended to be the stable reproduction interface.
Historical scripts inside component subdirectories are kept mainly for tracing
and debugging.

This repository does not ship large assets. Model weights, raw images, training
checkpoints, optimizer states, generated outputs, SAM3 checkpoints, Qwen
weights or API access, and xVerify weights must be obtained separately.

Except for third-party notes under `third_party/`, runtime documentation has
been consolidated into this file. `data/sage_as/README.md` is kept as the local
copy of the Hugging Face dataset card and is not the main execution guide.

## 1. Environment Setup

Validated environment:

- Python 3.10
- PyTorch 2.6.0 with CUDA 12.4 runtime
- Transformers 4.51.3
- DeepSpeed 0.16.7
- bf16 training
- SDPA or eager attention; FlashAttention2 is optional

Create the environment:

```bash
conda env create -f environment.yml
conda activate sage-repro
```

If you reuse an existing virtualenv, explicitly point the wrappers to it:

```bash
export PYTHON_BIN="$PWD/.venv_gemma/bin/python"
export PATH="$PWD/.venv_gemma/bin:$PATH"
```

General rules:

- All wrappers execute real commands by default and fail fast if required files
  are missing.
- All default paths are relative to the repository root and can be overridden
  by environment variables.
- Do not casually change the DeepSpeed stage, per-device batch size, gradient
  accumulation, precision, learning-rate schedule, or `max_steps`. Those
  parameters affect numerical stability and the effective learning-rate curve;
  unvalidated combinations may trigger NaN or Inf.

## 2. Dataset Generation

### 2.1 Required External Files

VQA-CMSV generation requires:

- COCO 2014 train images: `data/images/coco/train2014/`
- COCO 2014 annotations: `annotations/instances_train2014.json`
- VQAv2 train questions:
  `data/detect-shortcuts/data/vqa2/v2_OpenEnded_mscoco_train2014_questions.json`
- VQAv2 train annotations:
  `data/detect-shortcuts/data/vqa2/v2_mscoco_train2014_annotations.json`
- GMiner for shortcut mining:
  `code/shortcut_pipeline/bin/GMiner`
- CUDA matcher for shortcut matching:
  `code/shortcut_pipeline/bin/cuda`
- SAM3 tokenizer assets used by the stage-1 post-step mask generation:
  `code/sam3/sam3/assets/`

Official download entry points:

```text
http://images.cocodataset.org/annotations/annotations_trainval2014.zip
http://images.cocodataset.org/zips/train2014.zip
https://visualqa.org/download.html
```

File placement details:

- `annotations/instances_train2014.json` must be copied from
  `annotations_trainval2014.zip`, specifically the file
  `annotations/instances_train2014.json`, into `annotations/`.
- VQAv2 train questions must use the original filename
  `v2_OpenEnded_mscoco_train2014_questions.json` and be placed at
  `data/detect-shortcuts/data/vqa2/`.
- VQAv2 train annotations must use the original filename
  `v2_mscoco_train2014_annotations.json` and be placed at
  `data/detect-shortcuts/data/vqa2/`.
- COCO train2014 images must be extracted so that files like
  `COCO_train2014_000000000009.jpg` are directly under
  `data/images/coco/train2014/`.
- If you also evaluate LLaVA or Gemma later, `data/images/` is expected to
  contain at least `coco/train2014/`, `gqa/images/`, and `vg/` when those
  datasets are used.

Shortcut-mining binaries and their provenance:

- `code/shortcut_pipeline/bin/GMiner` comes from the detect-shortcuts project:
  `https://github.com/cdancette/detect-shortcuts`
- `code/shortcut_pipeline/bin/cuda` is the local CUDA matcher binary used in
  this project for shortcut-rule matching
- The bundled SAM3 code expects tokenizer assets under
  `code/sam3/sam3/assets/` during text-conditioned mask generation
- The SAM3 codebase itself comes from Meta's official repository:
  `https://github.com/facebookresearch/sam3`
- The SAM3 checkpoint used by this repo is downloaded from Meta's Hugging Face
  release:
  `https://huggingface.co/facebook/sam3`
- The CUDA matcher source is bundled under
  `code/shortcut_pipeline/find_shortcut/`
- The main CUDA source file is
  `code/shortcut_pipeline/find_shortcut/test.cu`
- The CMake build file is
  `code/shortcut_pipeline/find_shortcut/CMakeLists.txt`
- The compiled binary is not tracked in Git; build it locally before running
  stage 1
- Rebuild the matcher from the repository root with:

```bash
bash scripts/build_shortcut_matcher.sh
```

- This writes the compiled binary to `code/shortcut_pipeline/bin/cuda`
- Build prerequisites:
  - CUDA toolkit compatible with your driver
  - `cmake >= 3.18`
  - `gcc-12` and `g++-12` if available on the machine
  - Jansson development headers and library, e.g. `libjansson-dev`

The released benchmark can be downloaded from Hugging Face:

```text
https://huggingface.co/datasets/as-benchmark-artifacts/vqa-cmsv-benchmark
https://huggingface.co/datasets/as-benchmark-artifacts/vqa-cmsv-benchmark/resolve/main/croissant.json
```

Recommended layout after download:

```text
data/sage_as/
  data/vqa_v2_cmsv/{train,val,test}.json
  data/gqa_cmsv/{train,val,test}.jsonl
  data/vg_cmsv/{train,val,test}.jsonl
  masks/{vqa_v2_cmsv,gqa_cmsv,vg_cmsv}_masks.npz
```

### 2.2 Generation Pipeline

Stage 1: mine textual shortcut rules and candidate matches.

```bash
bash scripts/run_shortcut_stage1.sh
```

`run_shortcut_stage1.sh` now always continues into the stage-2 mask-preparation
prefix after stage-1 mining finishes. Concretely, it performs:

- `prepare_stage2_inputs.py`
- SAM3 runtime preflight
- `generate_union_masks_from_mapping.py`
- `apply_union_masks_to_images.py`

That means the wrapper also needs the SAM3 tokenizer assets under
`code/sam3/sam3/assets/`. These assets are not downloaded automatically by
the wrapper.

There is no longer a separate optional switch for skipping this prefix in the
default wrapper behavior.

Stage 2: generate CMSV samples from the stage-1 results.

```bash
bash scripts/run_shortcut_stage2.sh
```

If you want to submit stage-2 requests to an API, pass the API settings
explicitly. Provide:

```bash
OPENAI_BASE_URL=...
OPENAI_API_KEY=...
MODEL=...
SUBMIT_API=1 bash scripts/run_shortcut_stage2.sh
```

Convert generation outputs into released VQA v2-CMSV splits:

```bash
BATCH_OUTPUT_JSONL=outputs/shortcut_stage2/generated_samples.jsonl \
  bash scripts/run_build_vqa_v2_cmsv_splits.sh
```

If you only need to reproduce experiments rather than regenerate the dataset,
you can directly download the released splits:

```bash
bash scripts/run_download_vqa_v2_cmsv.sh
```

Split semantics:

- `train`: training split. For VQA v2-CMSV, this is the mixed stage-2 training
  set used in the main experiments. It contains generated train questions,
  generated train questions that remain in JSON but lose mask supervision after
  filtering, and original no-mask questions reconstructed from CMSV
  train/val/test.
- `val`: validation split for stage-2 loss evaluation.
- `test`: test split for Acc/SR evaluation.

At training time, whether mask loss is enabled is determined by matching
`question_id` against the NPZ mask package.

## 3. LLaVA Experiments

### 3.1 Required Files

Required models and components:

- `models/llava-v1.5-7b/`
- `models/clip-vit-large-patch14-336/`
- `models/sam3_ckpt/sam3.pt`, only needed for mask generation

Required data:

- LLaVA stage-1 raw mix:
  `data/llava_stage1/llava_v1_5_mix665k.json`
- LLaVA stage-1 image root: `data/playground_data/`
- VQA/GQA/VG CMSV splits and masks: `data/sage_as/`
- Evaluation image root: `data/images/`

Optional dependencies:

- Qwen model or API for visual-cue filtering
- POPE data under `data/pope/`
- BEAF data under `data/beaf/`
- xVerify is not distributed in this bundle

### 3.2 Workflow

#### 3.2.1 Stage-1 Data Preparation

Build the image-only, no-OCR, answer-length-limited LLaVA pretraining JSON:

```bash
bash scripts/run_build_pretrain_json.sh
```

Default output:

```text
data/pretrain_llava_v1_5_mix665k_single_noocr_max200_imageonly_strict_noocr.json
```

#### 3.2.2 Stage-1 Pretraining with Gate

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
OUTPUT_DIR=checkpoints/llava_pretrain_gate \
bash scripts/run_llava_pretrain_gate.sh
```

Key defaults:

- `--use_dual_input_gate True`
- `--tune_mm_mlp_adapter True`
- DeepSpeed config `code/llava_sage/scripts/zero2_bf16.json`
- bf16
- `learning_rate=1e-3`

#### 3.2.3 Stage-1 Pretraining without Gate

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
OUTPUT_DIR=checkpoints/llava_pretrain_nogate \
bash scripts/run_llava_pretrain_nogate.sh
```

This script uses the same pretraining data and optimization setup as the gated
version, but disables the gate and L1-related terms.

#### 3.2.4 Checkpoint Assembly

If stage 1 only saves a projector or gate adapter, assemble it into a directly
loadable LLaVA checkpoint:

```bash
ASSEMBLE_ADAPTER_PATH=checkpoints/llava_pretrain_gate/mm_projector.bin \
ASSEMBLE_OUTPUT_PATH=checkpoints/llava_pretrain_gate_assembled \
bash scripts/run_assemble_llava_checkpoint.sh
```

`ASSEMBLE_FORCE_GATE=auto` detects whether the adapter contains gate weights.

#### 3.2.5 Qwen Filtering

Qwen filtering is used to decide whether mask supervision is reliable:

```bash
bash scripts/run_qwen_visual_cue_filter.sh vqa
bash scripts/run_qwen_visual_cue_filter.sh gqa
bash scripts/run_qwen_visual_cue_filter.sh vg
```

Qwen filtering only affects NPZ mask rows and whether mask loss is enabled
during training. It does not delete QA samples from JSON or JSONL.

Run Qwen filtering before the mask generation/filter/build steps below. The
scripts are intended to be executed in order:

- `run_qwen_visual_cue_filter.sh`
- `run_mask_generation_and_filtering.sh ...-generate`
- `run_mask_generation_and_filtering.sh ...-filter`
- `run_mask_generation_and_filtering.sh ...-build`

#### 3.2.6 Mask NPZ Generation

VQA mask generation, filtering, and packaging:

```bash
bash scripts/run_mask_generation_and_filtering.sh vqa-generate
bash scripts/run_mask_generation_and_filtering.sh vqa-filter
bash scripts/run_mask_generation_and_filtering.sh vqa-build
```

These three wrapper commands are not independent. `vqa-filter` consumes the
Qwen keep/remove decisions, and `vqa-build` consumes the outputs of the earlier
steps to produce the final stage-2 SFT JSON plus NPZ package.

`vqa-build` uses only `data/shortcut_pipeline/vqa_v2_cmsv/train.json` as the
generated-train source for supervised rows. Original no-mask rows are rebuilt
from `original_question` and `original_answer` stored in
`data/shortcut_pipeline/vqa_v2_cmsv/{train,val,test}.json`.

This mixed stage-2 SFT file is separate from the NaPO preference data under
`data/napo/shortcut_generated_vqa/train.json`. Do not mix those paths.

The `vqa-build` mixing logic is:

- Generated-question side: only the generated `train` split goes through SAM3
  mask generation and Qwen keep/remove filtering.
- Original-question side: original questions from CMSV `train+val+test` are
  appended as no-mask rows.
- To avoid accidental mask matching, appended original rows are remapped to a
  new `question_id` range that does not overlap with generated-train mask IDs.

GQA/VG mask generation, filtering, and packaging:

```bash
bash scripts/run_mask_generation_and_filtering.sh gqa-vg-generate
bash scripts/run_mask_generation_and_filtering.sh gqa-vg-filter
bash scripts/run_mask_generation_and_filtering.sh gqa-vg-build
```

As with VQA, run these in order after the corresponding Qwen filtering step for
GQA or VG.

Mask rules:

- Training uses NPZ `question_id` matching as the authoritative signal for mask
  supervision.
- `mask_supervision` in JSON or JSONL is readable metadata, not the only
  supervision source.
- Number-answer samples are not removed from the QA splits. Their corresponding
  mask rows are removed from NPZ, or they are treated as no-mask supervision at
  training time.

#### 3.2.7 Stage-2 Training

Gated SAGE:

```bash
SAGE_AS_DATASET=vqa \
LLAVA_PRETRAIN_PROJECTOR=checkpoints/llava_pretrain_gate/mm_projector.bin \
LLAVA_STAGE2_CHECKPOINT=checkpoints/llava_stage2_sage_vqa \
bash scripts/run_llava_stage2_mask_sft.sh
```

Non-gated baseline:

```bash
SAGE_AS_DATASET=vqa \
LLAVA_STAGE2_NOGATE_CHECKPOINT=checkpoints/llava_stage2_nogate_vqa \
bash scripts/run_llava_stage2_mask_sft_nogate.sh
```

Set `SAGE_AS_DATASET` to `vqa`, `gqa`, or `vg` as needed.

#### 3.2.8 Acc/SR Evaluation

The LLaVA evaluation wrapper supports VQA, GQA, and VG:

```bash
LLAVA_EVAL_DATASET=vqa \
MODEL_PATH=checkpoints/llava_stage2_sage_vqa \
bash scripts/run_llava_eval_acc_sr.sh
```

Common overrides:

- `LLAVA_EVAL_DATASET=vqa|gqa|vg`
- `MODEL_PATH=...`
- `HAS_GATE=auto|true|false`
- `TORCH_DTYPE=bf16`

Important note: the current wrapper generates prediction outputs on the CMSV
test split. If you need Acc and SR numbers that exactly follow the paper’s
experimental protocol, you still need to configure xVerify separately as the
judge used for final Acc/SR computation.

#### 3.2.9 NaPO

Before running LLaVA NaPO:

- Download the official NaPO repository from
  `https://github.com/zhangzef/NaPO`
- Extract it without renaming and place it at `third_party/NaPO-master/`
- Install the extra dependency required by the upstream code:

```bash
pip install matplotlib
```

Bundle-specific compatibility notes for the validated environment:

- The local copy under `third_party/NaPO-master/` includes the upstream
  `muffin/data/` package, which must be present for LLaVA NaPO training.
- `third_party/NaPO-master/muffin/data/datasets.py` is patched to support
  Hugging Face datasets stored with `datasets.save_to_disk()`. This is the
  format produced by this bundle.
- The same file also lazy-loads TSV support so HF-only runs do not fail just
  because optional TSV utilities are absent.
- `third_party/NaPO-master/utils/utils.py` moves the `matplotlib` import into
  the plotting function so DPO training does not fail at import time when
  plotting is unused.
- `third_party/NaPO-master/muffin/train/trainers.py` accepts
  `num_items_in_batch=None`, matching `transformers==4.51.3`, and removes a
  debug print from the DPO path.
- These compatibility changes do not alter NaPO preference construction, DPO
  targets, or loss semantics.

Wrapper-specific differences from upstream:

- `scripts/run_napo_llava.sh` uses repository-relative paths and exports
  `PYTHONPATH` to `third_party/NaPO-master/`.
- The wrapper creates `third_party/clip-vit-large-patch14-336` as a local
  symlink to the configured CLIP tower when needed.
- The wrapper uses `--eval_strategy no` for compatibility with
  `transformers==4.51.3`.
- For normal bundle runs, use `scripts/run_napo_llava.sh` rather than the
  original scripts under `third_party/NaPO-master/script/train/`.

Build LLaVA NaPO preference data:

```bash
SAGE_AS_DATASET=vqa bash scripts/run_build_shortcut_napo_splits.sh
SAGE_AS_DATASET=vqa bash scripts/run_build_shortcut_napo_llava_dataset.sh
```

`scripts/run_build_shortcut_napo_splits.sh` writes:

- `data/napo/shortcut_generated_vqa/train.json`
- `data/napo/shortcut_generated_vqa/val.json`
- `data/napo/shortcut_generated_vqa/test.json`

These preference splits are built from the shortcut stage-2 files:

- `data/shortcut_pipeline/cross_modality_qa_input.json`
- `data/shortcut_pipeline/batch_outputs/cross_modality_qa_responses.jsonl`

The builder keeps only rows with both `generated_answer` and
`original_answer`, filters out pairs where the two answers are identical, and
stores them using the NaPO convention:

- negative answer: `original_answer`
- positive answer: `generated_answer`

Run NaPO:

```bash
SAGE_AS_DATASET=vqa \
NAPO_LLAVA_OUTPUT_ROOT=checkpoints/napo_llava_vqa \
bash scripts/run_napo_llava.sh
```

Validated local smoke run status:

- LLaVA NaPO has been verified in this bundle on a small HF dataset slice and
  reaches actual trainer steps, not just argument parsing.
- The expected input format for `NAPO_LLAVA_DATA_DIR` is a Hugging Face dataset
  directory containing a `train` split saved with `datasets.save_to_disk()`.

Validated smoke command:

```bash
PYTHON_BIN=$PWD/.venv_gemma/bin/python CUDA_VISIBLE_DEVICES=0,1,2,3 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 PER_DEVICE_EVAL_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=1 DATALOADER_NUM_WORKERS=0 LOGGING_STEPS=1 \
NUM_EPOCHS=1 OUTPUT_DIR=outputs/napo_llava_shortcut_generated_smoke/checkpoints \
LOGGING_DIR=outputs/napo_llava_shortcut_generated_smoke/log \
EXTRA_ARGS='--max_steps 2' bash scripts/run_napo_llava.sh
```

#### 3.2.10 CausalMM

CausalMM performs plug-and-play inference on the CMSV test split:

```bash
LLAVA_EVAL_DATASET=vqa \
MODEL_PATH=models/llava-v1.5-7b \
bash scripts/run_cmsv_causalmm_llava.sh
```

Set `LLAVA_EVAL_DATASET` to `vqa`, `gqa`, or `vg` as needed. This flow is
independent of POPE/BEAF and does not depend on xVerify.

#### 3.2.11 POPE and BEAF

Standard SAGE/LLaVA POPE:

```bash
MODEL_PATH=checkpoints/llava_stage2_sage_vqa \
bash scripts/run_pope_eval.sh
```

BEAF:

```bash
MODEL_PATH=checkpoints/llava_stage2_sage_vqa \
bash scripts/run_beaf_eval.sh
```

Raw POPE/BEAF data and images are not redistributed in this repository and must
be obtained separately under their original terms. Place the local POPE files at
`data/pope/llava_pope_test.jsonl`, `data/pope/coco/`, and `data/pope/val2014/`.
Place the local BEAF files at `data/beaf/beaf_qna.json` and
`data/beaf/images/`.

## 4. Gemma Experiments

### 4.1 Required Files

Required models:

- `models/Gemma-3-4B-IT/`
- `models/siglip-so400m-patch14-384/`

Required data:

- Pretraining JSON:
  `data/pretrain_llava_v1_5_mix665k_single_noocr_max200_imageonly_strict_noocr.json`
- Pretraining image root: `data/playground_data/`
- VQA/GQA/VG CMSV splits and masks: `data/sage_as/`
- Stage-2 image root: `data/images/`

Optional dependencies:

- Qwen model or API for mask-supervision filtering
- NaPO training data, which can be built from CMSV splits using this repository
- CausalMM evaluation inputs, derived from CMSV test splits

### 4.2 Workflow

#### 4.2.1 Stage-1 Data Preparation

Gemma reuses the same pretraining JSON as LLaVA:

```bash
bash scripts/run_build_pretrain_json.sh
```

#### 4.2.2 Stage-1 Pretraining with Gate

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PRETRAIN_CHECKPOINT=checkpoints/gemma3_4b_pretrain_gate_projector_l1_sdpa \
bash scripts/run_pretrain_gate.sh
```

Defaults: bf16, SDPA, 4 GPUs, `learning_rate=1e-3`.

#### 4.2.3 Stage-1 Pretraining without Gate

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PRETRAIN_NOGATE_CHECKPOINT=checkpoints/gemma3_4b_pretrain_projector_sdpa \
bash scripts/run_pretrain_nogate.sh
```

This script disables the gate and L1 loss while keeping the rest of the stage-1
configuration as close as possible to the gated version.

#### 4.2.4 Checkpoint Assembly

The default Gemma full fine-tuning scripts save directly loadable checkpoints,
so extra assembly is usually unnecessary. If you use a LoRA or adapter-only
variant, merge it using the corresponding upstream Gemma tooling.

#### 4.2.5 Qwen Filtering

Gemma and LLaVA share the same Qwen filtering results and NPZ masks:

```bash
bash scripts/run_qwen_visual_cue_filter.sh vqa
bash scripts/run_mask_generation_and_filtering.sh vqa-build
```

As with LLaVA, filtering only affects NPZ mask rows and whether mask loss is
enabled. It does not remove QA rows.

Run the shared Qwen filtering step before the NPZ generation/filter/build
pipeline. The intended order is the same as in the LLaVA section:

- `run_qwen_visual_cue_filter.sh`
- `run_mask_generation_and_filtering.sh ...-generate`
- `run_mask_generation_and_filtering.sh ...-filter`
- `run_mask_generation_and_filtering.sh ...-build`

#### 4.2.6 Mask NPZ Generation

Gemma reads the NPZ pointed to by `PATCH_MASK_ANALYSIS_PATH` and matches rows by
`question_id`:

```bash
bash scripts/run_mask_generation_and_filtering.sh vqa-generate
bash scripts/run_mask_generation_and_filtering.sh vqa-filter
bash scripts/run_mask_generation_and_filtering.sh vqa-build
```

`vqa-filter` depends on the Qwen filtering outputs, and `vqa-build` packages
the earlier outputs into the final stage-2 training JSON plus NPZ used by
Gemma.

GQA/VG:

```bash
bash scripts/run_mask_generation_and_filtering.sh gqa-vg-generate
bash scripts/run_mask_generation_and_filtering.sh gqa-vg-filter
bash scripts/run_mask_generation_and_filtering.sh gqa-vg-build
```

#### 4.2.7 Stage-2 Training

Gated SAGE:

```bash
SAGE_AS_DATASET=vqa \
PRETRAIN_CHECKPOINT=checkpoints/gemma3_4b_pretrain_gate_projector_l1_sdpa \
STAGE2_CHECKPOINT=checkpoints/gemma3_4b_stage2_gate_l1_mask_sdpa_vqa \
bash scripts/run_stage2_sft_gate.sh
```

Non-gated baseline:

```bash
SAGE_AS_DATASET=vqa \
PRETRAIN_NOGATE_CHECKPOINT=checkpoints/gemma3_4b_pretrain_projector_sdpa \
STAGE2_NOGATE_CHECKPOINT=checkpoints/gemma3_4b_stage2_nogate_sdpa_vqa \
bash scripts/run_stage2_sft_nogate.sh
```

Set `SAGE_AS_DATASET` to `vqa`, `gqa`, or `vg` as needed. During training, the
logs should show non-zero `mask_patch_loss` on supervised rows and finite total
loss values.

#### 4.2.8 Acc/SR Evaluation

Single-process Gemma evaluation:

```bash
SAGE_AS_DATASET=vqa \
MODEL_ID=checkpoints/gemma3_4b_stage2_gate_l1_mask_sdpa_vqa \
bash scripts/run_eval_test_raw.sh
```

4-GPU sharded inference:

```bash
SAGE_AS_DATASET=vqa \
MODEL_ID=checkpoints/gemma3_4b_stage2_gate_l1_mask_sdpa_vqa \
bash scripts/run_eval_test_raw_4gpu.sh
```

Important note: these wrappers generate prediction outputs on the CMSV test
split. If you need Acc and SR numbers that strictly match the paper’s reported
protocol, you must additionally configure xVerify for the final judge-based
Acc/SR computation.

#### 4.2.9 NaPO

Build Gemma NaPO data:

```bash
SAGE_AS_DATASET=vqa bash scripts/run_build_shortcut_napo_splits.sh
```

Run NaPO:

```bash
SAGE_AS_DATASET=vqa \
bash scripts/run_napo_shortcut.sh
```

If `NAPO_DATA` is not set explicitly, the script defaults to:

```text
data/napo/shortcut_generated_vqa/train.json
```

This file is produced by:

```bash
SAGE_AS_DATASET=vqa bash scripts/run_build_shortcut_napo_splits.sh
```

It is generated from:

- `data/shortcut_pipeline/cross_modality_qa_input.json`
- `data/shortcut_pipeline/batch_outputs/cross_modality_qa_responses.jsonl`

and is separate from the mixed stage-2 SFT data. `vqa-build` does not read the
preference files under `data/napo/`.

#### 4.2.10 CausalMM

Gemma CausalMM:

```bash
SAGE_AS_DATASET=vqa \
MODEL_PATH=models/Gemma-3-4B-IT \
bash scripts/run_cmsv_causalmm_gemma.sh
```

The default adaptation uses language-side counterfactual attention:

```text
causalmm_logits = (1 + gamma) * logits - gamma * cf_logits
```

Default behavior can be overridden through `CF_MODE`, `ATTENTION_METHOD`,
`GAMMA`, and `EPSILON`.

## 5. License and Dependency Notes

This repository distributes only anonymous code, configuration files, small
metadata, and wrapper scripts. Users must obtain external data, images, model
weights, and evaluation resources themselves and comply with the corresponding
licenses or terms of use.

Main dependency notes:

- LLaVA-related code is under `code/llava_sage/`; retain upstream licenses and
  citations when reusing it.
- Gemma fine-tuning code is under `code/gemma_gate/gemma/`; retain upstream
  licenses and citations. Gemma weights are governed by Gemma model terms and
  are not redistributed here.
- CausalMM-LLaVA code is under `code/evaluation/causalmm_llava/`; keep its
  license notice and cite the CausalMM paper when using it.
- The Gemma CausalMM adaptation is under `code/causalmm_gemma3/gemma3/` and is an
  adaptation of the counterfactual decoding idea to Gemma.
- Download the official NaPO repository from
  `https://github.com/zhangzef/NaPO`, extract it without renaming, and place
  the resulting directory at `third_party/NaPO-master/`. Retain upstream
  notices and license files when using it.
- POPE and BEAF are external hallucination-evaluation resources; this
  repository only keeps wrapper scripts and does not redistribute the datasets.
- SAM3 source code and Qwen-filtering scripts are included in the bundle, but
  the SAM3 checkpoint, Qwen weights, Qwen API access, and xVerify weights are
  not redistributed. xVerify-related files kept in this bundle are limited to
  format-conversion and result-summary helpers. If you want full paper-aligned
  Acc/SR numbers, xVerify still has to be configured separately.
- The VQA-CMSV data package contains only derived QA annotations and mask
  metadata, not raw or masked images. Upstream datasets such as COCO, VQAv2,
  GQA, and Visual Genome remain governed by their original licenses.
