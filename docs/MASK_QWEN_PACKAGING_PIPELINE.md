# Mask -> Qwen Filter -> Training Package Pipeline

This note documents the complete offline preprocessing chain used in this
bundle for:

- VQA stage-2 data under `data/stage2/`
- GQA/VG sampled10000 data under `data2/`

Bundle root assumption:

- Run commands from the extracted bundle root, referred to below as
  `/path/to/sage_repro_bundle`.

## What Each Stage Does

1. `SAM3 mask generation`
   - Read question rows plus `visual_cues`
   - Run SAM3 once per cue
   - Union all cue masks into one binary mask per `question_id`
2. `Qwen visual-cue filtering`
   - Read the per-question metadata emitted by the SAM3 stage
   - Ask Qwen whether the question text explicitly mentions the same object
     category as any visual cue
   - Split samples into `keep` and `remove`
   - In this pipeline:
     - `keep` means the visual cue is not explicitly mentioned in the question
     - `remove` means the visual cue is explicitly mentioned
3. `Training package build`
   - Keep masks only for rows that survive the Qwen filter
   - Convert binary masks into LLaVA-compatible patch NPZ files
   - Optionally remove or suppress mask supervision for `answer_type=number`
   - Optionally drop overly large masks by active-patch ratio

## Required External Assets

The bundle does not embed large upstream image corpora or model weights.
Before running the pipeline, place at least these assets:

- COCO train2014 images:
  - `data/images/coco/train2014`
- GQA images:
  - `data/images/gqa/images`
- VG images:
  - `data/images/vg/VG_100K`
  - `data/images/vg/VG_100K_2`
- SAM3 checkpoint:
  - `models/sam3_ckpt/sam3.pt`
- Qwen local model for filtering:
  - `models/Qwen3.5-9B`
- LLaVA + CLIP configs used when writing patch NPZ:
  - `models/llava-v1.5-7b/config.json`
  - `models/clip-vit-large-patch14-336/config.json`
  - `models/clip-vit-large-patch14-336/preprocessor_config.json`

For upstream dataset download locations, see
`data/stage2/STAGE2_DATA_README.source.md`.

## Main Entry Points

- Bundle wrapper:
  - `scripts/run_mask_generation_and_filtering.sh`
- Qwen filter wrapper:
  - `scripts/run_qwen_visual_cue_filter.sh`
- VQA package builder:
  - `code/llava_sage/scripts2/build_qwenkeep_stage2_package.py`
- GQA/VG package builders:
  - `code/data_tools/gqa_vg_sam3_masks_sampled10000_20260430/build_qwenkeep_packages.py`
  - `code/data_tools/gqa_vg_sam3_masks_sampled10000_20260430/build_nonumber_packages.py`
  - `code/data_tools/gqa_vg_sam3_masks_sampled10000_20260430/build_nonumber_mask_packages.py`
  - `code/data_tools/gqa_vg_sam3_masks_sampled10000_20260430/build_area_filtered_train_packages.py`

## VQA Stage-2 Pipeline

### Inputs

- Raw train rows:
  - `data/stage2/train_raw.json`
- SAM3 question JSONL:
  - `data/stage2/train_raw_llava.jsonl`
- SAM3 cue mapping:
  - `data/stage2/merged_output_rule_mapping.json`
- Extra VQAv2 rows appended during final mix build:
  - `data/stage2/vqa_train2014.json`

### Step 1. Generate SAM3 union masks

The VQA wrapper runs one shard at a time. Typical 4-shard run:

```bash
cd /path/to/sage_repro_bundle
for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$shard RUN=1 NUM_SHARDS=4 SHARD_INDEX=$shard \
    bash scripts/run_mask_generation_and_filtering.sh vqa-generate
done
```

Outputs:

- `outputs/sam3_train_raw_llava_union_masks/masks/<question_id>.png`
- `outputs/sam3_train_raw_llava_union_masks/shard_meta/shard_00.json` ... `shard_03.json`

Important:

- The current bundle already contains many copied VQA mask PNGs under
  `outputs/sam3_train_raw_llava_union_masks/masks/`
- It does not currently contain the corresponding `shard_meta/` directory
  needed to rerun Qwen filtering from this stage
- If you want to rerun Qwen filtering, rerun this SAM3 step first

### Step 2. Run Qwen visual-cue filtering

```bash
cd /path/to/sage_repro_bundle
FILTER_GPUS=0,1,2,3 RUN=1 \
  bash scripts/run_mask_generation_and_filtering.sh vqa-filter
```

Outputs:

- `analysis/visual_cue_question_filter_qwen35_strict4/full_all_shards/run_00/`
- `analysis/visual_cue_question_filter_qwen35_strict4/full_all_shards/run_01/`
- `analysis/visual_cue_question_filter_qwen35_strict4/full_all_shards/run_02/`
- `analysis/visual_cue_question_filter_qwen35_strict4/full_all_shards/run_03/`
- `analysis/visual_cue_question_filter_qwen35_strict4/full_all_shards/merged/keep.json`
- `analysis/visual_cue_question_filter_qwen35_strict4/full_all_shards/merged/remove.json`
- `analysis/visual_cue_question_filter_qwen35_strict4/full_all_shards/merged/summary.json`
- `analysis/visual_cue_question_filter_qwen35_strict4/full_all_shards/merged/audit.jsonl`

Per-run outputs are produced by
`code/llava_sage/scripts2/filter_visual_cue_mentions_qwen.py`
and merged by
`code/llava_sage/scripts2/merge_qwen_filter_runs.py`.

Important:

- The current bundle already contains the merged VQA filter outputs
- It does not currently contain the original `run_00` ... `run_03` directories

### Step 3. Build final VQA stage-2 training package

```bash
cd /path/to/sage_repro_bundle
RUN=1 bash scripts/run_mask_generation_and_filtering.sh vqa-build
```

Outputs:

- `data/stage2/train_raw_mixed_qwenratio_oldbase_sam3_plus_vqa.json`
- `data/stage2/patch_mask_analysis_train_raw_qwenkeep_sam3_compat.npz`
- `data/stage2/train_raw_mixed_qwenratio_oldbase_sam3_plus_vqa.summary.json`

What the builder does:

- Rows in Qwen `keep.json` keep SAM3 mask supervision
- Rows in Qwen `remove.json` stay in the JSON but are marked `mask_supervision=none`
- `vqa_train2014.json` rows are appended without mask supervision
- The NPZ contains only the kept SAM3-masked rows, converted to LLaVA patch space

## GQA / VG sampled10000 Pipeline

### Inputs

- GQA source rows:
  - `data2/GQA/GQA_filtered_sampled_10000.json`
- VG source rows:
  - `data2/vg/vg_filtered_sampled_10000.json`
- SAM3 JSONL + mapping files:
  - `code/data_tools/gqa_vg_sam3_masks_sampled10000_20260430/gqa_sampled10000.jsonl`
  - `code/data_tools/gqa_vg_sam3_masks_sampled10000_20260430/gqa_sampled10000_mapping.json`
  - `code/data_tools/gqa_vg_sam3_masks_sampled10000_20260430/vg_sampled10000.jsonl`
  - `code/data_tools/gqa_vg_sam3_masks_sampled10000_20260430/vg_sampled10000_mapping.json`

### Step 1. Generate SAM3 union masks

```bash
cd /path/to/sage_repro_bundle
MASK_GPUS=0,1,2,3 RUN=1 \
  bash scripts/run_mask_generation_and_filtering.sh gqa-vg-generate all
```

Outputs:

- `analysis/gqa_sampled10000_sam3_union_masks/masks/<question_id>.png`
- `analysis/gqa_sampled10000_sam3_union_masks/shard_meta/shard_00.json` ... `shard_03.json`
- `analysis/vg_sampled10000_sam3_union_masks/masks/<question_id>.png`
- `analysis/vg_sampled10000_sam3_union_masks/shard_meta/shard_00.json` ... `shard_03.json`

Important:

- The current bundle already contains copied GQA/VG mask PNGs under the
  corresponding `analysis/*_sam3_union_masks/masks/` directories
- It does not currently contain the corresponding `shard_meta/` directories
- If you want to rerun Qwen filtering, rerun this SAM3 step first

### Step 2. Run Qwen visual-cue filtering

```bash
cd /path/to/sage_repro_bundle
FILTER_GPUS=0,1,2,3 RUN=1 \
  bash scripts/run_mask_generation_and_filtering.sh gqa-vg-filter all
```

Outputs for each dataset:

- `analysis/<dataset>_sampled10000_qwen35_filter/run_00/`
- `analysis/<dataset>_sampled10000_qwen35_filter/run_01/`
- `analysis/<dataset>_sampled10000_qwen35_filter/run_02/`
- `analysis/<dataset>_sampled10000_qwen35_filter/run_03/`
- `analysis/<dataset>_sampled10000_qwen35_filter/merged/keep.json`
- `analysis/<dataset>_sampled10000_qwen35_filter/merged/remove.json`
- `analysis/<dataset>_sampled10000_qwen35_filter/merged/summary.json`
- `analysis/<dataset>_sampled10000_qwen35_filter/merged/audit.jsonl`

The current bundle already contains the merged filter outputs for:

- `analysis/gqa_sampled10000_qwen35_filter/merged/`
- `analysis/vg_sampled10000_qwen35_filter/merged/`

### Step 3. Build qwenkeep package

```bash
cd /path/to/sage_repro_bundle
RUN=1 python code/data_tools/gqa_vg_sam3_masks_sampled10000_20260430/build_qwenkeep_packages.py
```

Or through the wrapper:

```bash
cd /path/to/sage_repro_bundle
RUN=1 bash scripts/run_mask_generation_and_filtering.sh gqa-vg-build
```

Primary outputs:

- `data2/GQA/GQA_filtered_sampled_10000_qwenkeep_sam3.json`
- `patch_mask_analysis_gqa_sampled10000_qwenkeep_sam3_compat.npz`
- `data2/vg/vg_filtered_sampled_10000_qwenkeep_sam3.json`
- `patch_mask_analysis_vg_sampled10000_qwenkeep_sam3_compat.npz`

What this stage does:

- If a row is in Qwen `keep` and its mask exists:
  - `mask_supervision=sam3_patch_mask`
- If a row is in Qwen `keep` but the mask is missing:
  - keep row, downgrade to `mask_supervision=none`
- If a row is in Qwen `remove`:
  - keep row, but set `mask_supervision=none`

### Step 4. Build `nonumber`

Produced by
`code/data_tools/gqa_vg_sam3_masks_sampled10000_20260430/build_nonumber_packages.py`.

Outputs:

- `data2/GQA/GQA_filtered_sampled_10000_qwenkeep_sam3_nonumber.json`
- `patch_mask_analysis_gqa_sampled10000_qwenkeep_sam3_nonumber_compat.npz`
- `data2/vg/vg_filtered_sampled_10000_qwenkeep_sam3_nonumber.json`
- `patch_mask_analysis_vg_sampled10000_qwenkeep_sam3_nonumber_compat.npz`

Rule:

- Drop all rows with `answer_type == "number"`
- Keep NPZ entries only for the remaining masked rows

### Step 5. Build `nonumbermask`

Produced by
`code/data_tools/gqa_vg_sam3_masks_sampled10000_20260430/build_nonumber_mask_packages.py`.

Outputs:

- `data2/GQA/GQA_filtered_sampled_10000_qwenkeep_sam3_nonumbermask.json`
- `patch_mask_analysis_gqa_sampled10000_qwenkeep_sam3_nonumbermask_compat.npz`
- `data2/vg/vg_filtered_sampled_10000_qwenkeep_sam3_nonumbermask.json`
- `patch_mask_analysis_vg_sampled10000_qwenkeep_sam3_nonumbermask_compat.npz`

Rule:

- Keep all JSON rows
- For rows with `answer_type == "number"` and `mask_supervision == sam3_patch_mask`,
  change them to `mask_supervision=none`
- Remove those rows from the NPZ

### Step 6. Build area-filtered final LLaVA training JSONs

Produced by
`code/data_tools/gqa_vg_sam3_masks_sampled10000_20260430/build_area_filtered_train_packages.py`.

Outputs for each dataset and threshold:

- `data2/GQA/gqa_filtered_sampled_10000_qwenkeep_sam3_nonumbermask_area001_max0p5_llava.json`
- `patch_mask_analysis_gqa_sampled10000_qwenkeep_sam3_nonumbermask_area001_max0p5_compat.npz`
- `data2/GQA/gqa_filtered_sampled_10000_qwenkeep_sam3_nonumbermask_area001_max0p7_llava.json`
- `patch_mask_analysis_gqa_sampled10000_qwenkeep_sam3_nonumbermask_area001_max0p7_compat.npz`
- `data2/vg/vg_filtered_sampled_10000_qwenkeep_sam3_nonumbermask_area001_max0p5_llava.json`
- `patch_mask_analysis_vg_sampled10000_qwenkeep_sam3_nonumbermask_area001_max0p5_compat.npz`
- `data2/vg/vg_filtered_sampled_10000_qwenkeep_sam3_nonumbermask_area001_max0p7_llava.json`
- `patch_mask_analysis_vg_sampled10000_qwenkeep_sam3_nonumbermask_area001_max0p7_compat.npz`

Rule:

- Keep all JSON rows
- Keep NPZ mask rows only if:
  - active patch fraction `> 0`
  - active patch fraction `<= threshold`
- Thresholds used here are `0.5` and `0.7`
- Rows whose masks are empty or too large stay in JSON but become `mask_supervision=none`

## Current Bundle Status

The bundle already contains enough artifacts to inspect or reuse the released
results without recomputing everything:

- VQA merged Qwen filter outputs already present
- VQA final mixed JSON and compat NPZ already present
- GQA merged Qwen filter outputs already present
- VG merged Qwen filter outputs already present
- GQA/VG qwenkeep, nonumber, nonumbermask, and area-filtered package outputs
  already present

What is not fully preserved from the original workspace:

- VQA/GQA/VG copied mask directories currently do not include `shard_meta/`
- VQA per-shard Qwen run directories were not copied into the bundle
- GQA/VG per-shard Qwen run directories were not copied into the bundle

So:

- To inspect or train from final packaged data, the bundle is already enough
- To rerun Qwen filtering from scratch inside the bundle, rerun the
  corresponding SAM3 generation step first so `shard_meta/` is recreated

## Helpful Commands

Show wrapper help:

```bash
cd /path/to/sage_repro_bundle
bash scripts/run_mask_generation_and_filtering.sh help
```

Dry-run any stage:

```bash
cd /path/to/sage_repro_bundle
bash scripts/run_mask_generation_and_filtering.sh vqa-build
```

Actually execute:

```bash
cd /path/to/sage_repro_bundle
RUN=1 bash scripts/run_mask_generation_and_filtering.sh vqa-build
```
