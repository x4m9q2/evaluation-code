# Shortcut Pipeline Overview

The code lives under `code/shortcut_pipeline/`, and default artifacts are
written to `data/shortcut_pipeline/`.
The canonical entry points are under `scripts/`:

- `scripts/run_shortcut_stage1.sh`
- `scripts/run_shortcut_stage2.sh`

## Stage 1

Goal: mine shortcut rules from VQA2/COCO and generate merged outputs annotated
with `answer_type`.

Inputs:

- `annotations/instances_train2014.json`
- `data/detect-shortcuts/data/vqa2/v2_OpenEnded_mscoco_train2014_questions.json`
- `data/detect-shortcuts/data/vqa2/v2_mscoco_train2014_annotations.json`
- `code/shortcut_pipeline/bin/GMiner`
- `code/shortcut_pipeline/bin/cuda`

Steps:

1. `transfer_detection.py`
   - Convert COCO `instances_train2014.json` into `image_to_detection.json`
2. `generate_rules_json.py`
   - Mine rules from question-text tokens, visual-category tokens, and answers
   - Output `rules/rules.json`
3. CUDA matcher
   - Match questions against `rules.json`
   - Output `shortcuts_matches.json`
   - Keep only the highest-confidence rule for each question
4. `merge_shortcuts_output.py`
   - Merge rules, match results, questions, and annotations
   - Output `gqa_merged_output_with_answer_type.json`

Default artifacts:

- `data/shortcut_pipeline/image_to_detection.json`
- `data/shortcut_pipeline/rules/rules.json`
- `data/shortcut_pipeline/shortcuts_matches.json`
- `data/shortcut_pipeline/gqa_merged_output_with_answer_type.json`

Run stage 1:

```bash
bash scripts/run_shortcut_stage1.sh
```

To limit the sample count, set the environment variable directly:

```bash
STAGE1_LIMIT=384 bash scripts/run_shortcut_stage1.sh
```

When `PREPARE_STAGE2_MASKS=1`, stage 1 continues to generate the inputs and
masks required by stage 2.

## Stage 2

Goal: keep samples that contain both `text_keywords` and `visual_cues`, mask
all regions associated with `visual_cues`, and then generate cross-modal QA
requests.

Inputs:

- `data/shortcut_pipeline/gqa_merged_output_with_answer_type.json`
- `data/detect-shortcuts/data/vqa2/v2_OpenEnded_mscoco_train2014_questions.json`
- `data/images/coco/train2014/`
- `models/sam3_ckpt/sam3.pt`

Eligibility rules:

- A sample enters stage 2 only if both conditions are true:
  - `text_keywords` is non-empty
  - `visual_cues` is non-empty

Steps:

1. `prepare_stage2_inputs.py`
   - Filter eligible samples from the stage-1 merged output
   - Restore the original question text
   - Outputs:
     - `cross_modality_qa_input.json`
     - `cross_modality_qa_questions.jsonl`
     - `cross_modality_qa_mapping.json`
2. `code/sam3/scripts/generate_union_masks_from_mapping.py`
   - Generate masks for all `visual_cues` in each sample
   - Take their union so each question gets one union mask
3. `apply_union_masks_to_images.py`
   - Set all source-image pixels covered by the union mask to black
   - Output `output_mask/<question_id>_<image_id>.png`
4. `prepare_gqa_batch_requests.py`
   - Read the masked images and stage-2 inputs
   - Generate batch-request JSONL for `/v1/responses`

Default artifacts:

- `data/shortcut_pipeline/cross_modality_qa_input.json`
- `data/shortcut_pipeline/cross_modality_qa_questions.jsonl`
- `data/shortcut_pipeline/cross_modality_qa_mapping.json`
- `data/shortcut_pipeline/union_mask/`
- `data/shortcut_pipeline/output_mask/`
- `data/shortcut_pipeline/batch_inputs/cross_modality_qa_requests.jsonl`

Run stage 2:

```bash
bash scripts/run_shortcut_stage2.sh
```

To limit the number of generated samples:

```bash
STAGE2_LIMIT=28 bash scripts/run_shortcut_stage2.sh
```

## Additional Notes

- Stage-2 prompts do not include `visual_cues`.
- `visual_cues` are used only for sample filtering and masking.
- Stage-1 `generate_rules_json.py` has already been merged into
  `code/shortcut_pipeline/`.
- `GMiner` comes from:
  `https://github.com/cdancette/detect-shortcuts`
