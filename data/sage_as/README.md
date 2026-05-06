---
pretty_name: VQA-CMSV Benchmark
license: other
language:
- en
task_categories:
- visual-question-answering
tags:
- vision-language
- visual-question-answering
- shortcut-bias
- patch-mask
- llava
size_categories:
- 100K<n<1M
configs:
- config_name: vqa_v2_cmsv
  data_files:
  - split: train
    path: data/vqa_v2_cmsv/train.json
  - split: validation
    path: data/vqa_v2_cmsv/val.json
  - split: test
    path: data/vqa_v2_cmsv/test.json
- config_name: gqa_cmsv
  data_files:
  - split: train
    path: data/gqa_cmsv/train.jsonl
  - split: validation
    path: data/gqa_cmsv/val.jsonl
  - split: test
    path: data/gqa_cmsv/test.jsonl
- config_name: vg_cmsv
  data_files:
  - split: train
    path: data/vg_cmsv/train.jsonl
  - split: validation
    path: data/vg_cmsv/val.jsonl
  - split: test
    path: data/vg_cmsv/test.jsonl
---

# VQA-CMSV Benchmark Data Package

This repository contains annotation splits for VQA v2-CMSV, GQA-CMSV, and VG-CMSV, plus patch-mask NPZ files used for mask supervision experiments.

## Contents

- `data/vqa_v2_cmsv/train.json`, `data/vqa_v2_cmsv/val.json`, `data/vqa_v2_cmsv/test.json`
- `data/gqa_cmsv/train.jsonl`, `data/gqa_cmsv/val.jsonl`, `data/gqa_cmsv/test.jsonl`
- `data/vg_cmsv/train.jsonl`, `data/vg_cmsv/val.jsonl`, `data/vg_cmsv/test.jsonl`
- `masks/vqa_v2_cmsv_masks.npz`
- `masks/gqa_cmsv_masks.npz`
- `masks/vg_cmsv_masks.npz`
- `manifest.json` and `metadata/summary.json`

## Important Notes

Raw or masked image files are not included in this repository. Users must obtain or prepare the corresponding images separately and comply with the licenses of the original datasets.

`data/vqa_v2_cmsv/train.json` is the stage-2 experiment training mix used in the main SAGE runs. It contains `219,562` records: `75,196` masked generated-CMSV records with SAM3 patch-mask supervision, `22,386` retained generated-CMSV records without mask supervision, and `121,980` VQA train2014 no-mask records.

The JSON/JSONL split files keep all questions from the prepared splits. Questions filtered by Qwen during training are not removed from the JSON/JSONL files.

The mask NPZ files are filtered directly. NPZ rows were removed if the `question_id` was filtered out by Qwen during training or if the sample has `answer_type == "number"`. This filtering applies only to released NPZ mask rows used for mask-supervision metadata; the corresponding QA records remain in the JSON/JSONL train, validation, and test splits.

For GQA-CMSV and VG-CMSV, `image_path` has been sanitized to a relative placeholder under `masked_images/<dataset>/...`; these image files are not included.

## Split Sizes

| Dataset | Train | Validation | Test |
|---|---:|---:|---:|
| VQA v2-CMSV | 219,562 | 12,199 | 12,199 |
| GQA-CMSV | 8,007 | 1,010 | 983 |
| VG-CMSV | 8,002 | 996 | 1,002 |

## Mask NPZ Sizes

| Dataset | NPZ rows | Shape |
|---|---:|---|
| VQA v2-CMSV | 69,884 | `(69884, 24, 24)` |
| GQA-CMSV | 5,461 | `(5461, 24, 24)` |
| VG-CMSV | 3,772 | `(3772, 24, 24)` |

## Fields

Common fields include `question_id`, `image_id`, `answer_type`, and question/answer text. GQA-CMSV and VG-CMSV also include `text_keywords`, `visual_cues`, `original_answer`, `generated_question`, and `generated_answer`.

The NPZ files contain `question_ids`, `image_ids`, `coverage_ratio`, `has_mask`, and related image padding metadata. `coverage_ratio[i, row, col]` is the fraction of a LLaVA 24x24 visual patch covered by the binary mask after pad-to-square preprocessing.

## License and Upstream Data

This package is distributed as derived research annotations and mask metadata. The underlying source datasets and images retain their original licenses and terms. See `NOTICE.md` for details.
