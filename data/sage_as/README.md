---
pretty_name: SAGE-AS
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
- config_name: vqa
  data_files:
  - split: train
    path: data/vqa/train.json
  - split: validation
    path: data/vqa/val.json
  - split: test
    path: data/vqa/test.json
- config_name: gqa
  data_files:
  - split: train
    path: data/gqa/train.jsonl
  - split: validation
    path: data/gqa/val.jsonl
  - split: test
    path: data/gqa/test.jsonl
- config_name: vg
  data_files:
  - split: train
    path: data/vg/train.jsonl
  - split: validation
    path: data/vg/val.jsonl
  - split: test
    path: data/vg/test.jsonl
---

# SAGE-AS Data Package

This repository contains annotation splits for VQA-AS, GQA-AS, and Visual Genome-AS, plus patch-mask NPZ files used for mask supervision experiments.

## Contents

- `data/vqa/train.json`, `data/vqa/val.json`, `data/vqa/test.json`
- `data/gqa/train.jsonl`, `data/gqa/val.jsonl`, `data/gqa/test.jsonl`
- `data/vg/train.jsonl`, `data/vg/val.jsonl`, `data/vg/test.jsonl`
- `masks/vqa_masks.npz`
- `masks/gqa_masks.npz`
- `masks/vg_masks.npz`
- `manifest.json` and `metadata/summary.json`

## Important Notes

Raw or masked image files are not included in this repository. Users must obtain or prepare the corresponding images separately and comply with the licenses of the original datasets.

The JSON/JSONL split files keep all questions from the prepared splits. Questions filtered by Qwen during training are not removed from the JSON/JSONL files.

The mask NPZ files are filtered directly. NPZ rows were removed if the `question_id` was filtered out by Qwen during training or if the sample has `answer_type == "number"`. This filtering applies only to released NPZ mask rows used for mask-supervision metadata; the corresponding QA records remain in the JSON/JSONL train, validation, and test splits.

For GQA-AS and Visual Genome-AS, `image_path` has been sanitized to a relative placeholder under `masked_images/<dataset>/...`; these image files are not included.

## Split Sizes

| Dataset | Train | Validation | Test |
|---|---:|---:|---:|
| VQA-AS | 97,582 | 12,199 | 12,199 |
| GQA-AS | 8,007 | 1,010 | 983 |
| Visual Genome-AS | 8,002 | 996 | 1,002 |

## Mask NPZ Sizes

| Dataset | NPZ rows | Shape |
|---|---:|---|
| VQA-AS | 69,884 | `(69884, 24, 24)` |
| GQA-AS | 5,461 | `(5461, 24, 24)` |
| Visual Genome-AS | 3,772 | `(3772, 24, 24)` |

## Fields

Common fields include `question_id`, `image_id`, `answer_type`, and question/answer text. GQA-AS and Visual Genome-AS also include `text_keywords`, `visual_cues`, `original_answer`, `generated_question`, and `generated_answer`.

The NPZ files contain `question_ids`, `image_ids`, `coverage_ratio`, `has_mask`, and related image padding metadata. `coverage_ratio[i, row, col]` is the fraction of a LLaVA 24x24 visual patch covered by the binary mask after pad-to-square preprocessing.

## License and Upstream Data

This package is distributed as derived research annotations and mask metadata. The underlying source datasets and images retain their original licenses and terms. See `NOTICE.md` for details.
