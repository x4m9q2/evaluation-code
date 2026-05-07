# Shortcut Pipeline Overview

代码在 `code/shortcut_pipeline/`，默认产物在 `data/shortcut_pipeline/`。
规范入口在 `scripts/`：

- `scripts/run_shortcut_stage1.sh`
- `scripts/run_shortcut_stage2.sh`

## 阶段 1

目标：从 VQA2/COCO 挖 shortcut rule，并生成带 `answer_type` 的合并结果。

输入：

- `annotations/instances_train2014.json`
- `data/detect-shortcuts/data/vqa2/v2_OpenEnded_mscoco_train2014_questions.json`
- `data/detect-shortcuts/data/vqa2/v2_mscoco_train2014_annotations.json`
- `code/shortcut_pipeline/bin/GMiner`
- `code/shortcut_pipeline/bin/cuda`

步骤：

1. `transfer_detection.py`
   - 把 COCO `instances_train2014.json` 转成 `image_to_detection.json`
2. `generate_rules_json.py`
   - 基于问题文本 token、视觉类别 token 和答案挖规则
   - 输出 `rules/rules.json`
3. CUDA matcher
   - 用 `rules.json` 匹配问题
   - 输出 `shortcuts_matches.json`
   - 每题保留最高置信度规则
4. `merge_shortcuts_output.py`
   - 合并规则、匹配结果、问题和标注
   - 输出 `gqa_merged_output_with_answer_type.json`

默认产物：

- `data/shortcut_pipeline/image_to_detection.json`
- `data/shortcut_pipeline/rules/rules.json`
- `data/shortcut_pipeline/shortcuts_matches.json`
- `data/shortcut_pipeline/gqa_merged_output_with_answer_type.json`

阶段 1 一键运行：

```bash
bash scripts/run_shortcut_stage1.sh
```

如果需要限制样本数，直接设置环境变量：

```bash
STAGE1_LIMIT=384 bash scripts/run_shortcut_stage1.sh
```

当 `PREPARE_STAGE2_MASKS=1` 时，阶段 1 会继续生成阶段 2 所需输入和 mask。

## 阶段 2

目标：筛出同时包含 `text_keywords` 和 `visual_cues` 的样本，遮掉所有 `visual_cues` 对应区域，再生成跨模态 QA 请求。

输入：

- `data/shortcut_pipeline/gqa_merged_output_with_answer_type.json`
- `data/detect-shortcuts/data/vqa2/v2_OpenEnded_mscoco_train2014_questions.json`
- `data/images/coco/train2014/`
- `models/sam3_ckpt/sam3.pt`

参与规则：

- 仅当样本同时满足：
  - `text_keywords` 非空
  - `visual_cues` 非空
- 该样本才会进入阶段 2

步骤：

1. `prepare_stage2_inputs.py`
   - 从阶段 1 合并结果中过滤出可参与样本
   - 补回原始 question text
   - 输出：
     - `cross_modality_qa_input.json`
     - `cross_modality_qa_questions.jsonl`
     - `cross_modality_qa_mapping.json`
2. `code/sam3/scripts/generate_union_masks_from_mapping.py`
   - 对每条样本的全部 `visual_cues` 生成 mask
   - 做并集，得到每题一个 union mask
3. `apply_union_masks_to_images.py`
   - 用 union mask 把原图对应像素全部置黑
   - 输出 `output_mask/<question_id>_<image_id>.png`
4. `prepare_gqa_batch_requests.py`
   - 读取 masked image 和阶段 2 输入
   - 生成 `/v1/responses` 批量请求 JSONL

默认产物：

- `data/shortcut_pipeline/cross_modality_qa_input.json`
- `data/shortcut_pipeline/cross_modality_qa_questions.jsonl`
- `data/shortcut_pipeline/cross_modality_qa_mapping.json`
- `data/shortcut_pipeline/union_mask/`
- `data/shortcut_pipeline/output_mask/`
- `data/shortcut_pipeline/batch_inputs/cross_modality_qa_requests.jsonl`

阶段 2 一键运行：

```bash
bash scripts/run_shortcut_stage2.sh
```

限制生成数量：

```bash
STAGE2_LIMIT=28 bash scripts/run_shortcut_stage2.sh
```

## 补充说明

- 阶段 2 prompt 不传 `visual_cues`。
- `visual_cues` 只参与筛样本和做遮挡。
- 阶段 1 的 `generate_rules_json.py` 已并入 `code/shortcut_pipeline/`。
- `GMiner` 来自：
  `https://github.com/cdancette/detect-shortcuts`
