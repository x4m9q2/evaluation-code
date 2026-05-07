# VQA-CMSV / SAGE 匿名复现代码

本仓库是 VQA-CMSV benchmark 生成流程和 SAGE 实验的匿名代码包。推荐入口是
根目录下的 `scripts/`，这些脚本使用相对路径解析仓库位置；组件目录内的历史
脚本主要用于溯源和排查，不建议作为完整复现实验的启动入口。

本仓库不包含大文件资产：模型权重、原始图片、训练 checkpoint、优化器状态、
生成输出、SAM3 权重、Qwen 权重和 xVerify 权重均需要单独下载。

除 `third_party/` 下的第三方说明外，仓库内运行相关 README 已合并到本文档。
`data/sage_as/README.md` 是 Hugging Face 数据集卡片的本地副本，保留用于数据
集说明，不作为代码运行入口。

## 1. 环境配置

验证过的环境如下：

- Python 3.10
- PyTorch 2.6.0，CUDA 12.4 runtime
- Transformers 4.51.3
- DeepSpeed 0.16.7
- bf16 训练
- SDPA 或 eager attention；FlashAttention2 可选

创建环境：

```bash
conda env create -f environment.yml
conda activate sage-repro
```

如果复用本地 virtualenv，显式指定 Python：

```bash
export PYTHON_BIN="$PWD/.venv_gemma/bin/python"
export PATH="$PWD/.venv_gemma/bin:$PATH"
```

通用约定：

- 所有 wrapper 默认正式执行命令；缺少必需文件时会直接报错退出。
- 所有默认路径均相对仓库根目录，可通过环境变量覆盖。
- 不要随意改 DeepSpeed stage、per-device batch size、gradient
  accumulation、precision、学习率调度或 `max_steps`。这些参数会改变数值稳
  定性和学习率曲线；未验证组合可能触发 NaN/Inf。
- 如果下载时需要代理，只在当前 shell 设置代理环境变量，不要把本地代理地址
  或凭证写入代码或文档。

## 2. 数据集生成

### 2.1 跑数据集生成需要下载的文件

VQA-CMSV 生成需要以下外部文件：

- COCO 2014 train images：`data/images/coco/train2014/`
- COCO 2014 annotations：`annotations/instances_train2014.json`
- VQAv2 train questions：
  `data/detect-shortcuts/data/vqa2/v2_OpenEnded_mscoco_train2014_questions.json`
- VQAv2 train annotations：
  `data/detect-shortcuts/data/vqa2/v2_mscoco_train2014_annotations.json`
- shortcut mining 使用的 GMiner：
  `code/shortcut_pipeline/bin/GMiner`
- shortcut matching 使用的 CUDA matcher：
  `code/shortcut_pipeline/bin/cuda`

官方下载入口：

```text
http://images.cocodataset.org/annotations/annotations_trainval2014.zip
http://images.cocodataset.org/zips/train2014.zip
https://visualqa.org/download.html
```

已发布的数据集可从 Hugging Face 下载：

```text
https://huggingface.co/datasets/as-benchmark-artifacts/vqa-cmsv-benchmark
https://huggingface.co/datasets/as-benchmark-artifacts/vqa-cmsv-benchmark/resolve/main/croissant.json
```

下载后建议保持如下结构：

```text
data/sage_as/
  data/vqa_v2_cmsv/{train,val,test}.json
  data/gqa_cmsv/{train,val,test}.jsonl
  data/vg_cmsv/{train,val,test}.jsonl
  masks/{vqa_v2_cmsv,gqa_cmsv,vg_cmsv}_masks.npz
```

### 2.2 跑数据集生成的流程

阶段一：挖掘文本捷径规则和候选匹配结果。

```bash
bash scripts/run_shortcut_stage1.sh
```

阶段二：基于阶段一结果生成 CMSV 样本。

```bash
bash scripts/run_shortcut_stage2.sh
```

格式转换：将生成结果转换为发布用 VQA v2-CMSV split。

```bash
BATCH_OUTPUT_JSONL=outputs/shortcut_stage2/generated_samples.jsonl \
  bash scripts/run_build_vqa_v2_cmsv_splits.sh
```

如果只需要复现实验、不重新生成数据，可直接下载发布版 split：

```bash
bash scripts/run_download_vqa_v2_cmsv.sh
```

split 语义：

- `train`：训练 split。VQA v2-CMSV 的 train 是主实验二阶段训练 mix，包含
  带 mask 的 CMSV 样本、保留但无 mask 的 CMSV 样本和 VQA train2014
  no-mask 样本。
- `val`：二阶段训练验证 loss 使用的 split。
- `test`：Acc/SR 测评使用的 split。

训练时通过 NPZ 中的 `question_id` 匹配样本是否启用 mask loss。

## 3. LLaVA 实验

### 3.1 跑 LLaVA 所有实验需要准备的文件

必需模型和组件：

- `models/llava-v1.5-7b/`
- `models/clip-vit-large-patch14-336/`
- `models/sam3_ckpt/sam3.pt`，仅生成 mask 时需要

必需数据：

- LLaVA stage-1 原始 mix：
  `data/llava_stage1/llava_v1_5_mix665k.json`
- LLaVA stage-1 图片根目录：`data/playground_data/`
- VQA/GQA/VG CMSV split 和 mask：`data/sage_as/`
- 评测图片根目录：`data/images/`

可选依赖：

- Qwen 模型或 API：用于视觉线索过滤。
- POPE 数据：`data/pope/`
- BEAF 数据：`data/beaf/`
- xVerify 不随仓库分发，打包后的 Acc/SR 脚本不依赖 xVerify。

### 3.2 跑 LLaVA 所有实验的流程和注意事项

#### 3.2.1 一阶段训练数据预处理

生成 image-only、no-OCR、答案长度受限的 LLaVA 预训练 JSON：

```bash
bash scripts/run_build_pretrain_json.sh
```

默认输出：

```text
data/pretrain_llava_v1_5_mix665k_single_noocr_max200_imageonly_strict_noocr.json
```

#### 3.2.2 开启门控的一阶段训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
OUTPUT_DIR=checkpoints/llava_pretrain_gate \
bash scripts/run_llava_pretrain_gate.sh
```

关键默认参数：

- `--use_dual_input_gate True`
- `--tune_mm_mlp_adapter True`
- DeepSpeed `code/llava_sage/scripts/zero2_bf16.json`
- bf16
- `learning_rate=1e-3`

#### 3.2.3 关闭门控的一阶段训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
OUTPUT_DIR=checkpoints/llava_pretrain_nogate \
bash scripts/run_llava_pretrain_nogate.sh
```

该脚本与门控版使用相同预训练数据和优化配置，只关闭门控和 L1 相关项。

#### 3.2.4 模型组装

如果一阶段只保存 projector 或 gate adapter，需要组装成可直接加载的 LLaVA
checkpoint：

```bash
ASSEMBLE_ADAPTER_PATH=checkpoints/llava_pretrain_gate/mm_projector.bin \
ASSEMBLE_OUTPUT_PATH=checkpoints/llava_pretrain_gate_assembled \
bash scripts/run_assemble_llava_checkpoint.sh
```

`ASSEMBLE_FORCE_GATE=auto` 会自动检测 adapter 中是否包含 gate 权重。

#### 3.2.5 Qwen 过滤数据

Qwen 过滤用于判断 mask supervision 是否可靠：

```bash
bash scripts/run_qwen_visual_cue_filter.sh vqa
bash scripts/run_qwen_visual_cue_filter.sh gqa
bash scripts/run_qwen_visual_cue_filter.sh vg
```

Qwen 过滤只影响 NPZ mask rows 和训练时是否启用 mask loss，不删除 JSON/JSONL
中的 QA 样本。

#### 3.2.6 掩码 NPZ 生成

VQA mask 生成、过滤和打包：

```bash
bash scripts/run_mask_generation_and_filtering.sh vqa-generate
bash scripts/run_mask_generation_and_filtering.sh vqa-filter
bash scripts/run_mask_generation_and_filtering.sh vqa-build
```

GQA/VG mask 生成、过滤和打包：

```bash
bash scripts/run_mask_generation_and_filtering.sh gqa-vg-generate
bash scripts/run_mask_generation_and_filtering.sh gqa-vg-filter
bash scripts/run_mask_generation_and_filtering.sh gqa-vg-build
```

mask 规则：

- 训练时以 NPZ 中的 `question_id` 匹配作为 mask supervision 的依据。
- JSON/JSONL 中的 `mask_supervision` 只是可读 metadata，不是唯一依据。
- number-answer 样本不会从 QA split 删除，只会从 NPZ 中移除对应 mask row，
  或在训练时视作 no-mask supervision。

#### 3.2.7 二阶段训练

门控 SAGE：

```bash
SAGE_AS_DATASET=vqa \
LLAVA_PRETRAIN_PROJECTOR=checkpoints/llava_pretrain_gate/mm_projector.bin \
LLAVA_STAGE2_CHECKPOINT=checkpoints/llava_stage2_sage_vqa \
bash scripts/run_llava_stage2_mask_sft.sh
```

非门控对照：

```bash
SAGE_AS_DATASET=vqa \
LLAVA_STAGE2_NOGATE_CHECKPOINT=checkpoints/llava_stage2_nogate_vqa \
bash scripts/run_llava_stage2_mask_sft_nogate.sh
```

可把 `SAGE_AS_DATASET` 改为 `gqa` 或 `vg`。当前 LLaVA 二阶段默认训练 2
epoch，并使用 `LR_SCHEDULER_TOTAL_STEPS_SCALE=1.5`，使学习率曲线对齐原
3 epoch 训练的前 2 epoch。不要用改 `max_steps` 的方式替代正式训练，否则
学习率调度不等价。验证集 loss 默认按 epoch 评估。

#### 3.2.8 Acc/SR 测评

LLaVA Acc/SR wrapper 支持 VQA、GQA 和 VG：

```bash
LLAVA_EVAL_DATASET=vqa \
MODEL_PATH=checkpoints/llava_stage2_sage_vqa \
bash scripts/run_llava_eval_acc_sr.sh
```

常用覆盖项：

- `LLAVA_EVAL_DATASET=vqa|gqa|vg`
- `MODEL_PATH=...`
- `HAS_GATE=auto|true|false`
- `TORCH_DTYPE=bf16`

#### 3.2.9 NaPO 跑法

构建 LLaVA NaPO preference 数据：

```bash
SAGE_AS_DATASET=vqa bash scripts/run_build_shortcut_napo_llava_dataset.sh
```

启动 NaPO：

```bash
SAGE_AS_DATASET=vqa \
NAPO_LLAVA_OUTPUT_ROOT=checkpoints/napo_llava_vqa \
bash scripts/run_napo_llava.sh
```

NaPO 数据约定：负样本使用 `original_answer`，正样本使用
`generated_answer`。

#### 3.2.10 CausalMM

CausalMM 在 CMSV test split 上做 plug-and-play 推理：

```bash
LLAVA_EVAL_DATASET=vqa \
MODEL_PATH=models/llava-v1.5-7b \
bash scripts/run_cmsv_causalmm_llava.sh
```

可把 `LLAVA_EVAL_DATASET` 改为 `gqa` 或 `vg`。该流程与 POPE/BEAF 无关，也
不依赖 xVerify。

#### 3.2.11 BEAF 和 POPE

普通 SAGE/LLaVA POPE：

```bash
MODEL_PATH=checkpoints/llava_stage2_sage_vqa \
bash scripts/run_pope_eval.sh
```

CausalMM-LLaVA POPE：

```bash
MODEL_PATH=models/llava-v1.5-7b \
bash scripts/run_pope_causalmm_llava.sh
```

BEAF：

```bash
MODEL_PATH=checkpoints/llava_stage2_sage_vqa \
bash scripts/run_beaf_eval.sh
```

POPE/BEAF 的原始数据和图片不随仓库分发，需要按官方协议单独获取。

## 4. Gemma 实验

### 4.1 跑 Gemma 所有实验需要准备的文件

必需模型：

- `models/Gemma-3-4B-IT/`
- `models/siglip-so400m-patch14-384/`

必需数据：

- 预训练 JSON：
  `data/pretrain_llava_v1_5_mix665k_single_noocr_max200_imageonly_strict_noocr.json`
- 预训练图片根目录：`data/playground_data/`
- VQA/GQA/VG CMSV split 和 mask：`data/sage_as/`
- 二阶段图片根目录：`data/images/`

可选依赖：

- Qwen 模型或 API：用于过滤 mask supervision。
- NaPO 训练数据：可由本仓库脚本从 CMSV split 构建。
- CausalMM 评测输入：由 CMSV test split 转换得到。

### 4.2 跑 Gemma 所有实验的流程和注意事项

#### 4.2.1 一阶段训练数据预处理

Gemma 与 LLaVA 复用同一个预训练 JSON：

```bash
bash scripts/run_build_pretrain_json.sh
```

#### 4.2.2 开启门控的一阶段训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PRETRAIN_CHECKPOINT=checkpoints/gemma3_4b_pretrain_gate_projector_l1_sdpa \
bash scripts/run_pretrain_gate.sh
```

默认使用 bf16、SDPA、4 GPU、`learning_rate=1e-3`。

#### 4.2.3 关闭门控的一阶段训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PRETRAIN_NOGATE_CHECKPOINT=checkpoints/gemma3_4b_pretrain_projector_sdpa \
bash scripts/run_pretrain_nogate.sh
```

该脚本不启用门控，不启用 L1 loss，其余预训练配置尽量与门控版一致。

#### 4.2.4 模型组装

Gemma 默认 full fine-tuning 脚本会直接保存可加载 checkpoint，通常不需要额外
组装。若使用 LoRA 或 adapter-only 变体，应按对应上游 Gemma fine-tuning 工
具执行 merge。

#### 4.2.5 Qwen 过滤数据

Gemma 与 LLaVA 使用同一套 Qwen 过滤结果和 NPZ mask：

```bash
bash scripts/run_qwen_visual_cue_filter.sh vqa
bash scripts/run_mask_generation_and_filtering.sh vqa-build
```

过滤语义同 LLaVA：只影响 NPZ mask rows 和 mask loss 是否启用，不删除 QA
样本。

#### 4.2.6 掩码 NPZ 生成

Gemma 读取 `PATCH_MASK_ANALYSIS_PATH` 指向的 NPZ，并通过 `question_id`
匹配训练样本：

```bash
bash scripts/run_mask_generation_and_filtering.sh vqa-generate
bash scripts/run_mask_generation_and_filtering.sh vqa-build
```

GQA/VG：

```bash
bash scripts/run_mask_generation_and_filtering.sh gqa-vg-generate
bash scripts/run_mask_generation_and_filtering.sh gqa-vg-build
```

#### 4.2.7 二阶段训练

门控 SAGE：

```bash
SAGE_AS_DATASET=vqa \
PRETRAIN_CHECKPOINT=checkpoints/gemma3_4b_pretrain_gate_projector_l1_sdpa \
STAGE2_CHECKPOINT=checkpoints/gemma3_4b_stage2_gate_l1_mask_sdpa_vqa \
bash scripts/run_stage2_sft_gate.sh
```

非门控对照：

```bash
SAGE_AS_DATASET=vqa \
PRETRAIN_NOGATE_CHECKPOINT=checkpoints/gemma3_4b_pretrain_projector_sdpa \
STAGE2_NOGATE_CHECKPOINT=checkpoints/gemma3_4b_stage2_nogate_sdpa_vqa \
bash scripts/run_stage2_sft_nogate.sh
```

可把 `SAGE_AS_DATASET` 改为 `gqa` 或 `vg`。训练日志中应能看到有监督样本的 `mask_patch_loss` 非 0，且总体 loss 为有限值。

#### 4.2.8 Acc/SR 测评

单进程 Gemma 评测：

```bash
SAGE_AS_DATASET=vqa \
MODEL_ID=checkpoints/gemma3_4b_stage2_gate_l1_mask_sdpa_vqa \
bash scripts/run_eval_test_raw.sh
```

4 GPU 分片推理：

```bash
SAGE_AS_DATASET=vqa \
MODEL_ID=checkpoints/gemma3_4b_stage2_gate_l1_mask_sdpa_vqa \
bash scripts/run_eval_test_raw_4gpu.sh
```

#### 4.2.9 NaPO 跑法

构建 Gemma NaPO 数据：

```bash
SAGE_AS_DATASET=vqa bash scripts/run_build_shortcut_napo_splits.sh
```

启动 NaPO：

```bash
SAGE_AS_DATASET=vqa \
NAPO_DATA=data/napo/train_raw_pos_neg_shortcut.json \
bash scripts/run_napo_shortcut.sh
```

#### 4.2.10 CausalMM

Gemma CausalMM：

```bash
SAGE_AS_DATASET=vqa \
MODEL_PATH=models/Gemma-3-4B-IT \
bash scripts/run_cmsv_causalmm_gemma.sh
```

Gemma 适配默认使用 language-side counterfactual attention：

```text
causalmm_logits = (1 + gamma) * logits - gamma * cf_logits
```

可通过 `CF_MODE`、`ATTENTION_METHOD`、`GAMMA`、`EPSILON` 覆盖默认设置。

## 5. 对所有依赖的协议说明

本仓库只分发匿名代码、配置、小型 metadata 和 wrapper 脚本。使用者需要自行
下载外部数据、图片、模型权重和评测资源，并遵守对应许可证或使用条款。

主要依赖说明：

- LLaVA 相关代码位于 `code/llava_sage/`；复用时需保留上游许可证和引用信息。
- Gemma fine-tuning 相关代码位于 `code/gemma_gate/gemma/`；复用时需保留上游
  许可证和引用信息。Gemma 权重受 Gemma 模型条款约束，本仓库不分发权重。
- CausalMM-LLaVA 代码位于 `code/evaluation/causalmm_llava/`；许可证文件保留
  在该目录下，使用时应引用 CausalMM 原论文。
- Gemma CausalMM 适配代码位于 `code/beaf_causalmm/gemma3/`，是对 CausalMM
  counterfactual decoding 思路的 Gemma 适配。
- NaPO 参考实现位于 `third_party/napo_llava_ref/`，作为第三方代码快照保留，
  需要保留上游说明和许可证。
- POPE 和 BEAF 是外部幻觉评测资源；本仓库只保留调用脚本，不分发数据。
- SAM3 源码和 Qwen 过滤相关脚本随代码包提供；SAM3 checkpoint、Qwen 模型
  权重、Qwen API key 和 xVerify 权重不随仓库分发。xVerify 相关脚本仅保留
  数据格式转换和结果统计 helper，默认 Acc/SR 流程不依赖可运行的 xVerify
  模型。
- VQA-CMSV 数据包只包含派生 QA annotation 和 mask metadata，不包含原始或
  masked 图片。COCO、VQAv2、GQA、Visual Genome 等上游数据仍受其原始许可
  证约束。
