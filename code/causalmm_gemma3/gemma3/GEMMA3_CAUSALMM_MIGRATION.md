# CausalMM 适配 Gemma 3 的实现说明与迁移文档

本文档说明当前工程里是如何把 `CausalMM` 的解码思路适配到 `Gemma 3` 上的，并列出所有相关代码所在位置，方便迁移到其他机器或其他目录。

当前工作区根目录是 `.`。
文中出现的相对路径，默认都相对于这个根目录。

## 1. 结论先说

当前这套 `Gemma 3 + CausalMM` 代码有两个核心特点：

1. 没有改 Gemma 3 的模型权重，只改了推理期的前向和解码逻辑。
2. 主要是把原始 `CausalMM` 的 counterfactual decoding 思路，移植到了 Hugging Face `transformers` 的 `Gemma3ForConditionalGeneration` 上。

这套实现不是通过训练新模型完成的，而是通过下面几步完成适配：

1. 用 `AutoProcessor` 和 `Gemma3ForConditionalGeneration` 加载 Gemma 3。
2. 强制 Gemma 3 的文本注意力走 `eager` 路径，而不是纯 `sdpa/flash` 路径。
3. monkey patch `transformers.models.gemma3.modeling_gemma3.eager_attention_forward`，在 decoder attention 的 softmax 之后注入 counterfactual attention。
4. 手工构造 normal branch 和 counterfactual branch，两路各自维护 KV cache。
5. 按 `CausalMM` 的公式把两路 logits 合成最终 logits，再逐 token 解码。
6. 对视觉侧，普通版为了省显存，优先改 image soft tokens；专门的 `attn_shuffle` 实验版才显式读取 Gemma 3 vision tower 的 attention 并做 patch-weight shuffle。

## 2. 和原始 CausalMM 的关系

原始参考代码主要在：

- `CausalMM/llava-1.5/causalmm_cf/causalmm_sm.py`
- `CausalMM/llava-1.5/causalmm_cf/causalmm_mm.py`
- `CausalMM/README.md`

其中：

- `causalmm_sm.py` 是单 counterfactual branch 的思路，核心公式是：

```text
causalmm_logits = (1 + gamma) * logits - gamma * cf_logits
```

- `causalmm_mm.py` 是更重的多分支版本。

当前 `Gemma 3` 适配版，核心上更接近 `causalmm_sm.py`，也就是：

1. 一条 normal branch。
2. 一条 counterfactual branch。
3. 用 `(1 + gamma) * logits - gamma * cf_logits` 合成最终分布。

差别在于 `Gemma 3` 版把 counterfactual branch 的干预位置做成了可切换：

- `cf_mode=language`
- `cf_mode=vision`
- `cf_mode=both`

也就是说，当前 `Gemma 3` 版不是把原始 LLaVA 代码逐行搬过来，而是保留了 `CausalMM` 的核心思想，然后按照 `Gemma 3` 的模型接口重新实现了一套推理逻辑。

## 3. 适配是怎么做的

### 3.1 加载 Gemma 3，并把文本注意力切到 eager

核心代码：

- `CausalMM/llava-1.5/gemma3/causalmm_gemma3.py`

关键点：

1. 初始化时加载 `AutoProcessor` 和 `Gemma3ForConditionalGeneration`。
2. 模型加载时虽然传了 `attn_implementation="sdpa"`，但随后又把：
   - `self.model.config.text_config._attn_implementation = "eager"`
   - `self.model.language_model.config._attn_implementation = "eager"`
   强制改成 `eager`。

这样做的原因是：

- `CausalMM` 需要直接接管 decoder attention weights。
- 如果一直走 `sdpa/flash attention`，就很难在合适的位置稳定地改 attention weights。
- `eager` 路径会走到 `transformers.models.gemma3.modeling_gemma3.eager_attention_forward`，这正好给了一个明确的 hook 点。

如果以后迁移到别的环境，`transformers` 版本一旦变化，这里是最容易失效的地方之一。

### 3.2 monkey patch Gemma 3 的 decoder attention

核心代码仍在：

- `CausalMM/llava-1.5/gemma3/causalmm_gemma3.py`

关键函数：

- `_ORIGINAL_EAGER_ATTENTION`
- `apply_counterfactual_attention`
- `causalmm_eager_attention_forward`
- `causalmm_attention`

工作方式：

1. 先保存原始的 `modeling_gemma3.eager_attention_forward`。
2. 进入 counterfactual branch 时，用 context manager 暂时把它替换成 `causalmm_eager_attention_forward`。
3. 在这个新 forward 里：
   - 先正常算 `QK^T`
   - 再加 causal mask
   - 再做 softmax
   - 然后对已经 softmax 后的 attention weights 做 counterfactual 编辑
   - 最后再乘 `V`

目前支持的方法有：

- `none`
- `reverse`
- `reverse_and_normalize`
- `random`
- `uniform`
- `shuffle`

这里特别注意一点：

- 编辑 attention 时，不是对整行随便改，而是用 `allowed` mask 只在合法的 causal 区域内改。
- 改完后会重新归一化，避免破坏注意力分布。

### 3.3 手工构造 Gemma 3 的多模态输入

Gemma 3 不是直接给你一个可以随便插视觉 counterfactual 的高层接口，所以这里做了手工拼装。

核心代码：

- `CausalMM/llava-1.5/gemma3/causalmm_gemma3.py`

关键函数：

- `build_messages`
- `prepare_inputs`
- `prepare_batch_inputs`
- `_merge_image_features`

处理流程：

1. 用 `processor.apply_chat_template(...)` 生成 `input_ids`、`attention_mask`、`pixel_values`。
2. 用 `self.model.get_input_embeddings()` 先得到文本 embedding。
3. 用 `self.model.get_image_features(pixel_values)` 得到 Gemma 3 的 image soft tokens。
4. 根据 `image_token_index` 找到文本里代表图片的位置。
5. 用 `masked_scatter` 把 image features 填回输入 embedding 中。

这样做的结果是：

- normal branch 和 counterfactual branch 都能共用同一套输入模板。
- 你只需要决定 counterfactual branch 用哪种 image features 即可。

### 3.4 normal branch 和 counterfactual branch 分开跑

核心代码：

- `CausalMM/llava-1.5/gemma3/causalmm_gemma3.py`

关键函数：

- `_new_cache`
- `_forward`
- `_combine_logits`
- `_sample_next_token`
- `generate`
- `generate_batch`

实现方式：

1. normal branch 建一套 `HybridCache`。
2. counterfactual branch 再建一套 `HybridCache`。
3. prompt 阶段两路都先各跑一次完整前向。
4. 解码阶段按 token 循环：
   - normal branch 出 `logits`
   - counterfactual branch 出 `cf_logits`
   - 用 `CausalMM` 公式合成最终 logits
   - 从最终 logits 里采样或取 argmax
   - 再把新 token 喂回两路 cache

这里保留了 `epsilon` cutoff 逻辑，也就是只有原始 logits 足够高的 token 才会参与最终竞争。

### 3.5 通用版视觉 counterfactual：优先改 image features，不直接改完整 vision attention

核心代码：

- `CausalMM/llava-1.5/gemma3/causalmm_gemma3.py`

关键函数：

- `_edit_image_features`

支持的方法：

- `shuffle`
- `uniform`
- `reverse`
- `random`
- `none`

这里没有默认直接去改 Gemma 3 vision tower 的完整 attention map，原因很现实：

- Gemma 3 的视觉编码器是高分辨率 SigLIP 路线。
- 如果每次都显式拿出大尺寸 attention map，再做逐 head / 逐 patch 操作，显存和速度成本都很重。

所以通用版更偏工程折中：

- 直接对 image soft tokens 做 counterfactual 编辑。

### 3.6 专门的 vision attention shuffle 实验版

核心代码：

- `CausalMM/llava-1.5/gemma3/eval_test_raw_gemma3_attn_shuffle.py`

这个文件不是简单调参数，而是专门 subclass 了一套 runner：

- `Gemma3AttentionShuffleCausalMM(CausalMMGemma3)`

关键逻辑在：

- `_counterfactual_image_features_from_attention`

做法是：

1. 调 `self.model.vision_tower(..., output_attentions=True)` 显式拿 vision attention。
2. Gemma 3 / SigLIP 的 attention shape 是：

```text
[batch, heads, query_patches, key_patches]
```

3. 因为完整 attention 太大，所以没有直接保留每个 head 的 2D attention map，而是先聚合成 patch importance：

```text
patch_weights = attention.mean(dim=(1, 2))
```

4. 再把这个 patch importance 做归一化。
5. 再 shuffle patch 顺序。
6. 再把 shuffle 后的 patch weights 乘回 `vision_hidden`。
7. 最后送入 `self.model.multi_modal_projector(...)`，得到 counterfactual image soft tokens。

这一步是当前 `attn_shuffle` 实验和通用版 `vision shuffle` 的最大区别：

- 通用版：改的是已经得到的 image features。
- `attn_shuffle` 实验版：显式从 vision attention 提 patch importance，再去重加权视觉 hidden states。

### 3.7 为什么 4 卡分片版本能和单卡保持一致

还是在：

- `CausalMM/llava-1.5/gemma3/eval_test_raw_gemma3_attn_shuffle.py`

关键函数：

- `select_shard`
- `build_sample_seed`

做法：

1. 数据按 `source_index % num_shards` 做分片。
2. 每个样本的 shuffle seed 不是只靠全局随机数，而是由：

```text
sample_seed = f(base_seed, question_id)
```

生成。

这样可以保证：

- 单卡整集跑
- 多卡按 shard 跑

只要 `question_id` 和 `base_seed` 一样，单样本上的 shuffle 结果就一致。

## 4. 代码都在哪里

下面按“是否真正参与 Gemma 3 + CausalMM 运行”来列。

### 4.1 核心运行代码

这些是最重要的代码。

| 作用 | 相对路径 | 当前绝对路径 | 迁移时是否必须 |
|---|---|---|---|
| Gemma 3 + CausalMM 核心实现 | `CausalMM/llava-1.5/gemma3/causalmm_gemma3.py` | `code/causalmm_gemma3/gemma3/causalmm_gemma3.py` | 必须 |
| 单图命令行入口 | `CausalMM/llava-1.5/gemma3/chat_gemma3_causalmm.py` | `code/causalmm_gemma3/gemma3/chat_gemma3_causalmm.py` | 按需 |
| CausalMM API 服务 | `CausalMM/llava-1.5/gemma3/api_gemma3_causalmm.py` | `code/causalmm_gemma3/gemma3/api_gemma3_causalmm.py` | 按需 |
| 启动 CausalMM API 的脚本 | `CausalMM/llava-1.5/gemma3/run_causalmm_server.sh` | `code/causalmm_gemma3/gemma3/run_causalmm_server.sh` | 按需 |
| 普通 CausalMM 评测脚本 | `CausalMM/llava-1.5/gemma3/eval_test_raw_gemma3_causalmm.py` | `code/causalmm_gemma3/gemma3/eval_test_raw_gemma3_causalmm.py` | 做评测时需要 |
| vision attention shuffle 评测脚本 | `CausalMM/llava-1.5/gemma3/eval_test_raw_gemma3_attn_shuffle.py` | `code/causalmm_gemma3/gemma3/eval_test_raw_gemma3_attn_shuffle.py` | 做 `attn_shuffle` 实验时需要 |
| 分片结果合并脚本 | `CausalMM/llava-1.5/gemma3/merge_eval_test_raw_shards.py` | `code/causalmm_gemma3/gemma3/merge_eval_test_raw_shards.py` | 做多卡分片评测时需要 |
| 4 卡运行脚本 | `CausalMM/llava-1.5/gemma3/run_eval_test_raw_gemma3_attn_shuffle_4gpu.sh` | `code/causalmm_gemma3/gemma3/run_eval_test_raw_gemma3_attn_shuffle_4gpu.sh` | 做 4 卡 `attn_shuffle` 时需要 |
| 目录说明 | `CausalMM/llava-1.5/gemma3/README.md` | `code/causalmm_gemma3/gemma3/README.md` | 可选 |

### 4.2 Gemma 3 的本地基础部署代码

这部分不是 `CausalMM` 运行时必须 import 的代码，但当前工程里默认模型路径依赖这套目录。

| 作用 | 相对路径 | 当前绝对路径 | 迁移时是否必须 |
|---|---|---|---|
| 基础 Gemma 3 FastAPI 服务 | `gemma3_modelscope_deploy/app.py` | `models/Gemma-3-4B-IT/app.py` | 可选 |
| 模型下载脚本 | `gemma3_modelscope_deploy/download_model.py` | `models/Gemma-3-4B-IT/download_model.py` | 可选 |
| 基础服务启动脚本 | `gemma3_modelscope_deploy/run_server.sh` | `models/Gemma-3-4B-IT/run_server.sh` | 可选 |
| 依赖列表 | `gemma3_modelscope_deploy/requirements.txt` | `models/Gemma-3-4B-IT/requirements.txt` | 建议带走 |
| Gemma 3 本地权重目录 | `gemma3_modelscope_deploy/models/gemma-3-4b-it/` | `models/Gemma-3-4B-IT/` | 必须有等价目录 |
| 基础部署说明 | `gemma3_modelscope_deploy/README.md` | `models/Gemma-3-4B-IT/README.md` | 可选 |

重要说明：

- `CausalMM` 运行时并不依赖 `gemma3_modelscope_deploy/app.py` 这个服务。
- `CausalMM` 只需要一个本地可读取的 Gemma 3 模型目录。
- 现在默认路径写成了 `models/Gemma-3-4B-IT`，只是因为当前机器上权重放在这里。

如果你迁移后已经有别的 Gemma 3 本地目录，只要把 `--model-path` 或 `MODEL_PATH` 改掉即可，不必保留整个 `gemma3_modelscope_deploy` 目录结构。

### 4.3 原始 CausalMM 参考代码

这些文件主要用于对照原始思路，不是当前 Gemma 3 运行时直接 import 的核心依赖。

| 作用 | 相对路径 | 当前绝对路径 | 迁移时是否必须 |
|---|---|---|---|
| 原始单分支 CausalMM 解码参考 | `CausalMM/llava-1.5/causalmm_cf/causalmm_sm.py` | `code/causalmm_gemma3/causalmm_cf/causalmm_sm.py` | 可选 |
| 原始多分支 CausalMM 解码参考 | `CausalMM/llava-1.5/causalmm_cf/causalmm_mm.py` | `code/causalmm_gemma3/causalmm_cf/causalmm_mm.py` | 可选 |
| 原始项目说明 | `CausalMM/README.md` | `code/causalmm_gemma3/README.md` | 可选 |

## 5. 迁移时最少要拷哪些东西

### 5.1 只想跑 Gemma 3 + CausalMM

最小集合：

1. `CausalMM/llava-1.5/gemma3/causalmm_gemma3.py`
2. 你要用到的入口脚本，至少一个：
   - `chat_gemma3_causalmm.py`
   - `api_gemma3_causalmm.py`
   - `eval_test_raw_gemma3_causalmm.py`
   - `eval_test_raw_gemma3_attn_shuffle.py`
3. 一个本地可用的 Gemma 3 模型目录
4. Python 依赖环境

如果你只做最小迁移，目录可以长这样：

```text
your_project/
  gemma3_causalmm/
    causalmm_gemma3.py
    chat_gemma3_causalmm.py
    api_gemma3_causalmm.py
    eval_test_raw_gemma3_causalmm.py
    eval_test_raw_gemma3_attn_shuffle.py
    merge_eval_test_raw_shards.py
    run_causalmm_server.sh
    run_eval_test_raw_gemma3_attn_shuffle_4gpu.sh
  models/
    gemma-3-4b-it/
      config.json
      ...
```

但要注意：

- 上面这些脚本默认都写成 `from causalmm_gemma3 import CausalMMGemma3`
- 所以如果你改目录结构，要么把它们放在同一目录，要么自己改 import 路径。

### 5.2 还想保留当前的 Gemma 3 基础部署方式

再额外带走：

1. `gemma3_modelscope_deploy/requirements.txt`
2. `gemma3_modelscope_deploy/app.py`
3. `gemma3_modelscope_deploy/download_model.py`
4. `gemma3_modelscope_deploy/run_server.sh`
5. `gemma3_modelscope_deploy/models/gemma-3-4b-it/` 或者让它重新下载

### 5.3 不需要带走的东西

这些不是代码依赖，只是数据、结果或参考资源：

- `test_result/`
- `train2014/`
- `VG_100K/`
- `images/`
- `imgs/`
- 各种已经导出的 `.json` / `.jsonl` 结果文件

除非你也要把数据集和历史结果一起迁走，否则这些都不是运行 `Gemma 3 + CausalMM` 的必需代码。

## 6. 环境依赖

当前工程里明确写出来的依赖在：

- `gemma3_modelscope_deploy/requirements.txt`

内容是：

```text
modelscope
transformers==4.51.3
huggingface-hub<1.0
accelerate
fastapi
uvicorn[standard]
pillow
sentencepiece
protobuf>=3.20.0,<5
safetensors
```

如果只跑离线评测，`fastapi` 和 `uvicorn` 不一定必须，但建议一起装，避免漏依赖。

最重要的版本约束是：

- `transformers==4.51.3`

因为当前实现直接依赖了 Gemma 3 的内部接口：

- `transformers.models.gemma3.modeling_gemma3.eager_attention_forward`
- `transformers.cache_utils.HybridCache`
- `transformers.models.gemma3.modeling_gemma3.repeat_kv`
- `Gemma3ForConditionalGeneration.get_image_features`
- `model.vision_tower`
- `model.multi_modal_projector`

这些接口如果在新版本里改名、改签名或改行为，当前代码就可能失效。

## 7. 迁移步骤

### 7.1 建议的迁移顺序

1. 先复制代码文件。
2. 再准备 Python 环境。
3. 再放 Gemma 3 权重目录。
4. 再跑最简单的单图命令验证。
5. 最后再跑评测或多卡分片。

### 7.2 最小验证命令

进入：

```bash
cd /your/new/path/gemma3_causalmm
```

运行：

```bash
python chat_gemma3_causalmm.py \
  --model-path /your/model/path/gemma-3-4b-it \
  --image /path/to/example.jpg \
  --prompt "What is in this image? Answer briefly." \
  --cf-mode language \
  --attention-method reverse_and_normalize \
  --gamma 1.0 \
  --epsilon 0.1 \
  --max-new-tokens 64
```

如果这个命令能正常返回文本，说明最核心的迁移基本成功。

### 7.3 API 服务验证

```bash
export MODEL_PATH=/your/model/path/gemma-3-4b-it
bash run_causalmm_server.sh
```

默认端口是 `8001`。

### 7.4 普通评测

```bash
python eval_test_raw_gemma3_causalmm.py \
  --model-path /your/model/path/gemma-3-4b-it \
  --question-file /path/to/test_raw_llava.jsonl \
  --answer-file /path/to/test_raw.json \
  --image-folder /path/to/images \
  --output-file /path/to/output.json \
  --cf-mode language \
  --attention-method reverse_and_normalize \
  --gamma 1.0 \
  --epsilon 0.1 \
  --temperature 0
```

### 7.5 `attn_shuffle` 评测

```bash
python eval_test_raw_gemma3_attn_shuffle.py \
  --model-path /your/model/path/gemma-3-4b-it \
  --question-file /path/to/test_raw_llava.jsonl \
  --answer-file /path/to/test_raw.json \
  --image-folder /path/to/images \
  --output-file /path/to/output.json \
  --gamma 1.0 \
  --epsilon 0.1 \
  --temperature 0 \
  --attention-layer -1
```

### 7.6 4 卡 `attn_shuffle`

```bash
export MODEL_PATH=/your/model/path/gemma-3-4b-it
export QUESTION_FILE=/path/to/test_raw_llava.jsonl
export ANSWER_FILE=/path/to/test_raw.json
export IMAGE_FOLDER=/path/to/images
export OUT_DIR=/path/to/output_dir

bash run_eval_test_raw_gemma3_attn_shuffle_4gpu.sh
```

## 8. 迁移时最容易踩的坑

### 8.1 `transformers` 版本不一致

最危险的问题就是版本漂移。

如果你迁移后不是 `4.51.3`，最可能出问题的地方有：

1. `modeling_gemma3.eager_attention_forward` 的函数签名变了。
2. `HybridCache` 的构造参数变了。
3. `get_image_features` 的返回 shape 变了。
4. `vision_tower(..., output_attentions=True)` 的输出结构变了。

### 8.2 目录改了但 import 没改

这些脚本默认都在同一目录下直接 import：

```python
from causalmm_gemma3 import CausalMMGemma3
```

如果你把它们拆到不同目录，必须同步改 import。

### 8.3 模型路径改了但脚本默认值没改

很多脚本的默认 `--model-path` 仍然写的是当前机器路径：

```text
models/Gemma-3-4B-IT
```

迁移后如果不显式传 `--model-path`，很容易直接报路径不存在。

### 8.4 `attn_shuffle` 比通用版更吃资源

因为它会显式取：

```python
self.model.vision_tower(..., output_attentions=True)
```

所以：

- 显存更高
- 速度更慢
- 更依赖具体 GPU 配置

这不是 bug，而是实现方式决定的。

## 9. 如果你要迁到一个全新仓库，建议保留的目录结构

建议至少保留成这样：

```text
new_repo/
  causalmm_gemma3/
    causalmm_gemma3.py
    chat_gemma3_causalmm.py
    api_gemma3_causalmm.py
    eval_test_raw_gemma3_causalmm.py
    eval_test_raw_gemma3_attn_shuffle.py
    merge_eval_test_raw_shards.py
    run_causalmm_server.sh
    run_eval_test_raw_gemma3_attn_shuffle_4gpu.sh
    GEMMA3_CAUSALMM_MIGRATION.md
  models/
    gemma-3-4b-it/
      config.json
      model-00001-of-00002.safetensors
      model-00002-of-00002.safetensors
      tokenizer.json
      ...
```

如果你还想保留当前这套基础部署服务，再加：

```text
new_repo/
  gemma3_modelscope_deploy/
    app.py
    download_model.py
    run_server.sh
    requirements.txt
```

## 10. 一句话总结

当前这套适配，本质上是：

1. 用 Hugging Face Gemma 3 做底座。
2. 用 monkey patch 的方式接管 Gemma 3 decoder attention。
3. 用两路前向加 `CausalMM` 对比解码公式完成 counterfactual decoding。
4. 视觉侧根据资源开销，分别实现了 feature-level counterfactual 和 vision-attention-shuffle 两种版本。

如果你只是要迁移并复现当前结果，优先保证三件事不变：

1. `transformers==4.51.3`
2. `causalmm_gemma3.py` 及其相邻入口脚本目录关系不变
3. `--model-path` 指向一个可用的本地 `gemma-3-4b-it` 目录
