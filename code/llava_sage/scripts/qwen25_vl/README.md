# Qwen2.5-VL-7B-Instruct 本地部署

这套脚本会把 `Qwen/Qwen2.5-VL-7B-Instruct` 下载到固定目录，并用本机已有的 `vLLM` 环境启动一个本地多模态服务。

1. 下载模型到 `/root/models/Qwen2.5-VL-7B-Instruct`
   `bash scripts/qwen25_vl/download_qwen25_vl_7b_modelscope.sh`
2. 启动本地服务
   `CUDA_VISIBLE_DEVICES=0 bash scripts/qwen25_vl/serve_qwen25_vl_7b_vllm.sh`
3. 做一次多模态烟测
   `/root/venv/llava_old/bin/python scripts/qwen25_vl/smoke_test_qwen25_vl_7b.py`

默认约定：

- ModelScope 下载环境：`/root/venv/unifolm`
- vLLM 服务环境：`/root/venv/llava_old`
- 模型目录：`/root/models/Qwen2.5-VL-7B-Instruct`
- 服务端口：`8001`

可调环境变量：

- `VENV_DIR`
- `MODEL_DIR`
- `CUDA_VISIBLE_DEVICES`
- `PORT`
- `MAX_MODEL_LEN`
- `GPU_MEMORY_UTILIZATION`
- `MAX_NUM_SEQS`
