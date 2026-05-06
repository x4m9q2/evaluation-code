# Qwen3.5-9B Local Deployment

This directory contains a minimal local deployment flow for `Qwen/Qwen3.5-9B`:

1. Create an isolated runtime environment:
   `bash scripts/qwen35/setup_qwen35_vllm_env.sh`
2. Download the model from ModelScope:
   `bash scripts/qwen35/download_qwen35_9b_modelscope.sh`
3. Start the OpenAI-compatible server:
   `bash scripts/qwen35/serve_qwen35_9b_vllm.sh`
4. Verify the endpoint:
   `/root/venv/qwen35_vllm/bin/python scripts/qwen35/smoke_test_qwen35_9b.py`

Useful environment variables:

- `VENV_DIR`: location of the isolated environment
- `MODEL_DIR`: local path that holds the downloaded model
- `CUDA_VISIBLE_DEVICES`: GPU selection for the server
- `MAX_MODEL_LEN`: context length passed to vLLM; default is `32768` for a safer single-GPU start
- `PORT`: server port, default `8000`
