cd /path/to/local_scratch/LLaVA_except_playground_llava_clip_20260309

OMP_NUM_THREADS=1 PYTHONPATH=. python -m llava.eval.model_vqa_loader \
  --model-path /path/to/local_scratch/LLaVA/checkpoints/llava-grpo-debug-ori-only-checkpoint-3 \
  --image-folder /path/to/local_scratch/sam3/train2014 \
  --question-file /path/to/local_scratch/LLaVA_except_playground_llava_clip_20260309/tmp/test_raw_llava_smoke_8.jsonl \
  --answers-file /path/to/local_scratch/LLaVA/tmp/test_merged_smoke.jsonl \
  --conv-mode llava_v1 \
  --temperature 1.0 \
  --top_p 0.95 \
  --num_beams 1 \
  --max_new_tokens 8