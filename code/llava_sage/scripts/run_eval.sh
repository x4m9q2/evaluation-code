cd /path/to/sage_repro_bundle

OMP_NUM_THREADS=1 PYTHONPATH=. python -m llava.eval.model_vqa_loader \
  --model-path /path/to/sage_repro_bundle/checkpoints/assembled_llava_v15_from_mmproj_gate_20260311 \
  --image-folder /root/train2014 \
  --question-file /path/to/sage_repro_bundle/tmp/test_raw_llava_smoke_8.jsonl \
  --answers-file /path/to/sage_repro_bundle/tmp/test_merged_smoke_8.jsonl \
  --conv-mode llava_v1 \
  --temperature 0 \
  --num_beams 1 \
  --max_new_tokens 128