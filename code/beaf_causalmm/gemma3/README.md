# Gemma 3 4B CausalMM

This directory adapts CausalMM decoding to Gemma 3 4B (`LLM-Research/gemma-3-4b-it` from ModelScope).

The default path uses language-side counterfactual attention, matching the CausalMM README section "LLM Counterfactual Attention". It keeps the base Gemma 3 weights unchanged and runs two decoding paths:

```text
causalmm_logits = (1 + gamma) * logits - gamma * cf_logits
```

For Gemma 3's 896px SigLIP vision encoder, dumping all vision attention maps is very memory heavy. The `vision` mode therefore applies feature-space counterfactuals to Gemma 3 image soft tokens instead of materializing full vision attentions.

## One Image

```bash
cd code/beaf_causalmm/gemma3
python chat_gemma3_causalmm.py \
  --model-path models/Gemma-3-4B-IT \
  --image data/playground_data/vg/VG_100K/1.jpg \
  --prompt "What is in this image? Answer briefly." \
  --cf-mode language \
  --attention-method reverse_and_normalize \
  --gamma 1.0 \
  --epsilon 0.1 \
  --max-new-tokens 64
```

## POPE-Style Evaluation

```bash
python eval_pope_gemma3_causalmm.py \
  --model-path models/Gemma-3-4B-IT \
  --question-file /path/to/questions.jsonl \
  --image-folder /path/to/images \
  --answers-file code/beaf_causalmm/experiments/output/gemma3_causalmm_answers.jsonl \
  --cf-mode language \
  --attention-method reverse_and_normalize \
  --gamma 1.0 \
  --epsilon 0.1
```

## Counterfactual Modes

- `--cf-mode language`: edit Gemma 3 decoder attention during the counterfactual forward pass.
- `--cf-mode vision`: edit image soft-token features before they enter the decoder.
- `--cf-mode both`: edit both image features and decoder attention.

Language attention methods:

- `reverse_and_normalize`
- `reverse`
- `random`
- `uniform`
- `shuffle`
- `none`

Vision feature methods:

- `shuffle`
- `uniform`
- `reverse`
- `random`
- `none`

## API Server

```bash
cd code/beaf_causalmm/gemma3
bash run_causalmm_server.sh
```

The default port is `8001`.

```bash
curl http://127.0.0.1:8001/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "What is in this image? Answer briefly.",
    "image_path": "data/playground_data/vg/VG_100K/1.jpg",
    "max_new_tokens": 64,
    "cf_mode": "language",
    "attention_method": "reverse_and_normalize",
    "gamma": 1.0,
    "epsilon": 0.1
  }'
```
