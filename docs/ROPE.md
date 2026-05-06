# RoPE Notes

This bundle does not add a custom RoPE implementation for Gemma 3. RoPE is handled by Hugging Face `transformers` inside `Gemma3ForConditionalGeneration`.

Relevant local code:

- `code/gemma_gate/gemma/src/train/train_sft.py`: passes `--attn_implementation` to `from_pretrained`; optional Liger config enables `rope=True` only when `--use_liger True`.
- `code/gemma_gate/gemma/src/train/train_dpo.py`: same attention backend path for NaPO/DPO.
- `code/gemma_gate/gemma/src/train/monkey_patch_forward.py`: preserves `position_ids` and `cache_position` when patching Gemma 3 forward; it does not replace rotary embedding math.
- `code/beaf_causalmm/gemma3/causalmm_gemma3.py`: manually supplies `cache_position` during counterfactual decoding so generated-token positions stay aligned.

Operationally, use `ATTN_IMPLEMENTATION=sdpa` for the current stable path. `flash_attention_2` can be tested on a compatible CUDA/PyTorch/flash-attn stack, but it is not required by the included scripts.

