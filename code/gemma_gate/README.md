# Gemma3 Gate + Mask-Loss Experiment Package

This package contains the code, launch scripts, and evaluation utilities needed to reproduce the Gemma3 gate experiments.

Large assets are intentionally not included:

- Gemma3 model weights
- SigLIP text encoder weights
- xVerify model weights
- COCO images
- training/evaluation JSON datasets
- patch-mask `.npz` files
- generated checkpoints

See `docs/TRAINING_AND_EVAL.md` for required external paths and commands.

## Contents

- `gemma/`: modified Gemma3 training code.
- `gemma/src/gate_model/`: self-contained `DualInputGate` implementation used by the Gemma scripts.
- `scripts/`: DeepSpeed configs.
- `scripts2/`: Gemma inference and accuracy / shortcut-rate evaluation helpers.
- `x_verify/`: lightweight xVerify runner code, without model weights or old outputs.
- `run_scripts/`: reproducible launch scripts for pretraining, mask-loss finetuning, and evaluation.
- `log_examples/`: example logs from the current experiments.

## Key Logic

- Gate is inserted before Gemma3 `multi_modal_projector`.
- Gate text feature uses a frozen SigLIP text encoder.
- Pretraining trains only gate + multimodal projector.
- Finetuning can train LLM + projector + gate with L1 loss and mask patch loss.
- `--disable_number_mask_loss True` disables mask loss for `answer_type == "number"`.
- `scripts2/eval_test_raw_gemma3.py` can run Gemma inference and call xVerify to compute accuracy and shortcut rate.
