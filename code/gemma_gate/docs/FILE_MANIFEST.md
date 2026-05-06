# File Manifest

## Included

- `gemma/src/train/train_sft.py`: Gemma3 SFT entrypoint with gate setup.
- `gemma/src/train/monkey_patch_forward.py`: forward monkey patch; gate is before visual projector.
- `gemma/src/gate_model/build_gate_model.py`: self-contained DualInputGate implementation copied from the LLaVA workspace. It exposes `current_gate_l1_loss` and `current_gate_patch_activation`, which Gemma training depends on.
- `gemma/src/trainer/sft_trainer.py`: CE + gate L1 + mask patch loss composition and logging.
- `gemma/src/dataset/sft_dataset.py`: SFT dataset loader with patch mask coverage and `disable_number_mask_loss`.
- `gemma/src/params.py`: extra training/data arguments.
- `scripts/zero1_bf16.json`, `scripts/zero2_bf16.json`, `scripts/zero2.json`, `scripts/zero3.json`: DeepSpeed configs.
- `scripts2/eval_test_raw_gemma3.py`: Gemma3 inference on `test_raw_with_shortcut_answer.json`.
- `scripts2/eval_vqav2_testdev.py`: VQAv2-style evaluation helper kept for reference. It still imports `llava.eval` and therefore requires a LLaVA checkout if used; it is not needed for the main `test_raw` + xVerify evaluation.
- `run_scripts/`: launch scripts.
- `run_scripts/download_models_modelscope.py`: downloads Gemma3 and SigLIP weights from ModelScope into the default local paths.
- `docs/EXTERNAL_ASSETS.md`: lists required external weights and datasets that are not included in this code-only package.

## Not Included

- `gemma/.venv`
- `gemma/.git`
- Gemma3 model weights
- SigLIP model weights
- checkpoints
- COCO image folders
- `.npz` mask files
- old inference outputs
