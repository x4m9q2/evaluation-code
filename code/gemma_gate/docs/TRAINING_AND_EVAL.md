# Training And Evaluation Instructions

Assume commands are run from the bundle root or through the top-level scripts.

## 1. External Assets

Prepare these relative paths:

```bash
models/Gemma-3-4B-IT
models/siglip-so400m-patch14-384
code/gemma_gate/x_verify/xVerify-0.5B-I
data/pretrain_llava_v1_5_mix665k_single_noocr_max200_imageonly_strict_noocr.json
data/playground_data
data/stage2/train_raw_mixed_qwenratio_oldbase_sam3_plus_vqa.json
data/stage2/patch_mask_analysis_train_raw_qwenkeep_sam3_compat.npz
data/eval/test_raw_with_shortcut_answer.json
data/playground_data/coco/train2014
```

Model weights can be downloaded from ModelScope with:

```bash
pip install modelscope
python run_scripts/download_models_modelscope.py
```

See `docs/EXTERNAL_ASSETS.md` for the full external-asset checklist.

Optional VQAv2 training accuracy path:

```bash
data/stage2/vqa_train2014.json
```

## 2. Environment

The working environment used:

```bash
Python 3.10
torch 2.6.0+cu124
CUDA runtime 12.4 through PyTorch
system nvcc 12.1
flash-attn 2.8.3
deepspeed
transformers
trl
```

Install from the packaged requirements first:

```bash
cd code/gemma_gate/gemma
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-py310.txt
pip install flash-attn --no-build-isolation
```

If FlashAttention2 is unstable, set `DISABLE_FLASH_ATTN2=True` in the run scripts. Gemma3 emits a warning recommending eager attention, but the local 5-step smoke test completed successfully with FlashAttention2.

## 3. Pretraining: Gate + Projector + L1

This stage matches the pretrain-style setup:

- `freeze_llm=True`
- `freeze_vision_tower=True`
- `freeze_projector=False`
- `use_dual_input_gate=True`
- `freeze_gate_text_encoder=True`
- `gate_l1_loss_weight=0.1`
- `mask_patch_loss_weight=0.0`
- ZeRO-2, `per_device_train_batch_size=16`, `gradient_accumulation_steps=2`
- save every 2500 steps, no save-total limit

Run:

```bash
bash run_scripts/run_gemma_pretrain_gate_projector_l1.sh
```

Important output variables can be overridden:

```bash
OUTPUT_DIR=checkpoints/my_gemma_pretrain \
LOG_FILE=logs/my_gemma_pretrain.log \
bash run_scripts/run_gemma_pretrain_gate_projector_l1.sh
```

## 4. Finetuning: Gate + L1 + Mask Loss Without Number Mask Loss

This stage matches the later mask-loss experiment:

- `freeze_llm=False`
- `freeze_projector=False`
- default script keeps `freeze_vision_tower=True`
- `use_dual_input_gate=True`
- `gate_l1_loss_weight=0.01`
- `mask_patch_loss_weight=0.125`
- `disable_number_mask_loss=True`
- ZeRO-1, `per_device_train_batch_size=8`, `gradient_accumulation_steps=4`
- 3 epochs

Run:

```bash
bash run_scripts/run_gemma_finetune_gate_l1_mask_nonumber.sh
```

If you want the earlier fully-unfrozen vision-tower attempt, set:

```bash
FREEZE_VISION_TOWER=False
DEEPSPEED_CONFIG=code/gemma_gate/gemma/scripts/zero3.json
PER_DEVICE_TRAIN_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=1
bash run_scripts/run_gemma_finetune_gate_l1_mask_nonumber.sh
```

## 5. Accuracy And Shortcut Rate On `test_raw`

`test_raw_with_shortcut_answer.json` must contain at least:

- `question_id`
- `image_id`
- `question`
- `answer`
- `shortcut_answer`
- `answer_type`

Run inference and xVerify:

```bash
MODEL_PATH=checkpoints/your_model \
bash run_scripts/run_gemma_eval_test_raw_xverify.sh
```

The metrics file is:

```bash
infer_result/<model_name>/test_raw_with_shortcut_answer.xverify_metrics.json
```

It contains:

- overall accuracy
- overall shortcut rate
- `by_answer_type` breakdown for accuracy
- `by_answer_type` breakdown for shortcut rate

This main evaluation path does not depend on the original LLaVA package.
`scripts2/eval_vqav2_testdev.py` is a reference-only helper and still expects
`llava.eval` to be importable.

## 6. Accuracy Only On A Training Set

For a training-set file without `shortcut_answer`, run Gemma inference first and then xVerify only against `answer`. The packaged `eval_shortcut_metrics.py` expects both `answer` and `shortcut_answer`, so for accuracy-only use either:

- add a temporary `shortcut_answer` equal to an empty string and ignore SR, or
- adapt `scripts2/eval_shortcut_metrics.py` to call only the accuracy branch.

## 7. Current Example Runs

Example logs in `log_examples/`:

- `gemma3_4b_pretrain_gate_projector_l1p1_zero2_bs16_ga2_flashattn_save2500_20260429_0240.log`
- `gemma3_4b_qwenratio_sam3_gate_l1_mask_nonumber_zero1_bs8_ga4_20260428_182441.log`
- `gemma3_4b_qwenratio_sam3_gate_l1_mask_nonumber_full_bs8_20260428_145030.log`

These are useful for checking exact command lines and expected loss fields.

## 8. Loss Fields

Training logs include:

- `loss`: HuggingFace/Trainer logged loss, averaged according to Trainer logging behavior.
- `loss_ce`: model CE loss.
- `loss_gate_l1_raw`: raw gate L1 penalty.
- `loss_gate_l1_weighted`: weighted gate L1 penalty.
- `loss_mask_raw`: raw mask patch suppression loss.
- `loss_mask_weighted`: weighted mask patch suppression loss.
- `loss_total`: direct local composed loss used by the custom trainer.

For pretraining with `gate_l1_loss_weight=0.1`, `loss_gate_l1_weighted` should be roughly `0.1 * loss_gate_l1_raw`.
