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
data/stage2/train_raw_mixed_qwenratio_oldbase_sam3_plus_vqa_nonumbermask.json
data/stage2/patch_mask_analysis_train_raw_qwenkeep_sam3_nonumbermask_compat.npz
data/eval/test_raw_with_shortcut_answer.json
data/images/coco/train2014
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

Run inference first:

```bash
bash scripts/run_eval_test_raw.sh
```

This repository no longer ships the runnable `xVerify` evaluation code or its
weights. To reproduce the original `ACC` / `SR` protocol externally, build two
judge inputs from the merged prediction file:

- `ACC`: compare `model_pred` against `answer`
- `SR`: compare `model_pred` against `shortcut_answer`

Both comparisons should keep `question` and `answer_type`. Aggregate each
metric as judged-correct over valid rows, and optionally report the
`by_answer_type` breakdown.

The original local experiments used `xVerify-0.5B-I` as the judge model for
both metrics. Use the same judge if you need comparable numbers.

This main evaluation path does not depend on the original LLaVA package.
`scripts2/eval_vqav2_testdev.py` is a reference-only helper and still expects
`llava.eval` to be importable.

## 6. Accuracy Only On A Training Set

For a training-set file without `shortcut_answer`, run inference first and
evaluate only normal correctness against `answer`.

## 7. NaPO Notes

This repo currently keeps two separate NaPO paths:

- Gemma NaPO: top-level wrapper `scripts/run_napo_shortcut.sh`, reading
  `data/napo/train_raw_pos_neg_shortcut.json`
- LLaVA NaPO: top-level wrapper `scripts/run_napo_llava.sh`, reading
  `data/napo_llava/train_raw_pos_neg_shortcut_hf`

The LLaVA NaPO code under `third_party/napo_llava_ref/` is a local source copy
of `third_party/napo_llava_ref`, with a few small compatibility fixes documented
in `third_party/napo_llava_ref/README.md`.

The actual imported-source diff is limited to:

- `third_party/napo_llava_ref/utils/utils.py`: lazy `matplotlib` import
- `third_party/napo_llava_ref/muffin/train/trainers.py`:
  `compute_loss(..., num_items_in_batch=None)` for `transformers==4.51.3`,
  plus removal of the debug `print(data_dict.keys())`
- `third_party/napo_llava_ref/README.md`: local provenance / compatibility
  note for the imported tree

Bundle-side launch compatibility is handled in `scripts/run_napo_llava.sh`:

- repository-relative path resolution and `PYTHONPATH` setup
- local CLIP symlink bootstrap under `third_party/clip-vit-large-patch14-336`
- `--eval_strategy no` instead of upstream `--evaluation_strategy no`

Historical scripts inside `third_party/napo_llava_ref/script/train/` are kept
for traceability and may still use upstream absolute paths or
`--evaluation_strategy no`. Prefer `scripts/run_napo_llava.sh`.

Shortcut stage-2 outputs can be converted into the LLaVA NaPO HF dataset with:

```bash
RUN=1 PYTHON_BIN=$PWD/.venv_gemma/bin/python \
  bash scripts/run_build_shortcut_napo_splits.sh

RUN=1 PYTHON_BIN=$PWD/.venv_gemma/bin/python \
  bash scripts/run_build_shortcut_napo_llava_dataset.sh
```

Validated LLaVA NaPO smoke:

- `max_steps=1`
- `global_step=1`
- checkpoint written under `/tmp/napo_llava_shortcut_generated_smoke/checkpoints/checkpoint-1`

## 8. Current Example Runs

Example logs in `log_examples/`:

- `gemma3_4b_pretrain_gate_projector_l1p1_zero2_bs16_ga2_flashattn_save2500_20260429_0240.log`
- `gemma3_4b_qwenratio_sam3_gate_l1_mask_nonumber_zero1_bs8_ga4_20260428_182441.log`
- `gemma3_4b_qwenratio_sam3_gate_l1_mask_nonumber_full_bs8_20260428_145030.log`

These are useful for checking exact command lines and expected loss fields.

## 9. Loss Fields

Training logs include:

- `loss`: HuggingFace/Trainer logged loss, averaged according to Trainer logging behavior.
- `loss_ce`: model CE loss.
- `loss_gate_l1_raw`: raw gate L1 penalty.
- `loss_gate_l1_weighted`: weighted gate L1 penalty.
- `loss_mask_raw`: raw mask patch suppression loss.
- `loss_mask_weighted`: weighted mask patch suppression loss.
- `loss_total`: direct local composed loss used by the custom trainer.

For pretraining with `gate_l1_loss_weight=0.1`, `loss_gate_l1_weighted` should be roughly `0.1 * loss_gate_l1_raw`.
