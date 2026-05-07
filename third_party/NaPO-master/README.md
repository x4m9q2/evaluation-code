## NaPO LLaVA Local Source Copy

This directory is a local source copy of the public NaPO LLaVA code:

- upstream repository: `https://github.com/zhangzef/NaPO`

It is kept under `third_party/` so the bundle can run LLaVA NaPO without
depending on files outside this repository.

### Scope

This directory is intended as a third-party comparison baseline, not as part of
the core SAGE implementation.

The canonical bundle entrypoint is:

```bash
bash scripts/run_napo_llava.sh
```

Prefer that wrapper over upstream scripts inside `third_party/NaPO-master/`
unless you are debugging the imported NaPO code itself.

### Local Compatibility Changes

The imported code was kept as close as possible to the original source, with
only a few small compatibility patches required for the validated local
environment. The imported source tree differs from the public upstream code only
at the files listed below:

1. `utils/utils.py`
   - removed the top-level `import matplotlib.pyplot as plt`
   - moved the `matplotlib` import into `plot_images()`
   - reason: DPO training does not use plotting, so this avoids import-time
     failure when `matplotlib` is absent.
2. `muffin/train/trainers.py`
   - updated `LLaVA15DPOTrainer.compute_loss()` to accept
     `num_items_in_batch=None`, matching `transformers==4.51.3`.
   - removed `print(data_dict.keys())` from the DPO loss path
   - reason: `transformers==4.51.3` passes `num_items_in_batch`; the removed
     print was only debug noise.
3. `README.md`
   - added this bundle-local provenance / compatibility note.

Outside this imported source tree, the bundle wrapper
`scripts/run_napo_llava.sh` also differs from the upstream launch scripts:

- uses repository-relative paths and exports `PYTHONPATH` to this imported tree
- creates `third_party/clip-vit-large-patch14-336` as a local symlink to the
  configured CLIP tower when needed
- uses `--eval_strategy no` instead of upstream
  `--evaluation_strategy no`, also for `transformers==4.51.3`.

Historical scripts under `script/train/` are intentionally kept close to the
upstream layout for traceability. They may still contain upstream-style defaults
and may still use `--evaluation_strategy no`. For normal bundle runs, use the
top-level wrapper:

```bash
bash scripts/run_napo_llava.sh
```

No NaPO loss logic, DPO target construction, or shortcut preference semantics
were changed.

### Expected Data Path In This Bundle

The top-level wrapper defaults to:

- HF dataset dir: `data/napo_llava/train_raw_pos_neg_shortcut_hf`

Build it from the shortcut stage-2 outputs with:

```bash
PYTHON_BIN=$PWD/.venv_gemma/bin/python \
  bash scripts/run_build_shortcut_napo_splits.sh

PYTHON_BIN=$PWD/.venv_gemma/bin/python \
  bash scripts/run_build_shortcut_napo_llava_dataset.sh
```

The resulting LLaVA NaPO preference records use:

- `generated_question` as `question`
- `generated_answer` as `chosen`
- `original_answer` as `rejected`

### Validated Smoke Run

Validated in the bundle environment with:

```bash
DRY_RUN=0 PYTHON_BIN=$PWD/.venv_gemma/bin/python CUDA_VISIBLE_DEVICES=0,1,2,3 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 PER_DEVICE_EVAL_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=1 DATALOADER_NUM_WORKERS=0 LOGGING_STEPS=1 \
NUM_EPOCHS=1 OUTPUT_DIR=outputs/napo_llava_shortcut_generated_smoke/checkpoints \
LOGGING_DIR=outputs/napo_llava_shortcut_generated_smoke/log \
EXTRA_ARGS='--max_steps 1' bash scripts/run_napo_llava.sh
```

Observed result:

- training reached `global_step=1`
- checkpoint written to `outputs/napo_llava_shortcut_generated_smoke/checkpoints/checkpoint-1`
