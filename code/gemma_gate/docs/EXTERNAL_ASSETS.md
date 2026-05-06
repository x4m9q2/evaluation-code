# External Assets

This package does not include model weights, image file contents, checkpoints,
or generated training outputs.

The package is self-contained for the Gemma gate implementation: it includes
`gemma/src/gate_model/build_gate_model.py`. You do not need the original
`llava.model.gate_model` module unless you deliberately want to compare against
the source workspace.

## Model Weights

Download the three model directories with:

```bash
pip install modelscope
python run_scripts/download_models_modelscope.py
```

Default output paths:

```bash
models/Gemma-3-4B-IT
models/siglip-so400m-patch14-384
code/gemma_gate/x_verify/xVerify-0.5B-I
```

The local experiment used ModelScope mirrors. The original Hugging Face model
name for Gemma3 is `google/gemma-3-4b-it`.

## Data Paths Expected By The Run Scripts

```bash
data/playground_data/coco/train2014
data/playground_data/coco/train2017
data/playground_data/gqa/images
data/playground_data/vg/VG_100K
data/playground_data/vg/VG_100K_2
data/pretrain_llava_v1_5_mix665k_single_noocr_max200_imageonly_strict_noocr.json
data/stage2/train_raw_mixed_qwenratio_oldbase_sam3_plus_vqa.json
data/stage2/patch_mask_analysis_train_raw_qwenkeep_sam3_compat.npz
data/eval/test_raw_with_shortcut_answer.json
data/stage2/vqa_train2014.json
```

The image directories are placeholders in this trimmed bundle; place the actual
image files under the same relative paths.
