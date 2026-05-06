# Manifest

Bundle root: the extracted archive directory.

Included code:

- `code/gemma_gate/`: Gemma 3 gate training/evaluation code bundle, copied from `debug/gemma_gate_mask_code_only_20260429_from_bypy_inspect/gemma_code_bundle_20260429_v2`.
- `code/beaf_causalmm/`: Gemma 3 CausalMM/BEAF-style comparison code, copied from `debug/gemma3_causalmm_bundle_clean_20260501_inspect/gemma3_causalmm_bundle_clean`.
- `code/napo_gemma_debug/`: Gemma NaPO debug/probe scripts, copied from `debug/napo_smoke_20260501` with model/checkpoint outputs excluded.
- `code/llava_sage/`: LLaVA-v1.5 SAGE source tree, including model gate code, `train_xformers.py`, mask-supervised trainer changes, pretraining/finetuning scripts, inference scripts, and metric utilities.
- `third_party/napo_llava_ref/`: LLaVA NaPO comparison reference snapshot from `NaPO-master`, excluding large preference datasets, checkpoints, and evaluation outputs.
- `code/data_tools/`: mask generation, mask analysis, shortcut mining/filtering, and SAGE-AS package construction utilities.
- `code/data_tools/build_llava_pretrain_json.py` and `code/data_tools/llava_v1_5_strict_noocr_drop_hashes.txt`: merged build script and strict no-OCR drop-list used to build `data/pretrain_llava_v1_5_mix665k_single_noocr_max200_imageonly_strict_noocr.json` from raw `llava_v1_5_mix665k.json`.
- `code/data_tools/assemble_llava_base.py`: assemble a full LLaVA checkpoint from the LLaVA-v1.5 base model plus a projector/gate adapter (`mm_projector.bin`).
- `code/evaluation/pope_beaf_gate/`: POPE evaluation and gate activation visualization code.
- `code/evaluation/causalmm_llava/`: LLaVA CausalMM/BEAF-style comparison implementation copied from `CausalMM-main`.
- `code/evaluation/x_verify/`: xVerify source/wrapper files for Acc/SR evaluation, excluding model weights and generated outputs.
- `code/evaluation/shortcut_metrics_scripts/`: compact shortcut metric scripts from `eval_accuracy_shortcut_bundle_20260402`.

Included data:

- `data/pretrain_llava_v1_5_mix665k_single_noocr_max200_imageonly_strict_noocr.json`: pretraining JSON.
- `data/stage2/train_raw_mixed_qwenratio_oldbase_sam3_plus_vqa.json`: stage-2 mixed SFT JSON.
- `data/stage2/patch_mask_analysis_train_raw_qwenkeep_sam3_compat.npz`: SAM mask patch supervision.
- `data/stage2/train_raw.json`, `data/stage2/val_raw.json`, `data/stage2/vqa_train2014.json`: raw/reference datasets.
- `data/eval/test_raw.json`, `data/eval/test_raw_with_shortcut_answer.json`: evaluation data.
- `data/napo/train_raw_pos_neg_shortcut.json`: NaPO shortcut positive/negative data.
- `data/playground_data/coco/train2014`: empty placeholder for train2014 images used by stage-2, eval, and NaPO.
- `data/playground_data/coco/train2017`: empty placeholder for train2017 images referenced by the pretrain JSON.
- `data/playground_data/gqa/images`: empty placeholder for GQA images referenced by the pretrain JSON.
- `data/playground_data/vg/VG_100K` and `data/playground_data/vg/VG_100K_2`: empty placeholders for VG images referenced by the pretrain JSON.
- `models/Gemma-3-4B-IT`: empty placeholder for Gemma 3 weights.
- `models/siglip-so400m-patch14-384`: empty placeholder for SigLIP text encoder weights.
- `data/sage_as/`: released SAGE-AS dataset README/manifest/Croissant metadata copy.
- `models/llava-v1.5-7b`: empty placeholder for LLaVA-v1.5-7B base weights.
- `models/clip-vit-large-patch14-336`: empty placeholder for the CLIP vision tower.
- `models/xVerify-0.5B-I`: empty placeholder for xVerify weights.
- `models/sam3_ckpt/sam3.pt`: expected SAM3 checkpoint path.
- `data/images/`, `data/pope/`, `data/beaf/`, `data/napo_llava/`, `data/llava_stage1/`, `data/llava_stage2/`, `data/masks/`: relative-path placeholders used by the top-level LLaVA/SAGE wrappers.
  `data/beaf/` is intentionally placeholder-only and does not include official BEAF benchmark files.

Top-level LLaVA/SAGE scripts:

- `scripts/common_llava.sh`: shared relative-path environment setup.
- `scripts/run_build_pretrain_json.sh`: regenerate the filtered LLaVA pretraining JSON from the raw LLaVA mix file.
- `scripts/run_llava_pretrain_gate.sh`: LLaVA gate/projector pretraining.
- `scripts/run_assemble_llava_checkpoint.sh`: inject a saved projector/gate adapter into the LLaVA base model and save a directly loadable checkpoint.
- `scripts/run_llava_pretrain_nogate.sh`: LLaVA projector pretraining with the dual-input gate forced off and gate L1 loss set to zero.
- `scripts/run_llava_stage2_mask_sft.sh`: mask-supervised stage-2 SFT.
- `scripts/run_llava_eval_acc_sr.sh`: inference plus xVerify Acc/SR.
- `scripts/run_pope_eval.sh`: POPE evaluation.
- `scripts/run_beaf_eval.sh`: BEAF evaluation.
- `scripts/run_napo_llava.sh`: LLaVA NaPO comparison training wrapper targeting the third-party reference snapshot.
- `scripts/run_mask_generation_and_filtering.sh`: mask-generation/filtering entrypoint and checks.
- `scripts/run_all_llava_sage_pipeline.sh`: dry-run pipeline summary.

Excluded:

- model weights and training checkpoints.
- image file contents; only empty placeholder directories are included.
- local VG image symlinks from `code/data_tools/gqa_vg_sam3_masks_sampled10000_20260430/vg_images/`; use `data/images/vg/VG_100K` and `data/images/vg/VG_100K_2` instead.
- raw `data/llava_stage1/llava_v1_5_mix665k.json` and intermediate `data/llava_stage1/llava_v1_5_mix665k_single_noocr_max200_imageonly.json`; these are used to verify the pretraining JSON build script locally but are excluded from the compressed release archive to avoid duplicating large source data.
- full original NaPO archive metadata beyond the included trimmed reference snapshot; use the official NaPO release or your own archived source if needed for direct source comparison.
- Python virtual environments.
- wandb directories.
- `__pycache__`, `.pyc`, and generated model output shards.
- large SAGE/LLaVA/Gemma checkpoints, optimizer states, and base model weights.
- xVerify model weights.
- SAM3 model weights.
- raw COCO/GQA/VG/BEAF/POPE image files.
- official BEAF benchmark JSON, images, and upstream metric/example-answer files.
- generated inference outputs and evaluation caches.

All canonical scripts derive paths from the bundle location and expect assets in the relative directories listed above.
