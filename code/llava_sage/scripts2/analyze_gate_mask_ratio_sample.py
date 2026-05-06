#!/usr/bin/env python3
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPTokenizer

from llava.mm_utils import process_images, get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def parse_args():
    parser = argparse.ArgumentParser(description="Compare gate activation on masked vs non-masked patches.")
    parser.add_argument("--model-a", type=Path, required=True)
    parser.add_argument("--model-b", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--image-folder", type=Path, required=True)
    parser.add_argument("--patch-mask-analysis-path", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def load_patch_mask_analysis(path: Path):
    analysis = np.load(os.path.expanduser(path), allow_pickle=True)
    qids = analysis["question_ids"].astype(np.int64)
    coverage = analysis["coverage_ratio"].astype(np.float32).reshape(qids.shape[0], -1)
    return {int(qid): coverage[idx] for idx, qid in enumerate(qids.tolist())}


def sample_examples(data_path: Path, mask_map, sample_size: int, seed: int):
    rows = json.load(open(data_path))
    valid = []
    for row in rows:
        qid = int(row["question_id"])
        coverage = mask_map.get(qid)
        if coverage is None:
            continue
        mask_sum = float(np.sum(coverage))
        nonmask_sum = float(np.sum(1.0 - coverage))
        if mask_sum <= 1e-6 or nonmask_sum <= 1e-6:
            continue
        valid.append(row)
    rng = random.Random(seed)
    sampled = rng.sample(valid, min(sample_size, len(valid)))
    return sampled


def build_batches(rows, batch_size):
    for i in range(0, len(rows), batch_size):
        yield rows[i:i + batch_size]


def load_model_bundle(model_path: Path):
    model_name = get_model_name_from_path(str(model_path))
    tokenizer, model, image_processor, _ = load_pretrained_model(
        str(model_path),
        None,
        model_name,
        device_map=None,
        device="cuda",
    )
    model.eval()
    clip_tokenizer = CLIPTokenizer.from_pretrained(model.config.mm_vision_tower)
    return model, image_processor, clip_tokenizer


def prepare_clip_batch(rows, clip_tokenizer):
    clip_input_ids = []
    clip_attn_mask = []
    for row in rows:
        clip_ids = clip_tokenizer(
            row["question"],
            truncation=True,
            max_length=77,
        )
        input_ids = clip_ids["input_ids"]
        attn_mask = clip_ids["attention_mask"]
        eos_id = clip_tokenizer.eos_token_id
        if eos_id is not None and eos_id not in input_ids:
            if len(input_ids) >= 77:
                input_ids[-1] = eos_id
            else:
                input_ids.append(eos_id)
                attn_mask.append(1)
        clip_input_ids.append(torch.tensor((input_ids + [0] * 77)[:77], dtype=torch.long))
        clip_attn_mask.append(torch.tensor((attn_mask + [0] * 77)[:77], dtype=torch.long))

    return torch.stack(clip_input_ids, dim=0), torch.stack(clip_attn_mask, dim=0)


def prepare_images(rows, image_folder: Path, image_processor, model_config):
    pil_images = []
    for row in rows:
        image_name = f"COCO_train2014_{int(row['image_id']):012d}.jpg"
        pil_images.append(Image.open(image_folder / image_name).convert("RGB"))
    image_tensors = process_images(pil_images, image_processor, model_config)
    if isinstance(image_tensors, list):
        image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors


def run_model(model_path: Path, sampled_rows, image_folder: Path, mask_map, batch_size: int):
    disable_torch_init()
    model, image_processor, clip_tokenizer = load_model_bundle(model_path)
    out = {}
    with torch.inference_mode():
        for batch_rows in build_batches(sampled_rows, batch_size):
            image_tensors = prepare_images(batch_rows, image_folder, image_processor, model.config)
            clip_input_ids, clip_attn_mask = prepare_clip_batch(batch_rows, clip_tokenizer)
            patch_cov = torch.stack(
                [torch.tensor(mask_map[int(row["question_id"])], dtype=torch.float32) for row in batch_rows],
                dim=0,
            )
            model.encode_images(
                image_tensors.to(dtype=torch.float16, device="cuda", non_blocking=True),
                clip_input_ids.to(device="cuda", non_blocking=True),
                clip_attn_mask.to(device="cuda", non_blocking=True),
                patch_mask_coverage=patch_cov.to(device="cuda", non_blocking=True),
            )
            gate_module = getattr(model.get_model(), "gate", None)
            gate_patch = getattr(gate_module, "current_gate_patch_activation", None)
            if gate_patch is None:
                raise RuntimeError(f"No gate patch activation found for model {model_path}")
            aligned_cov = model._align_patch_mask_coverage(patch_cov, gate_patch.shape[1], gate_patch.device)
            masked_sum = aligned_cov.sum(dim=1).clamp_min(1e-6)
            nonmasked_cov = (1.0 - aligned_cov).clamp_min(0.0)
            nonmasked_sum = nonmasked_cov.sum(dim=1).clamp_min(1e-6)
            masked_mean = (gate_patch.float() * aligned_cov).sum(dim=1) / masked_sum
            nonmasked_mean = (gate_patch.float() * nonmasked_cov).sum(dim=1) / nonmasked_sum
            ratio = masked_mean / nonmasked_mean.clamp_min(1e-6)
            for idx, row in enumerate(batch_rows):
                qid = int(row["question_id"])
                out[qid] = {
                    "masked_gate_mean": float(masked_mean[idx].item()),
                    "nonmasked_gate_mean": float(nonmasked_mean[idx].item()),
                    "mask_to_nonmask_ratio": float(ratio[idx].item()),
                    "mask_patch_weight_sum": float(masked_sum[idx].item()),
                    "nonmask_patch_weight_sum": float(nonmasked_sum[idx].item()),
                }
    del model
    torch.cuda.empty_cache()
    return out


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("PYTHONPATH", "/path/to/sage_repro_bundle")
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

    mask_map = load_patch_mask_analysis(args.patch_mask_analysis_path)
    sampled_rows = sample_examples(args.data_path, mask_map, args.sample_size, args.seed)
    a_results = run_model(args.model_a, sampled_rows, args.image_folder, mask_map, args.batch_size)
    b_results = run_model(args.model_b, sampled_rows, args.image_folder, mask_map, args.batch_size)

    detailed = []
    for row in sampled_rows:
        qid = int(row["question_id"])
        a = a_results[qid]
        b = b_results[qid]
        detailed.append({
            "question_id": qid,
            "image_id": int(row["image_id"]),
            "answer_type": row.get("answer_type"),
            "question": row["question"],
            "answer": row.get("answer"),
            "mask_ratio_weighted": float(np.mean(mask_map[qid])),
            "model_a_masked_gate_mean": a["masked_gate_mean"],
            "model_a_nonmasked_gate_mean": a["nonmasked_gate_mean"],
            "model_a_mask_to_nonmask_ratio": a["mask_to_nonmask_ratio"],
            "model_b_masked_gate_mean": b["masked_gate_mean"],
            "model_b_nonmasked_gate_mean": b["nonmasked_gate_mean"],
            "model_b_mask_to_nonmask_ratio": b["mask_to_nonmask_ratio"],
            "ratio_diff_b_minus_a": b["mask_to_nonmask_ratio"] - a["mask_to_nonmask_ratio"],
            "masked_mean_diff_b_minus_a": b["masked_gate_mean"] - a["masked_gate_mean"],
            "nonmasked_mean_diff_b_minus_a": b["nonmasked_gate_mean"] - a["nonmasked_gate_mean"],
        })

    a_ratios = [x["model_a_mask_to_nonmask_ratio"] for x in detailed]
    b_ratios = [x["model_b_mask_to_nonmask_ratio"] for x in detailed]
    diff_ratios = [x["ratio_diff_b_minus_a"] for x in detailed]
    summary = {
        "sample_size": len(detailed),
        "seed": args.seed,
        "model_a": str(args.model_a),
        "model_b": str(args.model_b),
        "model_a_ratio_stats": summarize(a_ratios),
        "model_b_ratio_stats": summarize(b_ratios),
        "ratio_diff_b_minus_a_stats": summarize(diff_ratios),
        "paired_counts": {
            "b_ratio_gt_a_ratio": int(sum(x > 0 for x in diff_ratios)),
            "b_ratio_lt_a_ratio": int(sum(x < 0 for x in diff_ratios)),
            "b_ratio_eq_a_ratio": int(sum(abs(x) < 1e-12 for x in diff_ratios)),
        },
        "samples": detailed,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    import csv
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(detailed[0].keys()))
        writer.writeheader()
        writer.writerows(detailed)

    print(json.dumps({
        "output_json": str(args.output_json),
        "output_csv": str(args.output_csv),
        "sample_size": len(detailed),
        "model_a_ratio_mean": summary["model_a_ratio_stats"]["mean"],
        "model_b_ratio_mean": summary["model_b_ratio_stats"]["mean"],
        "ratio_diff_mean": summary["ratio_diff_b_minus_a_stats"]["mean"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
