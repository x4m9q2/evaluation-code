#!/usr/bin/env python3
"""
Generate one merged binary mask per question by running SAM3 on all visual cues
linked from a question->rule mapping.

This script is designed for large-scale offline preprocessing:
- input questions come from a JSONL file
- visual cues come from merged_output_rule_mapping.json
- only the final merged binary mask is saved for each question
- shard arguments allow multi-GPU parallel runs
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.eval.postprocessors import PostProcessImage
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    Image as SAMImage,
    InferenceMetadata,
)
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    NormalizeAPI,
    RandomResizeAPI,
    ToTensorAPI,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate merged binary masks with SAM3.")
    parser.add_argument("--qa-jsonl", type=Path, required=True)
    parser.add_argument("--mapping-json", type=Path, required=True)
    parser.add_argument(
        "--image-root",
        dest="image_roots",
        type=Path,
        action="append",
        required=True,
        help="Image root that contains the row['image'] filename. Repeat to search multiple roots.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--checkpoint-path", type=str, default="sam3_ckpt/sam3.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-load-from-hf", action="store_true")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def create_transform(resolution: int) -> ComposeAPI:
    return ComposeAPI(
        transforms=[
            RandomResizeAPI(
                sizes=resolution,
                max_size=resolution,
                square=True,
                consistent_transform=False,
            ),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def create_datapoint(
    image: Image.Image, prompt: str, query_id: int, image_height: int, image_width: int
) -> Datapoint:
    datapoint = Datapoint(
        find_queries=[],
        images=[SAMImage(data=image, objects=[], size=[image_height, image_width])],
    )
    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=prompt,
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=query_id,
                original_image_id=query_id,
                original_category_id=1,
                original_size=[image_height, image_width],
                object_id=0,
                frame_index=0,
            ),
        )
    )
    return datapoint


def chunk(items: List[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def tensor_mask_to_2d(mask_tensor: torch.Tensor, h: int, w: int) -> np.ndarray:
    masks = mask_tensor.detach().cpu()
    if masks.numel() == 0:
        return np.zeros((h, w), dtype=np.uint8)

    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 3:
        merged = masks.any(dim=0).to(torch.uint8).numpy() * 255
    elif masks.ndim == 2:
        merged = masks.to(torch.uint8).numpy() * 255
    else:
        raise ValueError(f"Unexpected mask shape: {tuple(mask_tensor.shape)}")

    if merged.shape != (h, w):
        merged = np.array(
            Image.fromarray(merged).resize((w, h), resample=Image.NEAREST), dtype=np.uint8
        )
    return merged


def result_masks_to_2d(result_masks: Any, h: int, w: int) -> np.ndarray:
    if result_masks is None:
        return np.zeros((h, w), dtype=np.uint8)
    if isinstance(result_masks, torch.Tensor):
        return tensor_mask_to_2d(result_masks, h, w)
    if isinstance(result_masks, list):
        merged = np.zeros((h, w), dtype=np.uint8)
        for item in result_masks:
            if isinstance(item, torch.Tensor):
                merged = np.maximum(merged, tensor_mask_to_2d(item, h, w))
        return merged
    return np.zeros((h, w), dtype=np.uint8)


def normalize_prompt(cue: str) -> str:
    parts = [x.strip() for x in str(cue).split(",") if x.strip()]
    return parts[0] if parts else str(cue).strip()


def resolve_image_path(image_name: str, image_roots: List[Path]) -> Path:
    image_path = Path(image_name)
    if image_path.is_absolute() and image_path.exists():
        return image_path

    checked: List[str] = []
    for root in image_roots:
        candidate = root / image_name
        checked.append(str(candidate))
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not resolve image {image_name!r}. Checked: {checked[:8]}"
    )


def load_mapping(path: Path) -> Dict[int, Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["results"] if isinstance(data, dict) and "results" in data else data
    by_qid: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        qid = int(row["question_id"])
        if qid not in by_qid:
            by_qid[qid] = row
    return by_qid


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    if (
        args.no_load_from_hf
        and args.checkpoint_path is None
        and os.getenv("SAM3_CHECKPOINT_PATH") is None
    ):
        raise ValueError(
            "--no-load-from-hf is set but no checkpoint is available. "
            "Set --checkpoint-path or SAM3_CHECKPOINT_PATH."
        )

    mapping_by_qid = load_mapping(args.mapping_json)

    selected: List[Dict[str, Any]] = []
    with args.qa_jsonl.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            row = json.loads(line)
            qid = int(row["question_id"])
            mapping_row = mapping_by_qid.get(qid)
            if mapping_row is None:
                continue
            cues = [normalize_prompt(x) for x in mapping_row.get("visual_cues", []) if str(x).strip()]
            if not cues:
                continue
            if idx % args.num_shards != args.shard_index:
                continue
            selected.append(
                {
                    "index": idx,
                    "question_id": qid,
                    "image_id": int(mapping_row["image_id"]),
                    "image_name": row["image"],
                    "text": row["text"],
                    "visual_cues": cues,
                }
            )
            if args.limit > 0 and len(selected) >= args.limit:
                break

    if not selected:
        raise RuntimeError("No matched questions selected for this shard.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = args.output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    shard_meta_dir = args.output_dir / "shard_meta"
    shard_meta_dir.mkdir(parents=True, exist_ok=True)

    model = build_sam3_image_model(
        device=args.device,
        eval_mode=True,
        checkpoint_path=args.checkpoint_path,
        load_from_HF=not args.no_load_from_hf,
    )
    transform = create_transform(args.resolution)
    postprocessor = PostProcessImage(
        max_dets_per_img=-1,
        iou_type="segm",
        use_original_sizes_box=True,
        use_original_sizes_mask=True,
        convert_mask_to_rle=False,
        detection_threshold=args.score_thresh,
        to_cpu=True,
    )
    device = torch.device(args.device)

    active_items: Dict[int, Dict[str, Any]] = {}
    pending_units: List[Dict[str, Any]] = []
    processed_units = 0
    finished_questions = 0
    total_questions = len(selected)
    total_units = sum(len(item["visual_cues"]) for item in selected)
    next_query_id = args.shard_index * 10_000_000_000 + 1
    shard_rows: List[Dict[str, Any]] = []

    def process_batch(batch_units: List[Dict[str, Any]]) -> None:
        nonlocal processed_units, finished_questions
        if not batch_units:
            return

        datapoints: List[Datapoint] = []
        for unit in batch_units:
            item = active_items[unit["item_uid"]]
            dp = create_datapoint(
                image=item["image_pil"],
                prompt=unit["cue_prompt"],
                query_id=unit["query_id"],
                image_height=item["height"],
                image_width=item["width"],
            )
            dp = transform(dp)
            datapoints.append(dp)

        batch = collate_fn_api(datapoints, dict_key="batch")["batch"]
        batch = copy_data_to_device(batch, device, non_blocking=True)
        with torch.inference_mode():
            outputs = model(batch)
        processed = postprocessor.process_results(outputs, batch.find_metadatas)

        for unit in batch_units:
            item = active_items[unit["item_uid"]]
            pred = processed.get(unit["query_id"], {})
            mask_2d = result_masks_to_2d(pred.get("masks"), item["height"], item["width"])
            item["merged_mask"] = np.maximum(item["merged_mask"], mask_2d)
            item["remaining_cues"] -= 1
            processed_units += 1

            if item["remaining_cues"] == 0:
                out_path = masks_dir / f"{item['question_id']}.png"
                Image.fromarray(item["merged_mask"], mode="L").save(out_path)
                shard_rows.append(
                    {
                        "question_id": item["question_id"],
                        "image_id": item["image_id"],
                        "image_name": item["image_name"],
                        "text": item["text"],
                        "visual_cues": item["visual_cues"],
                        "mask_path": str(out_path),
                        "mask_ratio": float((item["merged_mask"] > 0).mean()),
                    }
                )
                finished_questions += 1
                del active_items[unit["item_uid"]]

        if processed_units % max(args.batch_size * 20, 500) == 0 or finished_questions == total_questions:
            print(
                f"shard={args.shard_index} questions={finished_questions}/{total_questions} "
                f"cue_units={processed_units}/{total_units}",
                flush=True,
            )

    for item_uid, item in enumerate(selected, start=1):
        image_path = resolve_image_path(item["image_name"], args.image_roots)
        image_pil = Image.open(image_path).convert("RGB")
        width, height = image_pil.size
        active_items[item_uid] = {
            **item,
            "image_pil": image_pil,
            "width": width,
            "height": height,
            "merged_mask": np.zeros((height, width), dtype=np.uint8),
            "remaining_cues": len(item["visual_cues"]),
        }
        for cue_prompt in item["visual_cues"]:
            pending_units.append(
                {
                    "item_uid": item_uid,
                    "query_id": next_query_id,
                    "cue_prompt": cue_prompt,
                }
            )
            next_query_id += 1

        while len(pending_units) >= args.batch_size:
            process_batch(pending_units[: args.batch_size])
            pending_units = pending_units[args.batch_size:]

    while pending_units:
        process_batch(pending_units[: args.batch_size])
        pending_units = pending_units[args.batch_size:]

    if active_items:
        raise RuntimeError(f"Found unfinished items after inference: {len(active_items)}")

    meta = {
        "qa_jsonl": str(args.qa_jsonl),
        "mapping_json": str(args.mapping_json),
        "image_roots": [str(x) for x in args.image_roots],
        "output_dir": str(args.output_dir),
        "total_questions": total_questions,
        "total_prompt_units": total_units,
        "batch_size": args.batch_size,
        "resolution": args.resolution,
        "score_thresh": args.score_thresh,
        "device": args.device,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "rows": shard_rows,
    }
    out_meta = shard_meta_dir / f"shard_{args.shard_index:02d}.json"
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved shard meta: {out_meta}")


if __name__ == "__main__":
    main()
