#!/usr/bin/env python3
"""Convert COCO instances_train2014.json into image_to_detection.json.

The output format matches the loader used by the shortcut mining pipeline:

    {
      "123": {"classes": ["person", "chair"], "scores": [1.0, 1.0]},
      ...
    }

Optionally, a JSON file containing a list of image ids can be supplied to
restrict the output to a subset.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPE_ROOT = REPO_ROOT / "data" / "shortcut_pipeline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instances-json",
        default=str(REPO_ROOT / "annotations" / "instances_train2014.json"),
        help="Path to COCO instances_train2014.json.",
    )
    parser.add_argument(
        "--output-json",
        default=str(PIPE_ROOT / "image_to_detection.json"),
        help="Destination image_to_detection.json.",
    )
    parser.add_argument(
        "--image-ids-json",
        default="",
        help="Optional JSON file containing a list of image ids to keep.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Optional hard limit on the number of image entries written.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_keep_ids(path: str) -> Optional[Set[str]]:
    if not path:
        return None
    payload = load_json(Path(path))
    if isinstance(payload, dict):
        if "image_ids" in payload and isinstance(payload["image_ids"], list):
            return {str(v) for v in payload["image_ids"]}
        if "results" in payload and isinstance(payload["results"], list):
            ids = set()
            for item in payload["results"]:
                if isinstance(item, dict) and "image_id" in item:
                    ids.add(str(item["image_id"]))
            return ids
    if isinstance(payload, list):
        return {str(v) for v in payload}
    raise RuntimeError(f"Unsupported image id list format in {path}")


def main() -> None:
    args = parse_args()
    instances_path = Path(args.instances_json).resolve()
    output_path = Path(args.output_json).resolve()
    keep_ids = load_keep_ids(args.image_ids_json)

    if not instances_path.exists():
        raise FileNotFoundError(f"instances file not found: {instances_path}")

    coco_data = load_json(instances_path)
    category_id_to_name = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

    image_id_to_classes = defaultdict(list)
    seen = defaultdict(set)
    for ann in coco_data["annotations"]:
        image_id = str(ann["image_id"])
        if keep_ids is not None and image_id not in keep_ids:
            continue
        category_name = category_id_to_name[ann["category_id"]]
        if category_name in seen[image_id]:
            continue
        seen[image_id].add(category_name)
        image_id_to_classes[image_id].append(category_name)

    output: Dict[str, Dict[str, Any]] = {}
    for image_id, classes in image_id_to_classes.items():
        output[image_id] = {
            "image_id": int(image_id),
            "classes": sorted(classes),
            "scores": [1.0] * len(classes),
        }
        if args.max_images and len(output) >= args.max_images:
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False)

    print(f"Wrote {len(output):,} image entries to {output_path}")


if __name__ == "__main__":
    main()
