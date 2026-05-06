import argparse
import gc
import json
import re
from pathlib import Path


IMAGE_TOKEN_RE = re.compile(r"<image>")
WHITESPACE_RE = re.compile(r"\s+")
QUESTION_TOKEN_RE = re.compile(r"[a-z0-9]+")
VISUAL_CUE_SEP_RE = re.compile(r"[_\-/]+")
VISUAL_CUE_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+")

QUESTION_TOKEN_IRREGULAR_LEMMAS = {
    "people": "person",
    "men": "man",
    "women": "woman",
    "children": "child",
    "mice": "mouse",
    "geese": "goose",
    "teeth": "tooth",
    "feet": "foot",
    "buses": "bus",
    "lenses": "lens",
}

QUESTION_TOKEN_NO_SINGULARIZE = {
    "this",
    "yes",
    "his",
    "hers",
    "ours",
    "yours",
    "theirs",
    "its",
    "thus",
    "news",
    "series",
    "species",
    "glasses",
    "sunglasses",
    "eyeglasses",
    "goggles",
    "pants",
    "jeans",
    "trousers",
    "shorts",
    "scissors",
    "binoculars",
    "pliers",
    "clothes",
    "stairs",
}

VISUAL_CUE_ALIAS_MAP = {
    "television": "tv",
    "television set": "tv",
    "tv monitor": "tv",
    "cellphone": "cell phone",
    "mobile phone": "cell phone",
    "aeroplane": "airplane",
    "air plane": "airplane",
    "motorbike": "motorcycle",
    "motor bike": "motorcycle",
}

VISUAL_CUE_IRREGULAR_SINGULARS = {
    "people": "person",
    "men": "man",
    "women": "woman",
    "children": "child",
    "mice": "mouse",
    "geese": "goose",
    "teeth": "tooth",
    "feet": "foot",
    "buses": "bus",
    "lenses": "lens",
}

VISUAL_CUE_NO_SINGULARIZE = {
    "glasses",
    "sunglasses",
    "eyeglasses",
    "goggles",
    "pants",
    "jeans",
    "trousers",
    "shorts",
    "scissors",
    "binoculars",
    "pliers",
    "clothes",
    "stairs",
    "news",
    "series",
    "species",
}


def normalize_text(text):
    text = IMAGE_TOKEN_RE.sub(" ", text or "")
    return WHITESPACE_RE.sub(" ", text).strip()


def singularize_question_token(token):
    if token in QUESTION_TOKEN_IRREGULAR_LEMMAS:
        return QUESTION_TOKEN_IRREGULAR_LEMMAS[token]
    if token in QUESTION_TOKEN_NO_SINGULARIZE or len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ves") and len(token) > 4:
        stem = token[:-3]
        if stem.endswith("i"):
            return stem + "fe"
        return stem + "f"
    if token.endswith("sses") and len(token) > 4:
        return token[:-2]
    if token.endswith(("ches", "shes", "xes", "zes")) and len(token) > 4:
        return token[:-2]
    if token.endswith("es") and len(token) > 4 and token[:-1].endswith("e"):
        return token[:-1]
    if token.endswith("s") and not token.endswith(("ss", "us", "is", "es")):
        return token[:-1]
    return token


def normalize_question_token(token):
    token = (token or "").strip().lower()
    if not token:
        return ""
    if token == "s":
        return ""
    token = singularize_question_token(token)
    return token


def tokenize_question_keywords(text):
    normalized_text = normalize_text(text).lower()
    return [
        normalized_token
        for token in QUESTION_TOKEN_RE.findall(normalized_text)
        if (normalized_token := normalize_question_token(token))
    ]


def singularize_visual_cue_token(token):
    if token in VISUAL_CUE_IRREGULAR_SINGULARS:
        return VISUAL_CUE_IRREGULAR_SINGULARS[token]
    if token in VISUAL_CUE_NO_SINGULARIZE or len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ves") and len(token) > 4:
        stem = token[:-3]
        if stem.endswith("i"):
            return stem + "fe"
        return stem + "f"
    if token.endswith("sses") and len(token) > 4:
        return token[:-2]
    if token.endswith(("ches", "shes", "xes", "zes")) and len(token) > 4:
        return token[:-2]
    if token.endswith("es") and len(token) > 4 and token[:-1].endswith("e"):
        return token[:-1]
    if token.endswith("s") and not token.endswith(("ss", "us", "is", "es")):
        return token[:-1]
    return token


def normalize_visual_cue(text):
    cue = normalize_text(text).lower()
    if not cue:
        return ""

    cue = VISUAL_CUE_SEP_RE.sub(" ", cue)
    cue = VISUAL_CUE_NON_ALNUM_RE.sub(" ", cue)
    cue = WHITESPACE_RE.sub(" ", cue).strip()
    if not cue:
        return ""

    tokens = cue.split(" ")
    if len(tokens) > 1 and tokens[0] in {"a", "an", "the"}:
        tokens = tokens[1:]
    if not tokens:
        return ""

    cue = " ".join(tokens)
    cue = VISUAL_CUE_ALIAS_MAP.get(cue, cue)
    tokens = cue.split(" ")
    if tokens:
        tokens[-1] = singularize_visual_cue_token(tokens[-1])

    cue = " ".join(token for token in tokens if token)
    cue = VISUAL_CUE_ALIAS_MAP.get(cue, cue)
    return cue


def iter_json_array(path, chunk_size=1 << 20):
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as f:
        buffer = ""
        eof = False

        while True:
            if not eof and len(buffer) < chunk_size:
                chunk = f.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True

            stripped = buffer.lstrip()
            if stripped:
                if stripped[0] != "[":
                    raise ValueError(f"{path} is not a top-level JSON array")
                buffer = stripped[1:]
                break

            if eof:
                raise ValueError(f"{path} is empty")

        while True:
            buffer = buffer.lstrip()
            if buffer.startswith("]"):
                return
            if buffer.startswith(","):
                buffer = buffer[1:]
                continue
            if not buffer and eof:
                raise ValueError(f"{path} ended unexpectedly while reading array items")

            while True:
                try:
                    item, idx = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    chunk = f.read(chunk_size)
                    if chunk:
                        buffer += chunk
                    else:
                        eof = True

            yield item
            buffer = buffer[idx:]

            if not eof and len(buffer) < chunk_size:
                chunk = f.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True


class JsonArrayWriter:
    def __init__(self, path, key):
        self.path = path
        self.key = key
        self.first = True
        self.fp = open(path, "w", encoding="utf-8")
        self.fp.write("{\n")
        self.fp.write(json.dumps(key))
        self.fp.write(": [\n")

    def write(self, obj):
        if not self.first:
            self.fp.write(",\n")
        json.dump(obj, self.fp, ensure_ascii=False)
        self.first = False

    def close(self):
        self.fp.write("\n]\n}\n")
        self.fp.close()


class JsonObjectWriter:
    def __init__(self, path):
        self.first = True
        self.fp = open(path, "w", encoding="utf-8")
        self.fp.write("{\n")

    def write(self, key, value):
        if not self.first:
            self.fp.write(",\n")
        self.fp.write(json.dumps(str(key), ensure_ascii=False))
        self.fp.write(": ")
        json.dump(value, self.fp, ensure_ascii=False)
        self.first = False

    def close(self):
        self.fp.write("\n}\n")
        self.fp.close()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_qa_pair(conversations):
    question = None
    answer = None
    for turn in conversations:
        role = (turn.get("from") or "").lower()
        value = normalize_text(turn.get("value", ""))
        if role in {"human", "user"} and question is None:
            question = value
        elif role in {"gpt", "assistant"} and answer is None:
            answer = value
        if question is not None and answer is not None:
            break
    return question, answer


def build_questions_and_annotations(llava_json_path, out_dir, limit=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    questions_path = out_dir / "questions.json"
    annotations_path = out_dir / "annotations.json"
    image_map_path = out_dir / "image_mappings.jsonl"

    image_path_to_id = {}
    next_image_id = 1
    next_question_id = 1
    written = 0

    questions_writer = JsonArrayWriter(questions_path, "questions")
    annotations_writer = JsonArrayWriter(annotations_path, "annotations")

    with open(image_map_path, "w", encoding="utf-8") as image_map_fp:
        for item in iter_json_array(llava_json_path):
            question, answer = extract_qa_pair(item.get("conversations", []))
            if not question or answer is None:
                continue

            image_path = item["image"]
            dataset = image_path.split("/", 1)[0]
            if image_path not in image_path_to_id:
                image_id = next_image_id
                image_path_to_id[image_path] = image_id
                next_image_id += 1
                image_map_fp.write(
                    json.dumps(
                        {
                            "image_id": image_id,
                            "image": image_path,
                            "dataset": dataset,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            else:
                image_id = image_path_to_id[image_path]

            question_id = next_question_id
            next_question_id += 1

            questions_writer.write(
                {
                    "question_id": question_id,
                    "image_id": image_id,
                    "question": question,
                    "llava_id": item.get("id"),
                    "image": image_path,
                    "dataset": dataset,
                }
            )
            annotations_writer.write(
                {
                    "question_id": question_id,
                    "image_id": image_id,
                    "llava_id": item.get("id"),
                    "image": image_path,
                    "dataset": dataset,
                    "multiple_choice_answer": answer,
                    "answers": [{"answer_id": 1, "answer": answer}],
                }
            )

            written += 1
            if limit is not None and written >= limit:
                break

    questions_writer.close()
    annotations_writer.close()
    return {
        "num_questions": written,
        "num_unique_images": len(image_path_to_id),
        "image_path_to_id": image_path_to_id,
        "questions_path": questions_path,
        "annotations_path": annotations_path,
        "image_map_path": image_map_path,
    }


def write_detection_entry(writer, image_id, image_path, classes, written_paths):
    normalized = sorted({cue for x in classes if (cue := normalize_visual_cue(x))})
    writer.write(
        image_id,
        {
            "image_id": image_id,
            "image": image_path,
            "classes": normalized,
            "scores": [1.0] * len(normalized),
        },
    )
    written_paths.add(image_path)


def build_coco_detections(annotation_root, image_path_to_id, writer, written_paths):
    coco_needed = {path for path in image_path_to_id if path.startswith("coco/")}
    if not coco_needed:
        return 0

    count = 0
    for split in ("train2017", "val2017"):
        anno_path = annotation_root / "coco" / f"instances_{split}.json"
        if not anno_path.exists():
            continue

        data = load_json(anno_path)
        categories = {c["id"]: c["name"] for c in data["categories"]}
        image_id_to_path = {}
        for image in data["images"]:
            rel_path = f"coco/{split}/{image['file_name']}"
            if rel_path in coco_needed:
                image_id_to_path[image["id"]] = rel_path

        if not image_id_to_path:
            continue

        classes_by_path = {path: set() for path in image_id_to_path.values()}
        for ann in data["annotations"]:
            rel_path = image_id_to_path.get(ann["image_id"])
            if rel_path is None:
                continue
            name = categories.get(ann["category_id"])
            if name:
                classes_by_path[rel_path].add(name)

        for rel_path, classes in classes_by_path.items():
            write_detection_entry(
                writer,
                image_path_to_id[rel_path],
                rel_path,
                classes,
                written_paths,
            )
            count += 1

        del data
        gc.collect()

    return count


def build_gqa_detections(annotation_root, image_path_to_id, writer, written_paths):
    gqa_needed = {path for path in image_path_to_id if path.startswith("gqa/")}
    if not gqa_needed:
        return 0

    count = 0
    for split in ("train_sceneGraphs.json", "val_sceneGraphs.json"):
        anno_path = annotation_root / "gqa" / split
        if not anno_path.exists():
            continue

        data = load_json(anno_path)
        for raw_image_id, scene_graph in data.items():
            rel_path = f"gqa/images/{raw_image_id}.jpg"
            if rel_path not in gqa_needed or rel_path in written_paths:
                continue

            classes = []
            for obj in scene_graph.get("objects", {}).values():
                name = obj.get("name")
                if name:
                    classes.append(name)

            write_detection_entry(
                writer,
                image_path_to_id[rel_path],
                rel_path,
                classes,
                written_paths,
            )
            count += 1

        del data
        gc.collect()

    return count


def build_vg_detections(annotation_root, image_path_to_id, writer, written_paths):
    vg_needed = {path for path in image_path_to_id if path.startswith("vg/")}
    if not vg_needed:
        return 0

    count = 0
    anno_path = annotation_root / "visual_genome" / "objects.json"
    for entry in iter_json_array(anno_path):
        raw_image_id = entry["image_id"]
        candidate_paths = [
            f"vg/VG_100K/{raw_image_id}.jpg",
            f"vg/VG_100K_2/{raw_image_id}.jpg",
        ]
        matching_paths = [
            path for path in candidate_paths if path in vg_needed and path not in written_paths
        ]
        if not matching_paths:
            continue

        classes = []
        for obj in entry.get("objects", []):
            names = obj.get("names") or []
            classes.extend(names)

        for rel_path in matching_paths:
            write_detection_entry(
                writer,
                image_path_to_id[rel_path],
                rel_path,
                classes,
                written_paths,
            )
            count += 1

    return count


def build_detections(annotation_root, image_path_to_id, out_dir):
    detections_path = out_dir / "image_to_detection.json"
    writer = JsonObjectWriter(detections_path)
    written_paths = set()

    counts = {
        "coco": build_coco_detections(annotation_root, image_path_to_id, writer, written_paths),
        "gqa": build_gqa_detections(annotation_root, image_path_to_id, writer, written_paths),
        "vg": build_vg_detections(annotation_root, image_path_to_id, writer, written_paths),
    }

    missing = []
    for image_path, image_id in image_path_to_id.items():
        if image_path in written_paths:
            continue
        missing.append(image_path)
        write_detection_entry(writer, image_id, image_path, [], written_paths)

    writer.close()
    counts["missing"] = len(missing)
    return detections_path, counts, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llava-json",
        default="/path/to/sage_repro_bundle/playground/data/llava_v1_5_mix665k_single_noocr_max200_imageonly_strict_noocr.json",
        help="Path to the LLaVA mixture JSON file.",
    )
    parser.add_argument(
        "--annotation-root",
        default="/path/to/sage_repro_bundle/object_annotation_bundle",
        help="Directory containing coco/gqa/visual_genome object annotations.",
    )
    parser.add_argument(
        "--out-dir",
        default="/path/to/sage_repro_bundle/shortcut_inputs/llava_mix665k_single_noocr",
        help="Directory to store converted shortcut inputs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample limit for smoke tests.",
    )
    args = parser.parse_args()

    llava_json_path = Path(args.llava_json)
    annotation_root = Path(args.annotation_root)
    out_dir = Path(args.out_dir)

    if not llava_json_path.exists():
        raise FileNotFoundError(f"Missing LLaVA JSON: {llava_json_path}")
    if not annotation_root.exists():
        raise FileNotFoundError(f"Missing annotation root: {annotation_root}")

    question_stats = build_questions_and_annotations(llava_json_path, out_dir, limit=args.limit)
    detections_path, detection_counts, missing = build_detections(
        annotation_root, question_stats["image_path_to_id"], out_dir
    )

    metadata = {
        "llava_json": str(llava_json_path),
        "annotation_root": str(annotation_root),
        "questions_path": str(question_stats["questions_path"]),
        "annotations_path": str(question_stats["annotations_path"]),
        "image_mappings_path": str(question_stats["image_map_path"]),
        "detections_path": str(detections_path),
        "num_questions": question_stats["num_questions"],
        "num_unique_images": question_stats["num_unique_images"],
        "detection_counts": detection_counts,
        "missing_detection_examples": missing[:20],
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
