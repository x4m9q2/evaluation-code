import json
import os
from collections import Counter

DPO_PATH = "/path/to/local_scratch/LLaVA/dpo_train.json"
VQA_PATH = "/path/to/local_scratch/LLaVA/vqa_train2014.json"
ANTI_PATH = "/path/to/local_scratch/LLaVA/train_raw.json"
OUT_PATH = "/path/to/local_scratch/LLaVA/grpo_train.json"

# If you want "train2014/COCO_train2014_000000043500.jpg",
# set IMAGE_PREFIX = "train2014/"
IMAGE_PREFIX = "train2014/"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_question(text: str) -> str:
    text = text.strip()
    if text.startswith("<image>\n"):
        text = text[len("<image>\n"):].strip()
    return text

def build_image_name(image_id: int) -> str:
    return f"{IMAGE_PREFIX}COCO_train2014_{int(image_id):012d}.jpg"

def main():
    if not os.path.exists(DPO_PATH):
        raise FileNotFoundError(f"Missing file: {DPO_PATH}")
    if not os.path.exists(VQA_PATH):
        raise FileNotFoundError(f"Missing file: {VQA_PATH}")
    if not os.path.exists(ANTI_PATH):
        raise FileNotFoundError(f"Missing file: {ANTI_PATH}")

    dpo_data = load_json(DPO_PATH)
    vqa_data = load_json(VQA_PATH)
    anti_data = load_json(ANTI_PATH)

    # Build lookup tables by question_id
    vqa_map = {}
    for item in vqa_data:
        qid = int(item["question_id"])
        vqa_map[qid] = {
            "question": normalize_question(item["question"]),
            "answer": str(item["answer"]).strip(),
            "image_id": int(item["image_id"]),
            "answer_type": item.get("answer_type", None),
        }

    anti_map = {}
    for item in anti_data:
        qid = int(item["question_id"])
        anti_map[qid] = {
            "question": normalize_question(item["question"]),
            "answer": str(item["answer"]).strip(),
            "image_id": int(item["image_id"]),
            "answer_type": item.get("answer_type", None),
        }

    # Extract selected question_ids from dpo_train.json
    selected_qids = []
    for item in dpo_data:
        if "question_id" not in item:
            continue
        selected_qids.append(int(item["question_id"]))

    selected_qids = list(dict.fromkeys(selected_qids))  # deduplicate, preserve order

    missing_in_vqa = []
    missing_in_anti = []
    image_id_mismatch = []
    answer_type_mismatch = []

    out = []

    for qid in selected_qids:
        vqa_item = vqa_map.get(qid)
        anti_item = anti_map.get(qid)

        if vqa_item is None:
            missing_in_vqa.append(qid)
            continue
        if anti_item is None:
            missing_in_anti.append(qid)
            continue

        if vqa_item["image_id"] != anti_item["image_id"]:
            image_id_mismatch.append({
                "question_id": qid,
                "vqa_image_id": vqa_item["image_id"],
                "anti_image_id": anti_item["image_id"],
            })
            continue

        # Keep mismatch record, but still prefer anti answer_type if exists
        if vqa_item.get("answer_type") != anti_item.get("answer_type"):
            answer_type_mismatch.append({
                "question_id": qid,
                "vqa_answer_type": vqa_item.get("answer_type"),
                "anti_answer_type": anti_item.get("answer_type"),
            })

        answer_type = anti_item.get("answer_type") or vqa_item.get("answer_type") or "unknown"

        out.append({
            "group_id": str(qid),
            "question_id": int(qid),
            "image_id": int(vqa_item["image_id"]),
            "image": build_image_name(vqa_item["image_id"]),
            "answer_type": answer_type,
            "original": {
                "question": vqa_item["question"],
                "answer": vqa_item["answer"],
            },
            "anti_shortcut": {
                "question": anti_item["question"],
                "answer": anti_item["answer"],
            }
        })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    stats = {
        "selected_question_ids": len(selected_qids),
        "written_samples": len(out),
        "missing_in_vqa": len(missing_in_vqa),
        "missing_in_anti": len(missing_in_anti),
        "image_id_mismatch": len(image_id_mismatch),
        "answer_type_mismatch": len(answer_type_mismatch),
        "answer_type_distribution": dict(Counter(x["answer_type"] for x in out)),
    }

    print(json.dumps(stats, ensure_ascii=False, indent=2))

    if missing_in_vqa:
        print("\nFirst 20 missing_in_vqa:", missing_in_vqa[:20])
    if missing_in_anti:
        print("\nFirst 20 missing_in_anti:", missing_in_anti[:20])
    if image_id_mismatch:
        print("\nFirst 5 image_id_mismatch:", json.dumps(image_id_mismatch[:5], ensure_ascii=False, indent=2))
    if answer_type_mismatch:
        print("\nFirst 5 answer_type_mismatch:", json.dumps(answer_type_mismatch[:5], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()