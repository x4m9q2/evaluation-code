import json
import os
from typing import Any


def _load_rules_payload(path: str):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "rules" in payload:
        return payload["rules"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported rules JSON format: {path}")


def convert_rules_json(rules_json_path: str, output_json_path: str):
    rules = _load_rules_payload(rules_json_path)

    optimized_rules = []
    for rule in rules:
        rule_id = rule.get("rule_id", "")
        if isinstance(rule_id, int):
            rule_id = str(rule_id)

        optimized_rule = {
            "rule_id": rule_id,
            "text_keywords": [kw.strip().lower() for kw in rule.get("text_keywords", []) if kw],
            "visual_cues": [vc.strip().lower() for vc in rule.get("visual_cues", []) if vc],
            "confidence": float(rule.get("confidence", 0.0)),
            "support": rule.get("support", 0),
            "answer": rule.get("answer", "").strip().lower(),
        }
        optimized_rules.append(optimized_rule)

    optimized_rules.sort(key=lambda r: (r["confidence"], r["support"]), reverse=True)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump({"rules": optimized_rules}, f, ensure_ascii=False, indent=2)


def convert_and_merge_questions_annotations(
    questions_json_path: str,
    annotations_json_path: str,
    output_json_path: str,
):
    with open(questions_json_path, "r", encoding="utf-8") as f:
        q_data = json.load(f)
        questions = q_data["questions"] if isinstance(q_data, dict) and "questions" in q_data else q_data

    with open(annotations_json_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)["annotations"]

    answer_map = {int(a["question_id"]): a["answers"][0]["answer"].strip().lower() for a in annotations}

    optimized_questions = []
    for q in questions:
        qid = int(q["question_id"])
        optimized_question = {
            "question_id": qid,
            "image_id": int(q["image_id"]),
            "question_text": q["question"].strip().lower(),
            "answer": answer_map.get(qid, "").strip().lower(),
        }
        optimized_questions.append(optimized_question)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump({"questions": optimized_questions}, f, ensure_ascii=False, indent=2)


def convert_detections_json(det_json_path: str, output_json_path: str, score_thr: float = 0.3):
    with open(det_json_path, "r", encoding="utf-8") as f:
        detections = json.load(f)
    if isinstance(detections, dict) and "detections" in detections:
        detections = {str(item["image_id"]): item for item in detections["detections"]}

    optimized_detections = {}
    for img_id, det in detections.items():
        classes = [
            c.strip().lower() for c, s in zip(det.get("classes", []), det.get("scores", [])) if s is None or float(s) >= score_thr
        ]
        optimized_detections[img_id] = {"image_id": int(img_id), "classes": classes}

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump({"detections": list(optimized_detections.values())}, f, ensure_ascii=False, indent=2)


def convert_all_files(args: Any):
    output_dir = os.path.join(os.path.dirname(args.out_json), "optimized_json")
    os.makedirs(output_dir, exist_ok=True)

    print("Converting rules.json...")
    convert_rules_json(args.rules_json, os.path.join(output_dir, "optimized_rules.json"))

    print("Merging questions and annotations...")
    convert_and_merge_questions_annotations(
        args.questions_json,
        args.annotations_json,
        os.path.join(output_dir, "optimized_questions_answers.json"),
    )

    print("Converting detections.json...")
    convert_detections_json(
        args.det_json,
        os.path.join(output_dir, "optimized_detections.json"),
        score_thr=args.score_thr,
    )

    print(f"All files converted and saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rules_json", required=False, default="/home/arcaea/jty/rules.json")
    parser.add_argument("--questions_json", required=False, default="/home/arcaea/jty/v2_OpenEnded_mscoco_train2014_questions.json")
    parser.add_argument("--annotations_json", required=False, default="/home/arcaea/jty/v2_mscoco_train2014_annotations.json")
    parser.add_argument("--det_json", required=False, default="/home/arcaea/jty/image_to_detection.json")
    parser.add_argument("--out_json", required=False, default="/home/arcaea/jty/out.json")
    parser.add_argument("--score_thr", type=float, default=0.3)

    args = parser.parse_args()
    convert_all_files(args)
