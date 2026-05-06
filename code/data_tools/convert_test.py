import json

input_file = "train_raw.json"
output_file = "train_raw_llava.jsonl"

data = json.load(open(input_file))

with open(output_file, "w") as f:
    for d in data:
        image = f"COCO_train2014_{d['image_id']:012d}.jpg"
        text = d["question"]

        out = {
            "question_id": d["question_id"],
            "image": image,
            "text": text
        }

        f.write(json.dumps(out) + "\n")