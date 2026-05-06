import json

vqa_path = "/path/to/local_scratch/LLaVA/vqa_train2014.json"
train_raw_path = "/path/to/local_scratch/LLaVA/train_raw.json"
output_path = "/path/to/local_scratch/LLaVA/merged_sft_train.json"

# 读取数据
with open(vqa_path, "r") as f:
    vqa_data = json.load(f)

with open(train_raw_path, "r") as f:
    train_raw_data = json.load(f)

# 1. 收集 train_raw 的 question_id
train_raw_qids = set(item["question_id"] for item in train_raw_data)

# 2. 从 vqa 中筛选
filtered_vqa = [item for item in vqa_data if item["question_id"] in train_raw_qids]

print("filtered_vqa:", len(filtered_vqa))
print("train_raw:", len(train_raw_data))

# 3. 合并数据
merged = filtered_vqa + train_raw_data

# 4. 重新编号 question_id
for new_id, item in enumerate(merged):
    item["question_id"] = new_id

# 5. 保存
with open(output_path, "w") as f:
    json.dump(merged, f, indent=2)

print("Saved to:", output_path)
print("Total samples:", len(merged))