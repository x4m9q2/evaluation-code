#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

SCRIPT_PATH = Path(__file__).resolve()
BUNDLE_ROOT = SCRIPT_PATH.parents[3]
LLAVA_CODE_ROOT = BUNDLE_ROOT / "code/llava_sage"
DEFAULT_VISION_TOWER = BUNDLE_ROOT / "models/clip-vit-large-patch14-336"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run POPE inference and metrics for one LLaVA checkpoint.")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--has-gate", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--question-file", type=Path, default=BUNDLE_ROOT / "data/pope/llava_pope_test.jsonl")
    parser.add_argument("--annotation-dir", type=Path, default=BUNDLE_ROOT / "data/pope/coco")
    parser.add_argument("--image-folder", type=Path, default=BUNDLE_ROOT / "data/pope/val2014")
    parser.add_argument("--vision-tower", type=Path, default=DEFAULT_VISION_TOWER)
    parser.add_argument("--output-root", type=Path, default=BUNDLE_ROOT / "outputs/pope")
    parser.add_argument("--gpu", default="0,1,2,3")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
    return rows


def normalize_yes_no(text: str) -> str:
    text = text.strip()
    if "." in text:
        text = text.split(".", 1)[0]
    words = text.replace(",", "").split()
    lowered = {word.lower() for word in words}
    if "no" in lowered or "not" in lowered:
        return "no"
    return "yes"


def binary_metrics(preds: Iterable[int], labels: Iterable[int]) -> dict:
    pred_list = list(preds)
    label_list = list(labels)
    tp = fp = tn = fn = 0
    for pred, label in zip(pred_list, label_list):
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        elif pred == 0 and label == 1:
            fn += 1
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "samples": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_ratio": pred_list.count(1) / len(pred_list) if pred_list else 0.0,
    }


def load_predictions(path: Path) -> Dict[int, dict]:
    preds = {}
    for row in load_jsonl(path):
        qid = int(row["question_id"])
        if qid in preds:
            raise RuntimeError(f"Duplicate question_id {qid} in {path}")
        preds[qid] = row
    return preds


def merge_prediction_files(paths: List[Path]) -> Dict[int, dict]:
    merged = {}
    for path in paths:
        part = load_predictions(path)
        overlap = set(merged).intersection(part)
        if overlap:
            raise RuntimeError(f"Duplicate question_ids across chunks: {sorted(overlap)[:10]}")
        merged.update(part)
    return merged


def compute_pope_metrics(questions: List[dict], pred_map: Dict[int, dict], annotation_dir: Path) -> dict:
    categories = {}
    overall_preds = []
    overall_labels = []

    for ann_path in sorted(annotation_dir.glob("coco_pope_*.json")):
        category = ann_path.name[len("coco_pope_") : -len(".json")]
        label_rows = load_jsonl(ann_path)
        category_questions = [row for row in questions if row.get("category") == category]
        if len(category_questions) != len(label_rows):
            raise RuntimeError(
                f"POPE category {category} has {len(category_questions)} questions but "
                f"{len(label_rows)} labels in {ann_path}"
            )
        preds = []
        labels = []
        for question, label_row in zip(category_questions, label_rows):
            qid = int(question["question_id"])
            pred_row = pred_map.get(qid)
            if pred_row is None:
                raise RuntimeError(f"Missing prediction for question_id {qid}")
            pred_answer = normalize_yes_no(str(pred_row.get("text", "")))
            label_answer = str(label_row["label"]).lower()
            preds.append(1 if pred_answer == "yes" else 0)
            labels.append(1 if label_answer == "yes" else 0)
        categories[category] = binary_metrics(preds, labels)
        overall_preds.extend(preds)
        overall_labels.extend(labels)

    return {
        "categories": categories,
        "overall": binary_metrics(overall_preds, overall_labels),
    }


def run_inference(args: argparse.Namespace, out_dir: Path) -> Path:
    gpu_list = [x.strip() for x in args.gpu.split(",") if x.strip()]
    if not gpu_list:
        raise ValueError("At least one GPU is required")

    base_cmd = [
        sys.executable,
        "-m",
        "llava.eval.model_vqa_loader",
        "--model-path",
        str(args.model_path),
        "--image-folder",
        str(args.image_folder),
        "--question-file",
        str(args.question_file),
        "--conv-mode",
        args.conv_mode,
        "--temperature",
        str(args.temperature),
        "--num_beams",
        str(args.num_beams),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--force-use-dual-input-gate",
        args.has_gate,
    ]
    merged_path = out_dir / "predictions.jsonl"
    pred_files: List[Path] = []

    if len(gpu_list) == 1:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_list[0]
        env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
        env["PYTHONPATH"] = f"{LLAVA_CODE_ROOT}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(LLAVA_CODE_ROOT)
        env.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
        env["VISION_TOWER"] = str(args.vision_tower)
        cmd = base_cmd + ["--answers-file", str(merged_path)]
        subprocess.run(cmd, cwd=LLAVA_CODE_ROOT, env=env, check=True)
        return merged_path

    procs = []
    cmds = []
    for chunk_idx, gpu_id in enumerate(gpu_list):
        chunk_path = out_dir / f"predictions_chunk{chunk_idx}.jsonl"
        pred_files.append(chunk_path)
        cmd = base_cmd + [
            "--answers-file",
            str(chunk_path),
            "--num-chunks",
            str(len(gpu_list)),
            "--chunk-idx",
            str(chunk_idx),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
        env["PYTHONPATH"] = f"{LLAVA_CODE_ROOT}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(LLAVA_CODE_ROOT)
        env.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
        env["VISION_TOWER"] = str(args.vision_tower)
        print(f"[run] POPE chunk={chunk_idx} gpu={gpu_id}", flush=True)
        procs.append(subprocess.Popen(cmd, cwd=LLAVA_CODE_ROOT, env=env))
        cmds.append(cmd)

    for idx, proc in enumerate(procs):
        ret = proc.wait()
        if ret != 0:
            raise subprocess.CalledProcessError(ret, cmds[idx])

    pred_map = merge_prediction_files(pred_files)
    questions = load_jsonl(args.question_file)
    with merged_path.open("w", encoding="utf-8") as f:
        for row in questions:
            pred_row = pred_map.get(int(row["question_id"]))
            if pred_row is None:
                raise RuntimeError(f"Missing prediction for question_id {row['question_id']}")
            f.write(json.dumps(pred_row, ensure_ascii=False) + "\n")
    for path in pred_files:
        path.unlink(missing_ok=True)
    return merged_path


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"model path not found: {args.model_path}")
    for path, desc in [
        (args.question_file, "question file"),
        (args.annotation_dir, "annotation dir"),
        (args.image_folder, "image folder"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{desc} not found: {path}")

    model_tag = args.model_path.name
    out_dir = args.output_root / model_tag / "pope"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "pope_metrics.json"
    eval_txt_path = out_dir / "pope_eval.txt"
    predictions_path = out_dir / "predictions.jsonl"

    if metrics_path.exists() and predictions_path.exists() and not args.overwrite:
        print(f"[skip] POPE metrics already exist: {metrics_path}", flush=True)
        return

    predictions_path = run_inference(args, out_dir)
    questions = load_jsonl(args.question_file)
    pred_map = load_predictions(predictions_path)
    metrics = compute_pope_metrics(questions, pred_map, args.annotation_dir)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
        f.write("\n")

    lines = []
    for category, values in metrics["categories"].items():
        lines.append(f"Category: {category}, # samples: {values['samples']}")
        lines.append(
            "F1={f1:.4f} Acc={accuracy:.4f} Precision={precision:.4f} "
            "Recall={recall:.4f} Yes={yes_ratio:.4f}".format(**values)
        )
    overall = metrics["overall"]
    lines.append(f"Overall, # samples: {overall['samples']}")
    lines.append(
        "F1={f1:.4f} Acc={accuracy:.4f} Precision={precision:.4f} "
        "Recall={recall:.4f} Yes={yes_ratio:.4f}".format(**overall)
    )
    eval_txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] POPE: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
