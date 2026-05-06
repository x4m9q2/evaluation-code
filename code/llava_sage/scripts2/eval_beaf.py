#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


SCRIPT_PATH = Path(__file__).resolve()
LLAVA_CODE_ROOT = SCRIPT_PATH.parents[1]
BUNDLE_ROOT = LLAVA_CODE_ROOT.parents[1]

DEFAULT_QNA_PATH = BUNDLE_ROOT / "data/beaf/beaf_qna.json"
DEFAULT_IMAGE_FOLDER = BUNDLE_ROOT / "data/beaf/images"
DEFAULT_OUTPUT_ROOT = BUNDLE_ROOT / "outputs/beaf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run BEAF inference for a single model, merge chunk outputs, and "
            "compute BEAF metrics."
        )
    )
    parser.add_argument("model_path", type=Path, help="模型路径")
    parser.add_argument(
        "has_gate",
        choices=["auto", "true", "false"],
        help="推理时是否强制启用双输入门控",
    )
    parser.add_argument("batch_size", type=int, help="推理 batch size")
    parser.add_argument("gpu", help="使用哪些 GPU，例如 0,1,2")
    parser.add_argument("--qna-path", type=Path, default=DEFAULT_QNA_PATH)
    parser.add_argument("--image-folder", type=Path, default=DEFAULT_IMAGE_FOLDER)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果输出已存在则覆盖",
    )
    return parser.parse_args()


def ensure_exists(path: Path, kind: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{kind} not found: {path}")


def load_qna(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Expected JSON array in {path}")
    rows = sorted(rows, key=lambda x: int(x["id"]))
    return rows


def build_question_file(qna_rows: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in qna_rows:
            row = {
                "question_id": int(item["id"]),
                "image": item["image"],
                "text": item["question"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_predictions(path: Path) -> Dict[int, dict]:
    preds: Dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid prediction JSONL at {path}:{line_no}") from exc
            preds[int(row["question_id"])] = row
    return preds


def merge_prediction_files(paths: List[Path]) -> Dict[int, dict]:
    merged: Dict[int, dict] = {}
    for path in paths:
        preds = load_predictions(path)
        overlap = set(merged).intersection(preds)
        if overlap:
            raise RuntimeError(
                f"Duplicate question_id found across chunk outputs in {path}. "
                f"First few: {sorted(overlap)[:10]}"
            )
        merged.update(preds)
    return merged


def normalize_beaf_answer(text: str) -> Tuple[str, str]:
    lower = text.lower()
    if "yes" in lower:
        return "yes", "contains_yes"
    if "no" in lower:
        return "no", "contains_no"

    token = lower.strip().split()
    if token:
        if token[0].startswith("y"):
            return "yes", "starts_y"
        if token[0].startswith("n"):
            return "no", "starts_n"
    if re.search(r"\bnot\b", lower):
        return "no", "contains_not"
    return "invalid", "unparsed"


def derive_original_name(image_name: str) -> str:
    orig = re.sub(r"_[0-9]{2}\.(png|jpg)$", ".jpg", image_name)
    if orig == image_name:
        return image_name
    return orig


def compute_metrics(qna_rows: List[dict], answers: List[dict]) -> dict:
    qna_by_id = {int(row["id"]): row for row in qna_rows}
    orig_pairs: Dict[str, Dict[str, str]] = {}

    counts = {
        "TP": 0,
        "FP": 0,
        "TN": 0,
        "FN": 0,
        "TU": 0,
        "IG": 0,
        "SBp": 0,
        "SBn": 0,
        "ID": 0,
        "invalid": 0,
    }
    invalid_examples = []
    conv = {"TPTN": "TU", "FNFP": "IG", "TPFP": "SBp", "FNTN": "SBn"}

    detailed_rows = []
    for answer_row in answers:
        qid = int(answer_row["id"])
        qna = qna_by_id[qid]
        parsed_answer, parse_reason = normalize_beaf_answer(answer_row["answer"])
        if parsed_answer == "invalid":
            counts["invalid"] += 1
            if len(invalid_examples) < 20:
                invalid_examples.append(
                    {
                        "id": qid,
                        "image": qna["image"],
                        "question": qna["question"],
                        "raw_answer": answer_row["answer"],
                    }
                )
            # Keep binary eval robust by treating invalid as "no".
            parsed_answer = "no"

        gt = qna["gt"]
        if gt == "yes" and parsed_answer == "yes":
            cls = "TP"
        elif gt == "no" and parsed_answer == "no":
            cls = "TN"
        elif gt == "yes" and parsed_answer == "no":
            cls = "FN"
        else:
            cls = "FP"

        counts[cls] += 1

        if qna["orig_img"]:
            orig_pairs.setdefault(qna["image"], {})[qna["question"]] = cls

        detailed_rows.append(
            {
                "id": qid,
                "image": qna["image"],
                "question": qna["question"],
                "orig_img": bool(qna["orig_img"]),
                "removed_q": bool(qna["removed_q"]),
                "gt": gt,
                "raw_answer": answer_row["answer"],
                "parsed_answer": parsed_answer,
                "parse_reason": parse_reason,
                "label": cls,
            }
        )

    removed_eval_total = 0
    id_total = 0
    missing_orig_pairs = []
    for row in detailed_rows:
        if row["orig_img"]:
            continue
        orig_name = derive_original_name(row["image"])
        orig_answer = orig_pairs.get(orig_name, {}).get(row["question"])
        if orig_answer is None:
            if len(missing_orig_pairs) < 20:
                missing_orig_pairs.append(
                    {
                        "id": row["id"],
                        "image": row["image"],
                        "question": row["question"],
                        "expected_orig_image": orig_name,
                    }
                )
            continue

        if row["removed_q"]:
            removed_eval_total += 1
            mapped = conv.get(orig_answer + row["label"])
            if mapped is not None:
                counts[mapped] += 1
        else:
            id_total += 1
            if orig_answer[0] != row["label"][0]:
                counts["ID"] += 1

    total = counts["TP"] + counts["FP"] + counts["TN"] + counts["FN"]
    acc = 100.0 * (counts["TP"] + counts["TN"]) / total if total else 0.0
    precision = 100.0 * counts["TP"] / (counts["TP"] + counts["FP"]) if counts["TP"] + counts["FP"] else 0.0
    recall = 100.0 * counts["TP"] / (counts["TP"] + counts["FN"]) if counts["TP"] + counts["FN"] else 0.0
    f1_pr = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    tu = 100.0 * counts["TU"] / removed_eval_total if removed_eval_total else 0.0
    ig = 100.0 * counts["IG"] / removed_eval_total if removed_eval_total else 0.0
    sbp = 100.0 * counts["SBp"] / removed_eval_total if removed_eval_total else 0.0
    sbn = 100.0 * counts["SBn"] / removed_eval_total if removed_eval_total else 0.0
    id_rate = 100.0 * counts["ID"] / id_total if id_total else 0.0
    f1_tuid = 2 * tu * (100.0 - id_rate) / (tu + (100.0 - id_rate)) if tu + (100.0 - id_rate) else 0.0

    return {
        "counts": counts,
        "totals": {
            "all_pairs": total,
            "removed_eval_total": removed_eval_total,
            "id_total": id_total,
            "missing_orig_pairs": len(missing_orig_pairs),
        },
        "metrics": {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_pr": f1_pr,
            "tu": tu,
            "ig": ig,
            "sbp": sbp,
            "sbn": sbn,
            "id": id_rate,
            "f1_tuid": f1_tuid,
        },
        "invalid_examples": invalid_examples,
        "missing_orig_pair_examples": missing_orig_pairs,
    }


def write_eval_text(metrics: dict, out_path: Path) -> None:
    m = metrics["metrics"]
    lines = [
        "========================================================",
        "   Accuracy  |  Precision  |    Recall   |    F1(P,R) ",
        "--------------------------------------------------------",
        f"    {m['accuracy']:.2f}    |    {m['precision']:.2f}    |    {m['recall']:.2f}    |    {m['f1_pr']:.2f}",
        "=========================================================",
        "   TU   |   IG   |   SB+  |   SB-  |   ID   | F1(TU,ID)",
        "---------------------------------------------------------",
        f" {m['tu']:.2f}  |  {m['ig']:.2f}  |  {m['sbp']:.2f} |  {m['sbn']:.2f} |  {m['id']:.2f}  |   {m['f1_tuid']:.2f}",
        "=========================================================",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cleanup_intermediate_files(paths: List[Path]) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def main() -> None:
    args = parse_args()
    ensure_exists(args.model_path, "model path")
    ensure_exists(args.qna_path, "BEAF qna file")
    ensure_exists(args.image_folder, "BEAF image folder")
    if not (args.model_path / "config.json").exists():
        raise FileNotFoundError(f"config.json not found under model path: {args.model_path}")

    qna_rows = load_qna(args.qna_path)
    model_tag = args.model_path.name
    out_dir = args.output_root / model_tag / "beaf"
    out_dir.mkdir(parents=True, exist_ok=True)

    question_file = out_dir / "beaf_questions.jsonl"
    predictions_file = out_dir / "predictions.jsonl"
    answers_json = out_dir / "beaf_answers.json"
    metrics_json = out_dir / "beaf_metrics.json"
    eval_txt = out_dir / "beaf_eval.txt"

    if metrics_json.exists() and not args.overwrite:
        print(f"[skip] {model_tag}: {metrics_json} already exists", flush=True)
        return

    build_question_file(qna_rows, question_file)

    gpu_list = [x.strip() for x in args.gpu.split(",") if x.strip()]
    if not gpu_list:
        raise ValueError("At least one GPU must be provided")

    base_cmd = [
        sys.executable,
        "-m",
        "llava.eval.model_vqa_loader",
        "--model-path",
        str(args.model_path),
        "--image-folder",
        str(args.image_folder),
        "--question-file",
        str(question_file),
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

    pred_files: List[Path] = []
    if len(gpu_list) == 1:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_list[0]
        env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
        env["PYTHONPATH"] = f"{LLAVA_CODE_ROOT}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(LLAVA_CODE_ROOT)
        env.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
        cmd = base_cmd + ["--answers-file", str(predictions_file)]
        print(f"[run] {model_tag}", flush=True)
        subprocess.run(cmd, cwd=LLAVA_CODE_ROOT, env=env, check=True)
        pred_files = [predictions_file]
    else:
        procs = []
        cmds: List[List[str]] = []
        for chunk_idx, gpu_id in enumerate(gpu_list):
            chunk_answers = out_dir / f"predictions_chunk{chunk_idx}.jsonl"
            pred_files.append(chunk_answers)
            cmd = base_cmd + [
                "--answers-file",
                str(chunk_answers),
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
            print(f"[run] {model_tag} chunk={chunk_idx} gpu={gpu_id}", flush=True)
            procs.append(subprocess.Popen(cmd, cwd=LLAVA_CODE_ROOT, env=env))
            cmds.append(cmd)

        for idx, proc in enumerate(procs):
            ret = proc.wait()
            if ret != 0:
                raise subprocess.CalledProcessError(ret, cmds[idx])

    pred_map = merge_prediction_files(pred_files)
    merged_predictions: List[dict] = []
    beaf_answers: List[dict] = []
    missing = []
    for item in qna_rows:
        qid = int(item["id"])
        pred_row = pred_map.get(qid)
        if pred_row is None:
            missing.append(qid)
            continue
        merged_predictions.append(pred_row)
        beaf_answers.append({"id": qid, "answer": pred_row.get("text", "")})

    if missing:
        raise RuntimeError(
            f"Missing predictions for {len(missing)} questions. "
            f"First few question_ids: {missing[:10]}"
        )

    merged_predictions.sort(key=lambda x: int(x["question_id"]))
    beaf_answers.sort(key=lambda x: int(x["id"]))

    with predictions_file.open("w", encoding="utf-8") as f:
        for row in merged_predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with answers_json.open("w", encoding="utf-8") as f:
        json.dump(beaf_answers, f, ensure_ascii=False, indent=2)
        f.write("\n")

    metrics = compute_metrics(qna_rows, beaf_answers)
    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
        f.write("\n")
    write_eval_text(metrics, eval_txt)

    cleanup_targets = [question_file]
    cleanup_targets.extend(pred_files)
    cleanup_intermediate_files(cleanup_targets)

    print(f"[done] {model_tag}: {metrics_json}", flush=True)


if __name__ == "__main__":
    main()
