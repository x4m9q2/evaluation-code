#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional


SCRIPT_PATH = Path(__file__).resolve()
LLAVA_CODE_ROOT = SCRIPT_PATH.parents[1]
BUNDLE_ROOT = LLAVA_CODE_ROOT.parents[1]

DEFAULT_DATA_PATH = BUNDLE_ROOT / "data/eval/test_raw_with_shortcut_answer.json"
DEFAULT_IMAGE_FOLDER = BUNDLE_ROOT / "data/images/coco/train2014"
DEFAULT_OUTPUT_ROOT = BUNDLE_ROOT / "outputs/llava_infer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run inference for a single model checkpoint and write a JSON output "
            "with an added model_pred field."
        )
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--model-base",
        type=Path,
        default=None,
        help="Optional base model path. Required for projector-only checkpoints such as mm_projector pretrains.",
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        choices=["auto", "vqa", "vqa_v2_cmsv", "gqa", "gqa_cmsv", "vg", "vg_cmsv"],
        default="auto",
        help="Dataset format used to normalize question/answer/image fields.",
    )
    parser.add_argument(
        "--has-gate",
        required=True,
        choices=["auto", "true", "false"],
        help="Whether to force-enable the model's dual-input gate during inference.",
    )
    parser.add_argument("--image-folder", type=Path, default=DEFAULT_IMAGE_FOLDER)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--patch-mask-analysis-path", type=Path, default=None)
    parser.add_argument("--gate-patch-suppress-ratio", type=float, default=0.0)
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=1,
        help="Split inference into this many chunks. Defaults to 1.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
        default="bf16",
        help="Inference dtype passed to model_vqa_loader. Defaults to bf16.",
    )
    parser.add_argument(
        "--begin-suppress-eos",
        action="store_true",
        help="Pass --begin-suppress-eos to model_vqa_loader.",
    )
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing merged JSON output if it already exists.",
    )
    return parser.parse_args()


def ensure_exists(path: Path, kind: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{kind} not found: {path}")


def load_rows(path: Path) -> List[dict]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open() as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected a JSON object at {path}:{line_no}")
                rows.append(row)
        return rows
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return data


def canonical_dataset_name(dataset: str) -> str:
    if dataset in {"vqa", "vqa_v2_cmsv"}:
        return "vqa"
    if dataset in {"gqa", "gqa_cmsv"}:
        return "gqa"
    if dataset in {"vg", "vg_cmsv"}:
        return "vg"
    return dataset


def infer_dataset_from_row(row: dict) -> Optional[str]:
    image_path = str(row.get("image_path") or row.get("image") or "").lower()
    if "masked_images/gqa" in image_path or "masked_results_gqa" in image_path or "/gqa/" in image_path:
        return "gqa"
    if "masked_images/vg" in image_path or "masked_results_vg" in image_path or "/vg/" in image_path or "visual_genome" in image_path:
        return "vg"
    if row.get("generated_question") is None and row.get("question") is not None:
        return "vqa"
    return None


def resolve_dataset(dataset: str, row: dict) -> str:
    dataset = canonical_dataset_name(dataset)
    if dataset != "auto":
        return dataset
    inferred = infer_dataset_from_row(row)
    if inferred is None:
        raise ValueError(
            "Could not infer dataset for row. Set --dataset to vqa, gqa, or vg. "
            f"question_id={row.get('question_id')}"
        )
    return inferred


def first_nonempty(row: dict, keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return str(value)
    return ""


def find_vg_image(image_id: int, image_folder: Path) -> str:
    candidates = [
        f"{image_id}.jpg",
        f"VG_100K/{image_id}.jpg",
        f"VG_100K_2/{image_id}.jpg",
    ]
    for candidate in candidates:
        if (image_folder / candidate).exists():
            return candidate
    return f"VG_100K/{image_id}.jpg"


def infer_image_name(row: dict, dataset: str, image_folder: Path) -> str:
    image_value = first_nonempty(row, ["image"])
    if image_value:
        return image_value
    image_id = int(row["image_id"])
    if dataset == "vqa":
        return f"COCO_train2014_{image_id:012d}.jpg"
    if dataset == "gqa":
        return f"{image_id}.jpg"
    if dataset == "vg":
        return find_vg_image(image_id, image_folder)
    raise ValueError(f"Unsupported dataset for image resolution: {dataset}")


def normalize_eval_row(row: dict, dataset_arg: str, image_folder: Path) -> dict:
    dataset = resolve_dataset(dataset_arg, row)
    out = dict(row)
    question = first_nonempty(row, ["question", "generated_question", "text"])
    answer = first_nonempty(row, ["answer", "generated_answer"])
    shortcut_answer = first_nonempty(row, ["shortcut_answer", "original_answer"])
    if not question:
        raise ValueError(f"Missing question/generated_question for question_id={row.get('question_id')}")
    if not answer:
        raise ValueError(f"Missing answer/generated_answer for question_id={row.get('question_id')}")
    if not shortcut_answer:
        raise ValueError(f"Missing shortcut_answer/original_answer for question_id={row.get('question_id')}")
    out["question"] = question
    out["answer"] = answer
    out["shortcut_answer"] = shortcut_answer
    out["image"] = infer_image_name(row, dataset, image_folder)
    out["eval_dataset"] = dataset
    return out


def normalize_eval_rows(rows: Iterable[dict], dataset_arg: str, image_folder: Path) -> List[dict]:
    return [normalize_eval_row(row, dataset_arg, image_folder) for row in rows]


def build_eval_questions(src_rows: Iterable[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for item in src_rows:
            rec = {
                "question_id": int(item["question_id"]),
                "image": item["image"],
                "text": item["question"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_predictions(path: Path) -> Dict[int, str]:
    preds: Dict[int, str] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            preds[int(row["question_id"])] = row.get("text", "")
    return preds


def merge_prediction_files(paths: List[Path]) -> Dict[int, str]:
    merged: Dict[int, str] = {}
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


def merge_predictions(src_rows: List[dict], pred_map: Dict[int, str]) -> List[dict]:
    merged = []
    missing = []
    for item in src_rows:
        qid = int(item["question_id"])
        new_item = dict(item)
        if qid not in pred_map:
            missing.append(qid)
            new_item["model_pred"] = None
        else:
            new_item["model_pred"] = pred_map[qid]
        merged.append(new_item)
    if missing:
        raise RuntimeError(
            f"Missing predictions for {len(missing)} questions. "
            f"First few question_ids: {missing[:10]}"
        )
    return merged


def write_model_info(
    out_dir: Path,
    model_dir: Path,
    args: argparse.Namespace,
) -> None:
    info_path = out_dir / "model_info.txt"
    if info_path.exists():
        return
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"生成时间：{now_str}",
        f"模型路径：{model_dir}",
        "",
        "请在此处人工补充模型训练信息，只记录模型如何训练得到以及常规训练参数。",
        "不要在该文件中记录推理数据集、推理命令、输出文件名、GPU 分配等运行期信息。",
        "训练背景信息应来自模型目录中的 README、训练日志、配置或人工确认结果，不要由脚本自动猜测。",
    ]
    info_path.write_text("\n".join(lines) + "\n")


def cleanup_intermediate_files(paths: List[Path]) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def run_one_model(model_dir: Path, src_rows: List[dict], args: argparse.Namespace) -> None:
    model_tag = model_dir.name
    out_dir = args.output_root / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    question_file = out_dir / "questions.jsonl"
    answers_file = out_dir / "predictions.jsonl"
    merged_file = out_dir / args.data_path.name

    if merged_file.exists() and not args.overwrite:
        print(f"[skip] {model_tag}: {merged_file} already exists", flush=True)
        return

    build_eval_questions(src_rows, question_file)
    gpu_list = [x.strip() for x in args.gpu.split(",") if x.strip()]
    if args.num_chunks < 1:
        raise ValueError("--num-chunks must be >= 1")
    if args.num_chunks > 1 and len(gpu_list) != args.num_chunks:
        raise ValueError("When --num-chunks > 1, the number of GPUs must equal --num-chunks")

    base_cmd = [
        sys.executable,
        "-m",
        "llava.eval.model_vqa_loader",
        "--model-path",
        str(model_dir),
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
        "--torch-dtype",
        args.torch_dtype,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--force-use-dual-input-gate",
        args.has_gate,
    ]
    if args.begin_suppress_eos:
        base_cmd.append("--begin-suppress-eos")
    if args.model_base is not None:
        base_cmd.extend(["--model-base", str(args.model_base)])
    if args.patch_mask_analysis_path is not None:
        base_cmd.extend(["--patch-mask-analysis-path", str(args.patch_mask_analysis_path)])
    if args.gate_patch_suppress_ratio != 0:
        base_cmd.extend(["--gate-patch-suppress-ratio", str(args.gate_patch_suppress_ratio)])

    pred_files: List[Path] = []
    if args.num_chunks == 1:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_list[0]
        env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
        env["PYTHONPATH"] = f"{LLAVA_CODE_ROOT}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(LLAVA_CODE_ROOT)
        env.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
        cmd = base_cmd + ["--answers-file", str(answers_file)]
        print(f"[run] {model_tag}", flush=True)
        subprocess.run(cmd, cwd=LLAVA_CODE_ROOT, env=env, check=True)
        pred_files = [answers_file]
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
                str(args.num_chunks),
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

        with answers_file.open("w") as fout:
            for pred_file in pred_files:
                with pred_file.open() as fin:
                    for line in fin:
                        if line.strip():
                            fout.write(line)

    pred_map = merge_prediction_files(pred_files)
    merged = merge_predictions(src_rows, pred_map)
    with merged_file.open("w") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
        f.write("\n")

    info_cmd = base_cmd + ["--answers-file", str(answers_file)]
    if args.num_chunks > 1:
        info_cmd += ["--num-chunks", str(args.num_chunks)]
    write_model_info(out_dir, model_dir, args)

    cleanup_targets = [question_file, answers_file]
    cleanup_targets.extend(pred_files)
    cleanup_intermediate_files(cleanup_targets)

    print(f"[done] {model_tag}: {merged_file}", flush=True)


def main() -> None:
    args = parse_args()
    ensure_exists(args.model_path, "model path")
    if args.model_base is not None:
        ensure_exists(args.model_base, "model base path")
    ensure_exists(args.data_path, "data file")
    ensure_exists(args.image_folder, "image folder")
    args.output_root.mkdir(parents=True, exist_ok=True)
    if not (args.model_path / "config.json").exists():
        raise FileNotFoundError(f"config.json not found under model path: {args.model_path}")

    src_rows = normalize_eval_rows(load_rows(args.data_path), args.dataset, args.image_folder)
    if args.limit is not None:
        src_rows = src_rows[:args.limit]
    run_one_model(args.model_path, src_rows, args)


if __name__ == "__main__":
    main()
