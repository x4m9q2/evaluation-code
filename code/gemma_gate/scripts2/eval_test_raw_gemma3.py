#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import io
import contextlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = ROOT_DIR.parents[1]
GEMMA_DIR = ROOT_DIR / "gemma"
for extra_path in (ROOT_DIR, GEMMA_DIR, GEMMA_DIR / "src"):
    extra_path_str = str(extra_path)
    if extra_path_str not in sys.path:
        sys.path.insert(0, extra_path_str)

from transformers import AutoModel, AutoProcessor, AutoTokenizer, Gemma3ForConditionalGeneration
from transformers.modeling_utils import load_sharded_checkpoint
from safetensors import safe_open

from src.gate_model.build_gate_model import DualInputGate
from src.train.monkey_patch_forward import replace_gemma3_forward


DEFAULT_MODEL_PATH = BUNDLE_ROOT / "models/Gemma-3-4B-IT"
DEFAULT_DATA_PATH = BUNDLE_ROOT / "data/eval/test_raw_with_shortcut_answer.json"
DEFAULT_IMAGE_FOLDER = BUNDLE_ROOT / "data/images/coco/train2014"
DEFAULT_OUTPUT_ROOT = BUNDLE_ROOT / "outputs/infer_result"
DEFAULT_GATE_TEXT_MODEL = BUNDLE_ROOT / "models/siglip-so400m-patch14-384"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Gemma 3 multimodal inference on test_raw_with_shortcut_answer.json "
            "and optionally compute accuracy + shortcut rate with xVerify."
        )
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--image-folder", type=Path, default=DEFAULT_IMAGE_FOLDER)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--gate-text-model-id", type=Path, default=DEFAULT_GATE_TEXT_MODEL)
    parser.add_argument("--gate-text-max-length", type=int, default=64)
    parser.add_argument("--no-short-answer-prompt", action="store_true")
    return parser.parse_args()


def ensure_exists(path: Path, kind: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{kind} not found: {path}")


def load_json_array(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return rows


def batched(items: List[dict], batch_size: int) -> Iterable[List[dict]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def image_path_from_row(row: dict, image_folder: Path) -> Path:
    image_id = int(row["image_id"])
    return image_folder / f"COCO_train2014_{image_id:012d}.jpg"


def get_model_tag(model_path: Path) -> str:
    return model_path.name


def get_paths(args: argparse.Namespace, model_tag: str) -> tuple[Path, Path, Path, Path]:
    out_dir = args.output_root / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.data_path.stem
    if args.num_chunks > 1:
        stem = f"{stem}.chunk{args.chunk_idx}of{args.num_chunks}"
    merged_file = out_dir / f"{stem}{args.data_path.suffix}"
    stats_file = out_dir / f"{stem}.generation_stats.json"
    tmp_file = out_dir / f"{stem}.tmp.jsonl"
    return out_dir, merged_file, stats_file, tmp_file


def write_model_info_template(out_dir: Path, model_path: Path) -> None:
    info_path = out_dir / "model_info.txt"
    if info_path.exists():
        return

    lines = [
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"模型路径：{model_path}",
        "",
        "请在此处人工补充模型训练信息，只记录模型如何训练得到以及常规训练参数。",
        "不要在该文件中记录推理数据集、推理命令、输出文件名、GPU 分配等运行期信息。",
    ]
    info_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_tmp_predictions(tmp_path: Path) -> Dict[int, dict]:
    pred_map: Dict[int, dict] = {}
    if not tmp_path.exists():
        return pred_map
    with tmp_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid temp JSONL at {tmp_path}:{line_no}") from exc
            pred_map[int(row["question_id"])] = row
    return pred_map


def save_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def append_tmp_rows(tmp_path: Path, rows: List[dict]) -> None:
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def filter_model_load_log(text: str) -> str:
    filtered_lines: List[str] = []
    skip_prefixes = (
        "Some weights of the model checkpoint at ",
        "- This IS expected if you are initializing",
        "- This IS NOT expected if you are initializing",
    )
    for line in text.splitlines():
        if any(line.startswith(prefix) for prefix in skip_prefixes):
            continue
        filtered_lines.append(line)
    return "\n".join(filtered_lines).strip()


def merge_predictions(src_rows: List[dict], pred_map: Dict[int, dict]) -> List[dict]:
    merged = []
    missing = []
    for item in src_rows:
        qid = int(item["question_id"])
        pred = pred_map.get(qid)
        if pred is None:
            missing.append(qid)
            continue
        new_item = dict(item)
        new_item["model_pred"] = pred.get("model_pred", "")
        new_item["model_pred_num_output_tokens"] = int(pred.get("model_pred_num_output_tokens", 0))
        new_item["model_pred_hit_max_tokens"] = bool(pred.get("model_pred_hit_max_tokens", False))
        merged.append(new_item)
    if missing:
        raise RuntimeError(f"Missing predictions for {len(missing)} questions. First few: {missing[:10]}")
    return merged


def build_messages(question: str, image: Image.Image, use_short_answer_prompt: bool = True) -> List[dict]:
    messages: List[dict] = []
    if use_short_answer_prompt:
        messages.append(
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Answer the visual question with only the final short answer. "
                            "Do not provide reasoning or explanation. "
                            "For yes/no questions, reply with exactly 'yes' or 'no'."
                        ),
                    }
                ],
            }
        )
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    )
    return messages


def normalize_prediction(text: str) -> str:
    return " ".join(text.strip().split())


def build_gate_text(question: str) -> str:
    return " ".join(str(question).strip().split())


def configure_dual_input_gate(
    model: Gemma3ForConditionalGeneration,
    gate_text_model_id: Path,
    torch_dtype: torch.dtype,
) -> AutoTokenizer:
    siglip_model = AutoModel.from_pretrained(
        str(gate_text_model_id),
        torch_dtype=torch_dtype,
        trust_remote_code=False,
    )
    text_model = siglip_model.text_model
    gate = DualInputGate(model.config.vision_config.hidden_size, text_model.config.hidden_size)

    model.siglip_text_model = text_model
    model.gate = gate
    model.config.use_dual_input_gate = True

    device = next(model.parameters()).device
    model.siglip_text_model.to(device=device, dtype=torch_dtype)
    model.gate.to(device=device, dtype=torch_dtype)
    del siglip_model

    return AutoTokenizer.from_pretrained(str(gate_text_model_id))


def get_state_tensor_device(model: Gemma3ForConditionalGeneration, name: str) -> torch.Tensor:
    obj = model
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


def resolve_checkpoint_tensor(model_path: Path, tensor_name: str) -> torch.Tensor:
    for shard_path in sorted(model_path.glob("model-*.safetensors")):
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            if tensor_name in f.keys():
                return f.get_tensor(tensor_name)
    raise KeyError(f"Tensor not found in checkpoint shards: {tensor_name}")


def validate_gate_checkpoint_loaded(model: Gemma3ForConditionalGeneration, model_path: Path) -> None:
    check_names = [
        "gate.fc1.weight",
        "gate.fc2.bias",
        "siglip_text_model.embeddings.token_embedding.weight",
        "siglip_text_model.final_layer_norm.weight",
    ]
    max_diff = 0.0
    for name in check_names:
        model_tensor = get_state_tensor_device(model, name).detach()
        ckpt_tensor = resolve_checkpoint_tensor(model_path, name).to(
            device=model_tensor.device,
            dtype=model_tensor.dtype,
        )
        diff = float((model_tensor - ckpt_tensor).abs().max().item())
        max_diff = max(max_diff, diff)
        print(f"[gate-load-check] {name} max_abs_diff={diff:.6g}", flush=True)
    print(f"[gate-load-check] overall_max_abs_diff={max_diff:.6g}", flush=True)
    if max_diff != 0.0:
        raise RuntimeError(f"Gate/SigLIP checkpoint validation failed: max_abs_diff={max_diff}")


def load_model_and_processors(
    model_path: Path,
    gate_text_model_id: Path,
) -> tuple[AutoProcessor, Gemma3ForConditionalGeneration, Optional[AutoTokenizer]]:
    replace_gemma3_forward(use_liger=False)
    Gemma3ForConditionalGeneration._keys_to_ignore_on_load_unexpected = [
        r"^gate\.",
        r"^siglip_text_model\.",
    ]

    processor = AutoProcessor.from_pretrained(model_path)
    load_log = io.StringIO()
    with contextlib.redirect_stdout(load_log), contextlib.redirect_stderr(load_log):
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
    load_log_text = filter_model_load_log(load_log.getvalue())
    if load_log_text.strip():
        print(load_log_text, end="" if load_log_text.endswith("\n") else "\n", flush=True)

    gate_text_tokenizer: Optional[AutoTokenizer] = None
    if bool(getattr(model.config, "use_dual_input_gate", False)):
        print(
            "[gate-load] base Gemma3 weights loaded first; gate and SigLIP text weights will now be attached and restored from checkpoint shards.",
            flush=True,
        )
        gate_text_tokenizer = configure_dual_input_gate(
            model,
            gate_text_model_id=gate_text_model_id,
            torch_dtype=torch.bfloat16,
        )
        load_sharded_checkpoint(model, str(model_path), strict=False, prefer_safe=True)
        validate_gate_checkpoint_loaded(model, model_path)

    model.eval()
    return processor, model, gate_text_tokenizer


def main() -> None:
    args = parse_args()
    ensure_exists(args.model_path, "model path")
    ensure_exists(args.data_path, "data path")
    ensure_exists(args.image_folder, "image folder")

    if args.num_chunks < 1:
        raise ValueError("--num-chunks must be >= 1")
    if args.chunk_idx < 0 or args.chunk_idx >= args.num_chunks:
        raise ValueError("--chunk-idx must satisfy 0 <= chunk-idx < num-chunks")

    rows = load_json_array(args.data_path)
    if args.limit is not None:
        rows = rows[:args.limit]
    if args.num_chunks > 1:
        rows = rows[args.chunk_idx::args.num_chunks]

    model_tag = get_model_tag(args.model_path)
    out_dir, merged_file, stats_file, tmp_file = get_paths(args, model_tag)
    write_model_info_template(out_dir, args.model_path)

    if merged_file.exists() and not args.overwrite:
        print(f"[skip] merged output already exists: {merged_file}", flush=True)
        print(merged_file)
        return

    if args.overwrite:
        for path in (merged_file, stats_file, tmp_file):
            if path.exists():
                path.unlink()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    processor, model, gate_text_tokenizer = load_model_and_processors(
        model_path=args.model_path,
        gate_text_model_id=args.gate_text_model_id,
    )

    pred_map = load_tmp_predictions(tmp_file)
    pending_rows = [row for row in rows if int(row["question_id"]) not in pred_map]

    total_num_output_tokens = sum(int(v.get("model_pred_num_output_tokens", 0)) for v in pred_map.values())
    total_hit_max_tokens = sum(1 for v in pred_map.values() if bool(v.get("model_pred_hit_max_tokens", False)))
    started_at = time.time()

    for batch_idx, batch_rows in enumerate(batched(pending_rows, args.batch_size), start=1):
        images = []
        prompts = []
        qids = []
        for row in batch_rows:
            image = Image.open(image_path_from_row(row, args.image_folder)).convert("RGB")
            messages = build_messages(
                row["question"],
                image,
                use_short_answer_prompt=not args.no_short_answer_prompt,
            )
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images.append(image)
            prompts.append(prompt)
            qids.append(int(row["question_id"]))

        batched_images = [[image] for image in images]
        inputs = processor(text=prompts, images=batched_images, return_tensors="pt", padding=True)
        if gate_text_tokenizer is not None:
            gate_texts = [build_gate_text(row["question"]) for row in batch_rows]
            gate_tokens = gate_text_tokenizer(
                gate_texts,
                add_special_tokens=True,
                truncation=True,
                max_length=args.gate_text_max_length,
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            inputs["gate_input_ids"] = gate_tokens["input_ids"]
            inputs["gate_attention_mask"] = gate_tokens.get(
                "attention_mask", torch.ones_like(gate_tokens["input_ids"])
            )
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
            )

        input_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[:, input_len:]
        texts = processor.batch_decode(new_tokens, skip_special_tokens=True)

        tmp_rows = []
        for qid, text, token_ids in zip(qids, texts, new_tokens):
            token_count = int((token_ids != processor.tokenizer.pad_token_id).sum().item())
            hit_max = token_count >= args.max_new_tokens
            record = {
                "question_id": qid,
                "model_pred": normalize_prediction(text),
                "model_pred_num_output_tokens": token_count,
                "model_pred_hit_max_tokens": hit_max,
            }
            tmp_rows.append(record)
            pred_map[qid] = record
            total_num_output_tokens += token_count
            total_hit_max_tokens += int(hit_max)

        append_tmp_rows(tmp_file, tmp_rows)

        done = len(pred_map)
        elapsed = time.time() - started_at
        avg_sec = elapsed / max(batch_idx, 1)
        print(
            f"[batch {batch_idx}] done={done}/{len(rows)} "
            f"avg_batch_sec={avg_sec:.2f}",
            flush=True,
        )

    merged = merge_predictions(rows, pred_map)
    save_json(merged, merged_file)

    stats = {
        "model_path": str(args.model_path),
        "data_path": str(args.data_path),
        "num_rows": len(rows),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "total_num_output_tokens": total_num_output_tokens,
        "avg_num_output_tokens": total_num_output_tokens / len(rows) if rows else 0.0,
        "num_hit_max_tokens": total_hit_max_tokens,
        "pct_hit_max_tokens": total_hit_max_tokens / len(rows) if rows else 0.0,
    }
    save_json(stats, stats_file)
    print(merged_file)


if __name__ == "__main__":
    main()
