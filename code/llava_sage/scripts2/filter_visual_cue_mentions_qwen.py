#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_BUNDLE_ROOT = Path(__file__).resolve().parents[3]


PROMPT_TEMPLATE = """You are filtering VQA samples.

Task:
Decide whether the question text explicitly mentions, directly refers to, or clearly describes the same object category as any visual cue.

Mark `mentioned=true` for a cue if the question:
- directly names that object category,
- uses a near-synonym,
- uses an obvious subtype/supertype that clearly points to the same object kind,
- or clearly refers to that object with a descriptive noun phrase.

Mark `mentioned=false` if the cue is only background context, only weakly implied, or not clearly referred to.

Be strict, but if the question text literally contains the cue word or a clear same-category synonym, treat that as mentioned even when it is used as a spatial reference.
Only keep `mentioned=false` for clearly different compound terms such as `toilet paper`, `train tracks`, `bus stop`, or `clock tower`.

Question: {question}
Visual cues: {visual_cues_json}

Return JSON only in this schema:
{{
  "remove": true or false,
  "cue_results": [
    {{"cue": "cue text", "mentioned": true or false, "reason": "short reason"}}
  ],
  "summary": "one short sentence"
}}
"""


STRICT_ALIAS_MAP: Dict[str, List[str]] = {
    "person": [
        "people",
        "man",
        "woman",
        "boy",
        "girl",
        "guy",
        "lady",
        "child",
        "children",
        "kid",
        "kids",
        "player",
        "players",
        "surfer",
        "surfers",
        "skier",
        "skiers",
        "skateboarder",
        "skateboarders",
        "rider",
        "riders",
        "driver",
        "drivers",
        "batter",
        "batters",
        "catcher",
        "catchers",
        "pitcher",
        "pitchers",
        "handler",
        "handlers",
    ],
    "tv": ["television", "screen", "monitor"],
    "airplane": ["plane", "planes", "jet", "jets", "aircraft"],
    "tennis racket": ["racket", "rackets", "racquet", "racquets"],
    "wine glass": ["glass", "glasses", "goblet", "goblets"],
    "sports ball": [
        "ball",
        "balls",
        "soccer ball",
        "tennis ball",
        "baseball",
        "basketball",
        "football",
    ],
    "motorcycle": ["motorbike", "motorbikes", "bike", "bikes"],
    "bicycle": ["bike", "bikes"],
    "car": ["truck", "trucks", "van", "vans", "pickup", "pickups", "suv", "suvs", "jeep", "jeeps", "vehicle", "vehicles"],
    "dining table": ["table", "tables", "counter", "counters", "desk", "desks"],
    "couch": ["sofa", "sofas"],
    "cell phone": ["phone", "phones", "cellphone", "cellphones", "mobile phone", "mobile phones", "smartphone", "smartphones"],
    "potted plant": ["plant", "plants"],
}


STRICT_EXCLUSION_PATTERNS: Dict[str, List[str]] = {
    "toilet": [
        r"\btoilet paper\b",
        r"\btoilet paper holder\b",
        r"\btoilet brush\b",
        r"\btoilet brush holder\b",
        r"\btoilet seat\b",
        r"\btoilet lid\b",
    ],
    "train": [
        r"\btrain track\b",
        r"\btrain tracks\b",
        r"\btrain station\b",
        r"\btrain platform\b",
    ],
    "bus": [
        r"\bbus stop\b",
        r"\bbus shelter\b",
    ],
    "clock": [
        r"\bclock tower\b",
    ],
    "pizza": [
        r"\bpizza cutter\b",
        r"\bpizza sauce\b",
        r"\bpizza box\b",
        r"\bpizza-box\b",
    ],
    "orange": [
        r"\borange juice\b",
        r"\borange pumpkin\b",
        r"\borange heap\b",
        r"\borange fruit\b",
        r"\borange fruit chunks\b",
    ],
    "dog": [
        r"\bdog leash\b",
        r"\bdog handler\b",
    ],
    "cat": [
        r"\bcat picture\b",
    ],
    "banana": [
        r"\bbanana leaf\b",
    ],
    "mouse": [
        r"\bmouse pad\b",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter rows whose visual cue is mentioned in the question using local Qwen.")
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT / "models/Qwen3.5-9B",
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_rows(path: Path, limit: int | None) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["rows"] if isinstance(data, dict) and "rows" in data else data
    if limit is not None:
        rows = rows[:limit]
    return rows


def batched(items: List[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def make_prompt(row: Dict[str, Any]) -> str:
    return PROMPT_TEMPLATE.format(
        question=row["text"],
        visual_cues_json=json.dumps(row["visual_cues"], ensure_ascii=False),
    )


def contains_term(text: str, term: str) -> bool:
    return re.search(r"(?<![a-z])" + re.escape(term.lower()) + r"(?![a-z])", text.lower()) is not None


def matches_exclusion(text: str, cue: str) -> bool:
    for pattern in STRICT_EXCLUSION_PATTERNS.get(cue.lower(), []):
        if re.search(pattern, text.lower()):
            return True
    return False


def strict_lexical_match(question: str, cue: str) -> str | None:
    question_lower = question.lower()
    cue_lower = cue.lower()

    if matches_exclusion(question_lower, cue_lower):
        return None

    if contains_term(question_lower, cue_lower):
        return f"strict_literal_match:{cue_lower}"

    for alias in STRICT_ALIAS_MAP.get(cue_lower, []):
        if contains_term(question_lower, alias):
            return f"strict_alias_match:{alias}"

    return None


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group(0))
        raise


def normalize_result(row: Dict[str, Any], raw_text: str, parsed: Dict[str, Any]) -> Dict[str, Any]:
    cue_results = parsed.get("cue_results", [])
    cue_map: Dict[str, Dict[str, Any]] = {}
    for item in cue_results:
        cue = str(item.get("cue", "")).strip()
        if cue:
            cue_map[cue] = {
                "cue": cue,
                "mentioned": bool(item.get("mentioned", False)),
                "reason": str(item.get("reason", "")).strip(),
            }

    normalized_cues = []
    for cue in row["visual_cues"]:
        matched_reason = strict_lexical_match(row["text"], cue)
        default_item = {
            "cue": cue,
            "mentioned": False,
            "reason": "missing_from_model_output",
        }
        item = cue_map.get(cue, default_item)
        if matched_reason:
            item = {
                "cue": cue,
                "mentioned": True,
                "reason": matched_reason,
            }
        normalized_cues.append(
            item
        )

    remove = any(item["mentioned"] for item in normalized_cues)
    return {
        "question_id": row["question_id"],
        "image_id": row["image_id"],
        "text": row["text"],
        "visual_cues": row["visual_cues"],
        "mask_path": row.get("mask_path"),
        "remove": remove,
        "cue_results": normalized_cues,
        "summary": str(parsed.get("summary", "")).strip(),
        "raw_model_output": raw_text,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_jsonl = args.output_dir / "audit.jsonl"
    keep_json = args.output_dir / "keep.json"
    remove_json = args.output_dir / "remove.json"
    summary_json = args.output_dir / "summary.json"

    if args.overwrite:
        for path in [audit_jsonl, keep_json, remove_json, summary_json]:
            if path.exists():
                path.unlink()

    rows = load_rows(args.input_json, args.limit)
    done_qids = set()
    if audit_jsonl.exists():
        with audit_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                done_qids.add(int(json.loads(line)["question_id"]))

    pending = [row for row in rows if int(row["question_id"]) not in done_qids]

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(args.model_path),
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.batch_size,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    tokenizer = llm.get_tokenizer()

    if pending:
        with audit_jsonl.open("a", encoding="utf-8") as fout:
            processed = 0
            for batch in batched(pending, args.batch_size):
                prompts = []
                for row in batch:
                    messages = [{"role": "user", "content": make_prompt(row)}]
                    prompts.append(
                        tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                    )

                outputs = llm.generate(prompts, sampling_params)
                for row, out in zip(batch, outputs):
                    raw_text = out.outputs[0].text if out.outputs else ""
                    try:
                        parsed = extract_json(raw_text)
                    except Exception as exc:
                        parsed = {
                            "remove": False,
                            "cue_results": [
                                {
                                    "cue": cue,
                                    "mentioned": False,
                                    "reason": f"parse_error:{type(exc).__name__}",
                                }
                                for cue in row["visual_cues"]
                            ],
                            "summary": "parse_error",
                        }
                    record = normalize_result(row, raw_text, parsed)
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    processed += 1
                fout.flush()
                print(f"processed {processed}/{len(pending)} pending rows", flush=True)

    all_records: List[Dict[str, Any]] = []
    with audit_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_records.append(json.loads(line))

    all_records.sort(key=lambda x: int(x["question_id"]))
    keep_rows = [x for x in all_records if not x["remove"]]
    remove_rows = [x for x in all_records if x["remove"]]

    keep_json.write_text(json.dumps(keep_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    remove_json.write_text(json.dumps(remove_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    cue_total = sum(len(x["visual_cues"]) for x in all_records)
    cue_mentioned = sum(
        1 for x in all_records for c in x["cue_results"] if bool(c["mentioned"])
    )
    summary = {
        "input_json": str(args.input_json),
        "total_rows": len(all_records),
        "kept_rows": len(keep_rows),
        "removed_rows": len(remove_rows),
        "removed_ratio": len(remove_rows) / len(all_records) if all_records else 0.0,
        "total_cues": cue_total,
        "mentioned_cues": cue_mentioned,
        "model_path": str(args.model_path),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
