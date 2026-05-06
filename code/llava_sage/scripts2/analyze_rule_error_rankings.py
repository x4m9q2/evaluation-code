#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_MAPPING_PATH = Path("/path/to/sage_repro_bundle/merged_output_rule_mapping.json")
DEFAULT_XVERIFY_ROOT = Path("/path/to/sage_repro_bundle/x_verify")
DEFAULT_XVERIFY_MODEL = Path("/path/to/sage_repro_bundle/x_verify/xVerify-0.5B-I")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze wrong predictions by matched shortcut rule and report whether "
            "shortcut_answer is the top error for each rule."
        )
    )
    parser.add_argument(
        "--pred-path",
        type=Path,
        required=True,
        help="Merged inference JSON containing question_id, answer, answer_type, shortcut_answer and model_pred.",
    )
    parser.add_argument(
        "--mapping-path",
        type=Path,
        default=DEFAULT_MAPPING_PATH,
        help="Path to merged_output_rule_mapping.json.",
    )
    parser.add_argument(
        "--answer-types",
        default="number,other",
        help="Comma-separated answer types to analyze. Default: number,other",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output report path. Defaults to <pred_stem>.rule_error_rankings.json in the same directory.",
    )
    parser.add_argument(
        "--xverify-root",
        type=Path,
        default=DEFAULT_XVERIFY_ROOT,
        help="Path to the xVerify project root.",
    )
    parser.add_argument(
        "--xverify-model-path",
        type=Path,
        default=DEFAULT_XVERIFY_MODEL,
        help="Path to the local xVerify model directory.",
    )
    parser.add_argument(
        "--xverify-gpu",
        default="0",
        help="CUDA_VISIBLE_DEVICES value used when running xVerify.",
    )
    parser.add_argument(
        "--xverify-batch-size",
        type=int,
        default=32,
        help="Batch size for xVerify.",
    )
    parser.add_argument(
        "--accuracy-xverify-result",
        type=Path,
        default=None,
        help="Optional precomputed xVerify result JSON for standard-answer accuracy.",
    )
    parser.add_argument(
        "--shortcut-xverify-result",
        type=Path,
        default=None,
        help="Optional precomputed xVerify result JSON for shortcut-answer matching.",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def save_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def normalize_text(text: object) -> str:
    value = str(text or "").strip().lower()
    value = re.sub(r"^[\s\.,;:!?]+|[\s\.,;:!?]+$", "", value)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    value = " ".join(value.split())
    return value


def dense_rank_from_count_list(counts: List[int], target_count: int) -> int | None:
    if target_count <= 0:
        return None
    last_count = None
    rank = 0
    for count in sorted(counts, reverse=True):
        if count != last_count:
            rank += 1
            last_count = count
        if count == target_count:
            return rank
    return None


def build_default_output_path(pred_path: Path) -> Path:
    return pred_path.parent / f"{pred_path.stem}.rule_error_rankings.json"


def build_xverify_input(rows: List[dict], answer_key: str) -> List[dict]:
    output = []
    for row in rows:
        output.append(
            {
                "question_id": row.get("question_id"),
                "question": row.get("question", ""),
                "llm_output": row.get("model_pred", ""),
                "correct_answer": row.get(answer_key, ""),
                "answer_type": row.get("answer_type", ""),
                "answer": row.get("answer", ""),
                "shortcut_answer": row.get("shortcut_answer", ""),
                "model_pred": row.get("model_pred", ""),
            }
        )
    return output


def run_xverify(
    xverify_root: Path,
    xverify_model_path: Path,
    input_path: Path,
    output_dir: Path,
    gpu: str,
    batch_size: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = f"{xverify_root}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(xverify_root)
    cmd = [
        sys.executable,
        "run_local_xverify.py",
        "--data-path",
        str(input_path),
        "--output-path",
        str(output_dir),
        "--model-path",
        str(xverify_model_path),
        "--batch-size",
        str(batch_size),
    ]
    subprocess.run(cmd, cwd=xverify_root, env=env, check=True)
    result_files = sorted(output_dir.glob("Eval_Judge_*.json"), key=lambda p: p.stat().st_mtime)
    if not result_files:
        raise FileNotFoundError(f"No xVerify result file found under {output_dir}")
    return result_files[-1]


def build_xverify_row_map(result_path: Path, reference_rows: List[dict]) -> Dict[int, dict]:
    data = load_json(result_path)
    result_rows = data.get("results")
    if not isinstance(result_rows, list):
        raise ValueError(f"results not found in {result_path}")

    if result_rows and "question_id" in result_rows[0]:
        return {int(row["question_id"]): row for row in result_rows}

    if len(result_rows) != len(reference_rows):
        raise ValueError(
            f"xVerify result {result_path} does not contain question_id and its row count "
            f"({len(result_rows)}) does not match the reference rows ({len(reference_rows)})."
        )
    return {
        int(ref_row["question_id"]): result_row
        for ref_row, result_row in zip(reference_rows, result_rows)
    }


def is_xverify_valid(row: dict) -> bool:
    return str(row.get("judge_valid", "")).lower() == "true"


def is_xverify_correct(row: dict) -> bool:
    return is_xverify_valid(row) and str(row.get("xVerify-0.5B-I_judgment_result", "")).lower() == "correct"


def build_rule_report(
    rows: Iterable[dict],
    mapping_by_qid: Dict[int, dict],
    answer_type: str,
    accuracy_by_qid: Dict[int, dict],
    shortcut_by_qid: Dict[int, dict],
) -> dict:
    groups: Dict[int, dict] = {}
    valid_count = 0
    invalid_accuracy_judgment_count = 0
    total_wrong = 0
    total_semantic_shortcut_hit = 0
    total_literal_shortcut_hit = 0

    for row in rows:
        if row.get("answer_type") != answer_type:
            continue

        qid = int(row["question_id"])
        accuracy_row = accuracy_by_qid.get(qid)
        shortcut_row = shortcut_by_qid.get(qid)
        if accuracy_row is None:
            raise KeyError(f"question_id={qid} not found in accuracy xVerify result")
        if shortcut_row is None:
            raise KeyError(f"question_id={qid} not found in shortcut xVerify result")

        if not is_xverify_valid(accuracy_row):
            invalid_accuracy_judgment_count += 1
            continue

        valid_count += 1
        if is_xverify_correct(accuracy_row):
            continue

        mapping_row = mapping_by_qid.get(qid)
        if mapping_row is None:
            raise KeyError(f"question_id={qid} not found in mapping file")

        matched_rule = mapping_row["matched_rule"]
        rule_id = int(matched_rule["rule_id"])
        pred = normalize_text(row.get("model_pred"))
        shortcut_answer = normalize_text(row.get("shortcut_answer"))
        semantic_shortcut_hit = is_xverify_correct(shortcut_row)
        literal_shortcut_hit = bool(shortcut_answer) and pred == shortcut_answer

        group = groups.setdefault(
            rule_id,
            {
                "rule_id": rule_id,
                "trigger": matched_rule.get("trigger"),
                "support": matched_rule.get("support"),
                "confidence": matched_rule.get("confidence"),
                "shortcut_answer": shortcut_answer,
                "wrong_count": 0,
                "shortcut_count": 0,
                "literal_shortcut_count": 0,
                "candidate_error_counts": Counter(),
                "sample_question_ids": [],
                "sample_questions": [],
            },
        )
        group["wrong_count"] += 1
        group["shortcut_count"] += int(semantic_shortcut_hit)
        group["literal_shortcut_count"] += int(literal_shortcut_hit)
        group["candidate_error_counts"][pred] += 1
        if len(group["sample_question_ids"]) < 5:
            group["sample_question_ids"].append(qid)
            group["sample_questions"].append(row.get("question"))

        total_wrong += 1
        total_semantic_shortcut_hit += int(semantic_shortcut_hit)
        total_literal_shortcut_hit += int(literal_shortcut_hit)

    per_rule: List[dict] = []
    shortcut_first_rules: List[int] = []
    shortcut_tied_first_rules: List[int] = []
    non_shortcut_first_rules: List[int] = []

    weighted_shortcut_first = 0
    weighted_shortcut_tied_first = 0
    weighted_non_shortcut_first = 0

    for rule_id, group in sorted(groups.items()):
        sorted_candidates = sorted(
            group["candidate_error_counts"].items(),
            key=lambda item: (-item[1], item[0]),
        )
        shortcut_answer = group["shortcut_answer"]
        shortcut_count = group["shortcut_count"]
        literal_shortcut_count = group["literal_shortcut_count"]
        top_count = sorted_candidates[0][1]
        top_answers = [answer for answer, count in sorted_candidates if count == top_count]
        shortcut_rank = dense_rank_from_count_list(
            [count for _, count in sorted_candidates] + ([shortcut_count] if shortcut_count > 0 else []),
            shortcut_count,
        )
        is_shortcut_first = shortcut_count > 0 and shortcut_count >= top_count
        is_shortcut_unique_first = shortcut_count > top_count
        is_shortcut_tied_first = shortcut_count > 0 and shortcut_count == top_count

        if is_shortcut_unique_first:
            shortcut_first_rules.append(rule_id)
            weighted_shortcut_first += group["wrong_count"]
        elif is_shortcut_tied_first:
            shortcut_tied_first_rules.append(rule_id)
            weighted_shortcut_tied_first += group["wrong_count"]
        else:
            non_shortcut_first_rules.append(rule_id)
            weighted_non_shortcut_first += group["wrong_count"]

        per_rule.append(
            {
                "rule_id": rule_id,
                "trigger": group["trigger"],
                "support": group["support"],
                "confidence": group["confidence"],
                "wrong_count": group["wrong_count"],
                "shortcut_answer": shortcut_answer,
                "shortcut_count": shortcut_count,
                "literal_shortcut_count": literal_shortcut_count,
                "shortcut_rate_among_wrong": (shortcut_count / group["wrong_count"]) if group["wrong_count"] else 0.0,
                "literal_shortcut_rate_among_wrong": (literal_shortcut_count / group["wrong_count"]) if group["wrong_count"] else 0.0,
                "shortcut_rank": shortcut_rank,
                "is_shortcut_first": is_shortcut_first,
                "is_shortcut_unique_first": is_shortcut_unique_first,
                "is_shortcut_tied_first": is_shortcut_tied_first,
                "top_literal_answers": top_answers,
                "top_literal_count": top_count,
                "top_answers": top_answers,
                "top_count": top_count,
                "candidate_error_counts": [
                    {"answer": answer, "count": count}
                    for answer, count in sorted_candidates
                ],
                "sample_question_ids": group["sample_question_ids"],
                "sample_questions": group["sample_questions"],
            }
        )

    per_rule.sort(key=lambda item: (-item["wrong_count"], item["rule_id"]))

    return {
        "answer_type": answer_type,
        "valid_sample_count": valid_count,
        "invalid_accuracy_judgment_count": invalid_accuracy_judgment_count,
        "correct_sample_count": valid_count - total_wrong,
        "wrong_sample_count": total_wrong,
        "semantic_shortcut_hit_count_among_wrong": total_semantic_shortcut_hit,
        "semantic_shortcut_rate_among_wrong": (total_semantic_shortcut_hit / total_wrong) if total_wrong else 0.0,
        "literal_shortcut_hit_count_among_wrong": total_literal_shortcut_hit,
        "literal_shortcut_rate_among_wrong": (total_literal_shortcut_hit / total_wrong) if total_wrong else 0.0,
        "rule_count_with_wrong": len(per_rule),
        "shortcut_first_rule_count": len(shortcut_first_rules),
        "shortcut_tied_first_rule_count": len(shortcut_tied_first_rules),
        "non_shortcut_first_rule_count": len(non_shortcut_first_rules),
        "wrong_samples_under_shortcut_first_rules": weighted_shortcut_first,
        "wrong_samples_under_shortcut_tied_first_rules": weighted_shortcut_tied_first,
        "wrong_samples_under_non_shortcut_first_rules": weighted_non_shortcut_first,
        "shortcut_first_rules": shortcut_first_rules,
        "shortcut_tied_first_rules": shortcut_tied_first_rules,
        "non_shortcut_first_rules": non_shortcut_first_rules,
        "per_rule": per_rule,
    }


def main() -> None:
    args = parse_args()
    pred_rows = load_json(args.pred_path)
    mapping_rows = load_json(args.mapping_path)["results"]
    mapping_by_qid = {int(row["question_id"]): row for row in mapping_rows}

    answer_types = [item.strip() for item in args.answer_types.split(",") if item.strip()]
    analysis_rows = [row for row in pred_rows if row.get("answer_type") in answer_types]
    if not analysis_rows:
        raise ValueError(f"No rows found for answer types: {answer_types}")
    output_path = args.output_path or build_default_output_path(args.pred_path)

    with tempfile.TemporaryDirectory(prefix="rule_error_rankings_xverify_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)

        accuracy_result_path = args.accuracy_xverify_result
        accuracy_result_is_temporary = accuracy_result_path is None
        if accuracy_result_path is None:
            accuracy_input = tmp_dir / "accuracy_input.json"
            save_json(build_xverify_input(analysis_rows, "answer"), accuracy_input)
            accuracy_result_path = run_xverify(
                xverify_root=args.xverify_root,
                xverify_model_path=args.xverify_model_path,
                input_path=accuracy_input,
                output_dir=tmp_dir / "accuracy_out",
                gpu=args.xverify_gpu,
                batch_size=args.xverify_batch_size,
            )

        shortcut_result_path = args.shortcut_xverify_result
        shortcut_result_is_temporary = shortcut_result_path is None
        if shortcut_result_path is None:
            shortcut_input = tmp_dir / "shortcut_input.json"
            save_json(build_xverify_input(analysis_rows, "shortcut_answer"), shortcut_input)
            shortcut_result_path = run_xverify(
                xverify_root=args.xverify_root,
                xverify_model_path=args.xverify_model_path,
                input_path=shortcut_input,
                output_dir=tmp_dir / "shortcut_out",
                gpu=args.xverify_gpu,
                batch_size=args.xverify_batch_size,
            )

        accuracy_by_qid = build_xverify_row_map(accuracy_result_path, analysis_rows)
        shortcut_by_qid = build_xverify_row_map(shortcut_result_path, analysis_rows)

    report = {
        "pred_path": str(args.pred_path),
        "mapping_path": str(args.mapping_path),
        "judgment_basis": {
            "wrong_definition": "Rows with valid xVerify accuracy judgment marked Incorrect.",
            "shortcut_definition": "Rows with valid xVerify shortcut judgment marked Correct among the xVerify-defined wrong rows.",
            "candidate_error_counts_definition": "Normalized literal model outputs counted only on xVerify-defined wrong rows.",
        },
        "accuracy_xverify_result_path": None if accuracy_result_is_temporary else str(accuracy_result_path),
        "shortcut_xverify_result_path": None if shortcut_result_is_temporary else str(shortcut_result_path),
        "answer_types": answer_types,
        "reports": {},
    }

    for answer_type in answer_types:
        report["reports"][answer_type] = build_rule_report(
            rows=pred_rows,
            mapping_by_qid=mapping_by_qid,
            answer_type=answer_type,
            accuracy_by_qid=accuracy_by_qid,
            shortcut_by_qid=shortcut_by_qid,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(output_path)


if __name__ == "__main__":
    main()
