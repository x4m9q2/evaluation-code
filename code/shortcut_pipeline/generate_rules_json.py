#!/usr/bin/env python3
"""Mine shortcut rules and export a matcher-compatible rules.json."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_TOOLS_DIR = REPO_ROOT / "code" / "data_tools"
if str(DATA_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_TOOLS_DIR))

from prepare_llava_shortcut_inputs import (  # noqa: E402
    normalize_text,
    normalize_visual_cue,
    tokenize_question_keywords,
)
from rule_mining import Rule, fit  # noqa: E402


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def unwrap_list(payload: Any, key: str) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        items = payload.get(key, [])
    else:
        items = payload
    if not isinstance(items, list):
        raise RuntimeError(f"Expected a list under '{key}'.")
    return items


def normalize_answer(value: Any) -> str:
    return normalize_text(str(value or "")).lower()


def take_prefix(items: Sequence[Any], proportion: float) -> Sequence[Any]:
    if not items:
        return items
    if proportion >= 1.0:
        return items
    total = max(1, int(len(items) * proportion))
    return items[:total]


def build_visual_index(payload: Any) -> Dict[str, Dict[str, Any]]:
    if isinstance(payload, dict) and "detections" in payload:
        values = payload["detections"]
    elif isinstance(payload, dict):
        return {str(key): value for key, value in payload.items()}
    else:
        values = payload

    indexed: Dict[str, Dict[str, Any]] = {}
    for item in values:
        image_id = item.get("image_id")
        if image_id is None:
            continue
        indexed[str(image_id)] = item
    return indexed


def extract_visual_tokens(
    visual_index: Mapping[str, Mapping[str, Any]],
    image_id: Any,
    visual_threshold: float,
) -> List[str]:
    if image_id is None:
        return []

    raw = visual_index.get(str(image_id), {})
    classes = list(raw.get("classes", []))
    scores = list(raw.get("scores", []))
    tokens: List[str] = []

    for idx, class_name in enumerate(classes):
        score = scores[idx] if idx < len(scores) else 1.0
        if score < visual_threshold:
            continue
        normalized = normalize_visual_cue(class_name)
        if normalized:
            tokens.append("V_" + normalized)
    return tokens


def build_transaction(
    textual_tokens: Iterable[str],
    visual_tokens: Iterable[str],
) -> List[str]:
    merged = {token for token in textual_tokens if token}
    merged.update(token for token in visual_tokens if token)
    return sorted(merged)


def extract_vqa_answer(annotation: Mapping[str, Any]) -> str:
    answer = normalize_answer(annotation.get("multiple_choice_answer"))
    if answer:
        return answer

    answers = annotation.get("answers") or []
    if answers:
        counts = Counter(normalize_answer(item.get("answer")) for item in answers)
        counts.pop("", None)
        if counts:
            return counts.most_common(1)[0][0]
    return ""


def build_vqa_rows(
    questions_path: str,
    annotations_path: str,
    visual_words_path: str,
    textual: bool,
    visual: bool,
    visual_threshold: float,
    proportion: float,
) -> List[Tuple[List[str], str, str]]:
    questions = unwrap_list(load_json(questions_path), "questions")
    annotations = unwrap_list(load_json(annotations_path), "annotations")
    questions = list(take_prefix(questions, proportion))
    annotations = list(take_prefix(annotations, proportion))

    if len(questions) != len(annotations):
        raise RuntimeError(
            f"VQA questions/annotations length mismatch: {len(questions)} vs {len(annotations)}"
        )

    visual_index = build_visual_index(load_json(visual_words_path))
    rows: List[Tuple[List[str], str, str]] = []
    for question, annotation in zip(questions, annotations):
        textual_tokens = tokenize_question_keywords(question.get("question", "")) if textual else []
        visual_tokens = (
            extract_visual_tokens(visual_index, question.get("image_id"), visual_threshold)
            if visual
            else []
        )
        answer = extract_vqa_answer(annotation)
        if not answer:
            continue
        answer_type = normalize_answer(annotation.get("answer_type")) or "unknown"
        rows.append((build_transaction(textual_tokens, visual_tokens), answer, answer_type))
    return rows


def build_vg_rows(
    qa_path: str,
    visual_words_path: str,
    textual: bool,
    visual: bool,
    visual_threshold: float,
    proportion: float,
) -> List[Tuple[List[str], str, str]]:
    payload = load_json(qa_path)
    if not isinstance(payload, list):
        raise RuntimeError("Visual Genome QA input must be a JSON list.")

    entries = list(take_prefix(payload, proportion))
    visual_index = build_visual_index(load_json(visual_words_path))
    rows: List[Tuple[List[str], str, str]] = []

    for entry in entries:
        nested_qas = entry.get("qas")
        if isinstance(nested_qas, list):
            base_image_id = entry.get("image_id", entry.get("id"))
            qa_iter = nested_qas
        else:
            base_image_id = entry.get("image_id", entry.get("id"))
            qa_iter = [entry]

        for qa in qa_iter:
            textual_tokens = tokenize_question_keywords(qa.get("question", "")) if textual else []
            visual_tokens = (
                extract_visual_tokens(
                    visual_index,
                    qa.get("image_id", base_image_id),
                    visual_threshold,
                )
                if visual
                else []
            )
            answer = normalize_answer(qa.get("answer"))
            if not answer:
                continue
            answer_type = normalize_answer(qa.get("answer_type")) or "unknown"
            rows.append((build_transaction(textual_tokens, visual_tokens), answer, answer_type))
    return rows


def build_gqa_rows(
    qa_path: str,
    visual_words_path: str,
    textual: bool,
    visual: bool,
    visual_threshold: float,
    proportion: float,
) -> List[Tuple[List[str], str, str]]:
    payload = load_json(qa_path)
    if isinstance(payload, dict):
        items = sorted(payload.items(), key=lambda pair: str(pair[0]))
        entries = [entry for _, entry in take_prefix(items, proportion)]
    elif isinstance(payload, list):
        entries = list(take_prefix(payload, proportion))
    else:
        raise RuntimeError("GQA input must be a JSON dict or list.")

    visual_index = build_visual_index(load_json(visual_words_path))
    rows: List[Tuple[List[str], str, str]] = []

    for entry in entries:
        textual_tokens = tokenize_question_keywords(entry.get("question", "")) if textual else []
        visual_tokens = (
            extract_visual_tokens(visual_index, entry.get("imageId", entry.get("image_id")), visual_threshold)
            if visual
            else []
        )
        answer = normalize_answer(entry.get("answer"))
        if not answer:
            continue
        answer_type = normalize_answer(entry.get("answer_type")) or "unknown"
        rows.append((build_transaction(textual_tokens, visual_tokens), answer, answer_type))
    return rows


def filter_rows_by_answer_frequency(
    rows: Sequence[Tuple[List[str], str, str]],
    most_common_answers: int,
) -> List[Tuple[List[str], str, str]]:
    if most_common_answers <= 0:
        return list(rows)

    counts = Counter(answer for _, answer, _ in rows)
    keep = {answer for answer, _ in counts.most_common(most_common_answers)}
    return [row for row in rows if row[1] in keep]


def vectorize_rows(
    rows: Sequence[Tuple[List[str], str, str]],
) -> Tuple[List[List[int]], List[str], List[str], List[int], List[str]]:
    if not rows:
        raise RuntimeError("No rows available after preprocessing.")

    tokens = sorted({token for transaction, _, _ in rows for token in transaction})
    if not tokens:
        raise RuntimeError("No textual or visual tokens were extracted from the dataset.")

    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    answer_vocab = sorted({answer for _, answer, _ in rows})
    answer_to_id = {answer: idx for idx, answer in enumerate(answer_vocab)}

    transactions: List[List[int]] = []
    answer_ids: List[int] = []
    answer_type_counts: Dict[str, Counter[str]] = defaultdict(Counter)

    for transaction, answer, answer_type in rows:
        if not transaction:
            continue
        transactions.append([token_to_id[token] for token in transaction])
        answer_ids.append(answer_to_id[answer])
        answer_type_counts[answer][answer_type] += 1

    if not transactions:
        raise RuntimeError("All rows became empty after tokenization/filtering.")

    answer_types = [
        answer_type_counts[answer].most_common(1)[0][0] if answer_type_counts[answer] else "unknown"
        for answer in answer_vocab
    ]
    return transactions, answer_vocab, tokens, answer_ids, answer_types


def serialize_rules(
    rules: Sequence[Rule],
    tokens: Sequence[str],
    answers: Sequence[str],
    answer_types: Sequence[str],
) -> Dict[str, List[Dict[str, Any]]]:
    serialized: List[Dict[str, Any]] = []
    for idx, rule in enumerate(rules, start=1):
        raw_tokens = [tokens[token_id] for token_id in rule.itemset]
        text_keywords = sorted({token for token in raw_tokens if not token.startswith("V_")})
        visual_cues = sorted({token[2:] for token in raw_tokens if token.startswith("V_")})
        serialized.append(
            {
                "rule_id": idx,
                "itemset_ids": list(rule.itemset),
                "tokens": raw_tokens,
                "text_keywords": text_keywords,
                "visual_cues": visual_cues,
                "trigger": " ".join(text_keywords),
                "answer_id": rule.ans,
                "answer": answers[rule.ans],
                "answer_type": answer_types[rule.ans] if rule.ans < len(answer_types) else "unknown",
                "support": float(rule.sup),
                "confidence": float(rule.conf),
            }
        )
    return {"rules": serialized}


def timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


@contextmanager
def stage(label: str):
    start = time.time()
    print(f"[{timestamp()}] {label} ...", flush=True)
    try:
        yield
    finally:
        elapsed = time.time() - start
        print(f"[{timestamp()}] {label} done in {elapsed:.1f}s", flush=True)


def log_status(message: str) -> None:
    print(f"[{timestamp()}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine shortcut rules and export rules.json only")
    parser.add_argument("--dataset", choices=["vqa", "vg", "gqa"], default="vqa")
    parser.add_argument(
        "--train_questions_path",
        default=str(REPO_ROOT / "data" / "detect-shortcuts" / "data" / "vqa2" / "v2_OpenEnded_mscoco_train2014_questions.json"),
    )
    parser.add_argument(
        "--train_annotations_path",
        default=str(REPO_ROOT / "data" / "detect-shortcuts" / "data" / "vqa2" / "v2_mscoco_train2014_annotations.json"),
    )
    parser.add_argument(
        "--visual_words",
        default=str(REPO_ROOT / "data" / "shortcut_pipeline" / "image_to_detection.json"),
    )
    parser.add_argument("--gminer_path", default=str(REPO_ROOT / "code" / "shortcut_pipeline" / "bin" / "GMiner"))
    parser.add_argument("--save_dir", default=str(REPO_ROOT / "data" / "shortcut_pipeline" / "rules"))
    parser.add_argument("--support", type=float, default=2.1e-5, help="Minimum GMiner support")
    parser.add_argument("--max_length", type=int, default=5, help="Max antecedent size")
    parser.add_argument("--min_conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--visual_threshold", type=float, default=0.5)
    parser.add_argument("--most_common_answers", type=int, default=3000)
    parser.add_argument("--proportion", type=float, default=1.0)
    parser.add_argument("--textual", action="store_true", default=True)
    parser.add_argument("--no-textual", dest="textual", action="store_false")
    parser.add_argument("--visual", action="store_true", default=True)
    parser.add_argument("--no-visual", dest="visual", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with stage("Stage 1/3: Prepare token transactions and answer ids"):
        if args.dataset == "vqa":
            rows = build_vqa_rows(
                questions_path=args.train_questions_path,
                annotations_path=args.train_annotations_path,
                visual_words_path=args.visual_words,
                textual=args.textual,
                visual=args.visual,
                visual_threshold=args.visual_threshold,
                proportion=args.proportion,
            )
        elif args.dataset == "vg":
            rows = build_vg_rows(
                qa_path=args.train_questions_path,
                visual_words_path=args.visual_words,
                textual=args.textual,
                visual=args.visual,
                visual_threshold=args.visual_threshold,
                proportion=args.proportion,
            )
        else:
            rows = build_gqa_rows(
                qa_path=args.train_questions_path,
                visual_words_path=args.visual_words,
                textual=args.textual,
                visual=args.visual,
                visual_threshold=args.visual_threshold,
                proportion=args.proportion,
            )

        rows = filter_rows_by_answer_frequency(rows, args.most_common_answers)
        transactions, answer_vocab, tokens, answer_ids, answer_types = vectorize_rows(rows)
    log_status(
        f"Prepared {len(transactions):,} transactions | token vocab {len(tokens):,} | answers {len(answer_vocab):,}"
    )

    effective_support = args.support
    min_support = 1.0 / len(transactions)
    if effective_support * len(transactions) < 1.0:
        effective_support = min_support
        log_status(
            f"Adjusted support from {args.support} to {effective_support} so GMiner sees at least one example per rule"
        )

    with stage("Stage 2/3: Run GMiner and compute rule confidences"):
        rules = fit(
            transactions,
            answer_ids,
            gminer_support=effective_support,
            gminer_max_length=args.max_length,
            gminer_path=args.gminer_path,
        )
    log_status(f"Mined {len(rules):,} candidate rules before confidence filtering")
    rules = [rule for rule in rules if rule.conf >= args.min_conf]
    log_status(f"{len(rules):,} rules remain after enforcing min_conf >= {args.min_conf}")

    if args.dataset == "gqa":
        before = len(rules)
        rules = [
            rule
            for rule in rules
            if any(tokens[token_id].startswith("V_") for token_id in rule.itemset)
        ]
        log_status(f"{before - len(rules):,} rules dropped for missing visual cues")

    with stage("Stage 3/3: Serialize rules.json"):
        os.makedirs(args.save_dir, exist_ok=True)
        payload = serialize_rules(rules, tokens, answer_vocab, answer_types)
        output_path = os.path.join(args.save_dir, "rules.json")
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    log_status(f"Saved {len(payload['rules']):,} rules to {output_path}")


if __name__ == "__main__":
    main()
