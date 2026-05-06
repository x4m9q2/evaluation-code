import argparse
import glob
import json
from collections import Counter

from prepare_llava_shortcut_inputs import (
    normalize_text,
    normalize_visual_cue,
    normalize_question_token,
    tokenize_question_keywords,
)


def resolve_inputs(patterns):
    resolved = []
    for item in patterns:
        matches = sorted(glob.glob(item))
        if matches:
            resolved.extend(matches)
        else:
            resolved.append(item)
    deduped = []
    seen = set()
    for path in resolved:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def iter_jsonl(paths):
    for path in paths:
        with open(path, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def normalize_answer(text):
    return normalize_text(text).lower()


def load_rules(path, top_k, min_confidence, min_support):
    with open(path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)
    raw_rules = payload["rules"] if isinstance(payload, dict) else payload

    normalized_rules = []
    for raw_rule in raw_rules:
        text_keywords = []
        for keyword in raw_rule.get("text_keywords", []):
            normalized_keyword = normalize_question_token(keyword)
            if normalized_keyword:
                text_keywords.append(normalized_keyword)
        visual_cues = []
        for cue in raw_rule.get("visual_cues", []):
            normalized_cue = normalize_visual_cue(cue)
            if normalized_cue:
                visual_cues.append(normalized_cue)

        support = float(raw_rule.get("support", 0.0))
        confidence = float(raw_rule.get("confidence", 0.0))
        if confidence < min_confidence or support < min_support:
            continue

        normalized_rules.append(
            {
                "rule_id": str(raw_rule.get("rule_id", "")),
                "text_keywords": tuple(sorted(set(text_keywords))),
                "visual_cues": tuple(sorted(set(visual_cues))),
                "answer": normalize_answer(raw_rule.get("answer", "")),
                "support": support,
                "confidence": confidence,
            }
        )

    normalized_rules.sort(
        key=lambda rule: (
            -rule["support"],
            -rule["confidence"],
            len(rule["text_keywords"]) + len(rule["visual_cues"]),
            rule["rule_id"],
        )
    )

    if top_k > 0:
        normalized_rules = normalized_rules[:top_k]
    return normalized_rules


def load_image_to_dataset(path):
    mapping = {}
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            mapping[int(item["image_id"])] = item.get("dataset", "")
    return mapping


def load_image_to_cues(path):
    with open(path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)

    if isinstance(payload, dict) and "detections" in payload:
        values = payload["detections"]
    elif isinstance(payload, dict):
        values = payload.values()
    else:
        values = payload

    image_to_cues = {}
    for item in values:
        image_id = int(item["image_id"])
        cues = tuple(
            sorted(
                {
                    normalized_cue
                    for cue in item.get("classes", [])
                    if (normalized_cue := normalize_visual_cue(cue))
                }
            )
        )
        image_to_cues[image_id] = cues
    return image_to_cues


def pick_rule_anchor(rule, item_rule_freq):
    items = [f"kw:{keyword}" for keyword in rule["text_keywords"]]
    items.extend(f"cue:{cue}" for cue in rule["visual_cues"])
    if not items:
        return None
    return min(items, key=lambda item: (item_rule_freq[item], item))


def build_anchor_index(rules):
    item_rule_freq = Counter()
    for rule in rules:
        items = {f"kw:{keyword}" for keyword in rule["text_keywords"]}
        items.update(f"cue:{cue}" for cue in rule["visual_cues"])
        item_rule_freq.update(items)

    anchor_to_rule_idxs = {}
    for idx, rule in enumerate(rules):
        anchor = pick_rule_anchor(rule, item_rule_freq)
        if anchor is None:
            continue
        anchor_to_rule_idxs.setdefault(anchor, []).append(idx)
        rule["anchor"] = anchor
    return anchor_to_rule_idxs


def cue_match(rule_cues, image_cues_set, mode):
    if not rule_cues:
        return True
    if mode == "all":
        return all(cue in image_cues_set for cue in rule_cues)
    return any(cue in image_cues_set for cue in rule_cues)


def update_counts(counter_by_dataset, dataset, value=1):
    counter_by_dataset["full"] += value
    counter_by_dataset[dataset] += value


def summarize_rule(rule, active_by_dataset, positive_by_dataset, answer_rate_by_dataset):
    datasets = sorted(set(active_by_dataset) | set(positive_by_dataset) | set(answer_rate_by_dataset) | {"full"})
    metrics = {}
    active_full = active_by_dataset.get("full", 0)
    positive_full = positive_by_dataset.get("full", 0)
    confidence_full = (positive_full / active_full) if active_full else 0.0
    baseline_full = answer_rate_by_dataset.get("full", 0.0)

    for dataset in datasets:
        active = active_by_dataset.get(dataset, 0)
        positive = positive_by_dataset.get(dataset, 0)
        confidence = (positive / active) if active else 0.0
        baseline = answer_rate_by_dataset.get(dataset, 0.0)
        metrics[dataset] = {
            "active": active,
            "positive": positive,
            "confidence": round(confidence, 6),
            "baseline_answer_rate": round(baseline, 6),
            "lift": round((confidence / baseline), 6) if baseline > 0 else None,
        }

    return {
        "rule_id": rule["rule_id"],
        "answer": rule["answer"],
        "text_keywords": list(rule["text_keywords"]),
        "visual_cues": list(rule["visual_cues"]),
        "anchor": rule.get("anchor"),
        "original_support": round(rule["support"], 6),
        "original_confidence": round(rule["confidence"], 6),
        "full_active": active_full,
        "full_positive": positive_full,
        "full_confidence": round(confidence_full, 6),
        "full_baseline_answer_rate": round(baseline_full, 6),
        "confidence_drop": round(rule["confidence"] - confidence_full, 6),
        "metrics_by_dataset": metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules-json", default="/path/to/sage_repro_bundle/rules.json")
    parser.add_argument("--compressed-jsonl", nargs="+", required=True)
    parser.add_argument(
        "--image-mappings-jsonl",
        default="/path/to/sage_repro_bundle/shortcut_inputs/llava_mix665k_single_noocr/image_mappings.jsonl",
    )
    parser.add_argument(
        "--detections-json",
        default="/path/to/sage_repro_bundle/shortcut_inputs/llava_mix665k_single_noocr/image_to_detection_lightnorm.json",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--top-k-rules", type=int, default=1000)
    parser.add_argument("--min-rule-confidence", type=float, default=0.9)
    parser.add_argument("--min-rule-support", type=float, default=0.0)
    parser.add_argument("--min-score", type=float, default=15.0)
    parser.add_argument("--cue-match-mode", choices=("any", "all"), default="all")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100000)
    args = parser.parse_args()

    compressed_paths = resolve_inputs(args.compressed_jsonl)
    if not compressed_paths:
        raise FileNotFoundError("No compressed JSONL inputs matched")

    rules = load_rules(
        args.rules_json,
        top_k=args.top_k_rules,
        min_confidence=args.min_rule_confidence,
        min_support=args.min_rule_support,
    )
    if not rules:
        raise ValueError("No rules selected after filtering")

    image_to_dataset = load_image_to_dataset(args.image_mappings_jsonl)
    image_to_cues = load_image_to_cues(args.detections_json)
    anchor_to_rule_idxs = build_anchor_index(rules)

    active_counts = [Counter() for _ in rules]
    positive_counts = [Counter() for _ in rules]
    dataset_row_totals = Counter()
    answer_totals_by_dataset = {}

    rows_seen = 0
    rows_qualified = 0
    rows_with_candidates = 0
    for row in iter_jsonl(compressed_paths):
        rows_seen += 1
        if args.max_rows > 0 and rows_seen > args.max_rows:
            break

        selection_score = float(row.get("selection_score", 0.0) or 0.0)
        short_answer = normalize_answer(row.get("short_answer", ""))
        if selection_score < args.min_score or not short_answer:
            continue

        rows_qualified += 1
        image_id = int(row["image_id"])
        dataset = image_to_dataset.get(image_id, "unknown")
        image_cues = image_to_cues.get(image_id, ())
        image_cues_set = set(image_cues)
        question_tokens = set(tokenize_question_keywords(row.get("question", "")))

        update_counts(dataset_row_totals, dataset, 1)
        dataset_answer_totals = answer_totals_by_dataset.setdefault(short_answer, Counter())
        update_counts(dataset_answer_totals, dataset, 1)

        items = {f"kw:{keyword}" for keyword in question_tokens}
        items.update(f"cue:{cue}" for cue in image_cues_set)

        candidate_rule_idxs = set()
        for item in items:
            rule_idxs = anchor_to_rule_idxs.get(item)
            if rule_idxs:
                candidate_rule_idxs.update(rule_idxs)

        if not candidate_rule_idxs:
            continue

        rows_with_candidates += 1

        for rule_idx in candidate_rule_idxs:
            rule = rules[rule_idx]
            if rule["text_keywords"] and not all(keyword in question_tokens for keyword in rule["text_keywords"]):
                continue
            if not cue_match(rule["visual_cues"], image_cues_set, args.cue_match_mode):
                continue

            update_counts(active_counts[rule_idx], dataset, 1)
            if short_answer == rule["answer"]:
                update_counts(positive_counts[rule_idx], dataset, 1)

        if args.progress_every > 0 and rows_seen % args.progress_every == 0:
            print(
                json.dumps(
                    {
                        "rows_seen": rows_seen,
                        "rows_qualified": rows_qualified,
                        "rows_with_candidates": rows_with_candidates,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    answer_rate_by_rule = []
    for rule in rules:
        per_dataset_rates = {}
        total_rows = dataset_row_totals.get("full", 0)
        per_dataset_rates["full"] = (answer_totals_by_dataset.get(rule["answer"], Counter()).get("full", 0) / total_rows) if total_rows else 0.0

        answer_counter = answer_totals_by_dataset.get(rule["answer"], Counter())
        for dataset, total in dataset_row_totals.items():
            if dataset == "full":
                continue
            per_dataset_rates[dataset] = (answer_counter.get(dataset, 0) / total) if total else 0.0
        answer_rate_by_rule.append(per_dataset_rates)

    rule_summaries = []
    for idx, rule in enumerate(rules):
        rule_summaries.append(
            summarize_rule(
                rule,
                active_by_dataset=active_counts[idx],
                positive_by_dataset=positive_counts[idx],
                answer_rate_by_dataset=answer_rate_by_rule[idx],
            )
        )

    rule_summaries.sort(key=lambda item: (-item["full_active"], -item["full_confidence"], item["rule_id"]))

    mean_conf_drop = sum(item["confidence_drop"] for item in rule_summaries) / len(rule_summaries)
    output = {
        "rules_json": args.rules_json,
        "compressed_jsonl": compressed_paths,
        "image_mappings_jsonl": args.image_mappings_jsonl,
        "detections_json": args.detections_json,
        "top_k_rules": args.top_k_rules,
        "min_rule_confidence": args.min_rule_confidence,
        "min_rule_support": args.min_rule_support,
        "min_score": args.min_score,
        "cue_match_mode": args.cue_match_mode,
        "rows_seen": rows_seen,
        "rows_qualified": rows_qualified,
        "rows_with_candidates": rows_with_candidates,
        "selected_rule_count": len(rules),
        "dataset_totals": dict(dataset_row_totals),
        "mean_confidence_drop": round(mean_conf_drop, 6),
        "rules": rule_summaries,
    }

    with open(args.output_json, "w", encoding="utf-8") as fp:
        json.dump(output, fp, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "rows_seen": rows_seen,
                "rows_qualified": rows_qualified,
                "rows_with_candidates": rows_with_candidates,
                "selected_rule_count": len(rules),
                "mean_confidence_drop": round(mean_conf_drop, 6),
                "output_json": args.output_json,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
