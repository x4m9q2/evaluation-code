import argparse
import glob
import json
import math
import os
import subprocess
from collections import Counter, namedtuple
from typing import Dict, Iterable, List, Sequence, Tuple

from prepare_llava_shortcut_inputs import (
    normalize_text,
    normalize_visual_cue,
    tokenize_question_keywords,
)
from rule_utils import superset_filtering


Rule = namedtuple("Rule", ["itemset", "ans", "sup", "conf"])


def tokenize_question(text: str) -> List[str]:
    return tokenize_question_keywords(text)


def iter_jsonl(paths: Sequence[str]) -> Iterable[dict]:
    for path in paths:
        with open(path, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def resolve_input_paths(inputs: Sequence[str]) -> List[str]:
    resolved: List[str] = []
    for item in inputs:
        matches = sorted(glob.glob(item))
        if matches:
            resolved.extend(matches)
        elif os.path.exists(item):
            resolved.append(item)
    unique = []
    seen = set()
    for path in resolved:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    if not unique:
        raise FileNotFoundError(f"No JSONL inputs matched: {inputs}")
    return unique


def load_image_to_cues(path: str) -> Dict[int, Tuple[str, ...]]:
    with open(path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)

    if isinstance(payload, dict) and "detections" in payload:
        values = payload["detections"]
    elif isinstance(payload, dict):
        values = payload.values()
    else:
        values = payload

    image_to_cues: Dict[int, Tuple[str, ...]] = {}
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


def first_pass(
    input_paths: Sequence[str],
    image_to_cues: Dict[int, Tuple[str, ...]],
    min_score: float,
):
    answer_counts = Counter()
    keyword_counts = Counter()
    cue_counts = Counter()
    total_rows = 0

    for row in iter_jsonl(input_paths):
        if float(row.get("selection_score", 0.0)) < min_score:
            continue

        short_answer = normalize_text(row.get("short_answer", "")).lower()
        if not short_answer:
            continue

        total_rows += 1
        answer_counts[short_answer] += 1

        question_tokens = sorted(set(tokenize_question(row.get("question", ""))))
        keyword_counts.update(question_tokens)

        image_id = int(row.get("image_id", 0))
        cue_counts.update(image_to_cues.get(image_id, ()))

    return total_rows, answer_counts, keyword_counts, cue_counts


def select_vocab(
    total_rows: int,
    answer_counts: Counter,
    keyword_counts: Counter,
    cue_counts: Counter,
    min_support: float,
    max_cues: int,
):
    support_count = max(1, math.ceil(total_rows * min_support))

    kept_answers = {
        answer
        for answer, count in answer_counts.items()
        if count >= support_count
    }
    kept_keywords = {
        token
        for token, count in keyword_counts.items()
        if count >= support_count
    }
    cue_candidates = [
        cue
        for cue, count in sorted(cue_counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= support_count
    ]
    if max_cues <= 0:
        kept_cues = set(cue_candidates)
    else:
        kept_cues = set(cue_candidates[:max_cues])
    return support_count, kept_answers, kept_keywords, kept_cues


def build_feature_vocabs(
    kept_keywords: Sequence[str],
    kept_cues: Sequence[str],
    kept_answers: Sequence[str],
):
    item_to_feature = {}
    feature_to_item = {}
    next_item_id = 1

    keyword_to_item = {}
    for token in sorted(kept_keywords):
        keyword_to_item[token] = next_item_id
        item_to_feature[next_item_id] = ("kw", token)
        feature_to_item[("kw", token)] = next_item_id
        next_item_id += 1

    cue_to_item = {}
    for cue in sorted(kept_cues):
        cue_to_item[cue] = next_item_id
        item_to_feature[next_item_id] = ("cue", cue)
        feature_to_item[("cue", cue)] = next_item_id
        next_item_id += 1

    answer_to_item = {}
    answer_id_lookup = {}
    answer_list = sorted(kept_answers)
    for ans_id, answer in enumerate(answer_list):
        item_id = next_item_id
        answer_to_item[answer] = item_id
        answer_id_lookup[item_id] = ans_id
        next_item_id += 1

    return {
        "keyword_to_item": keyword_to_item,
        "cue_to_item": cue_to_item,
        "answer_to_item": answer_to_item,
        "item_to_feature": item_to_feature,
        "answer_id_lookup": answer_id_lookup,
        "answer_list": answer_list,
    }


def write_gminer_input(
    input_paths: Sequence[str],
    image_to_cues: Dict[int, Tuple[str, ...]],
    min_score: float,
    kept_answers: set,
    keyword_to_item: Dict[str, int],
    cue_to_item: Dict[str, int],
    answer_to_item: Dict[str, int],
    output_path: str,
) -> int:
    transaction_count = 0
    with open(output_path, "w", encoding="utf-8") as fp:
        for row in iter_jsonl(input_paths):
            if float(row.get("selection_score", 0.0)) < min_score:
                continue

            answer = normalize_text(row.get("short_answer", "")).lower()
            if answer not in kept_answers:
                continue

            question_tokens = {
                keyword_to_item[token]
                for token in tokenize_question(row.get("question", ""))
                if token in keyword_to_item
            }
            image_id = int(row.get("image_id", 0))
            cue_items = {
                cue_to_item[cue]
                for cue in image_to_cues.get(image_id, ())
                if cue in cue_to_item
            }

            feature_items = sorted(question_tokens | cue_items)
            answer_item = answer_to_item[answer]
            transaction = feature_items + [answer_item]

            fp.write(" ".join(str(item) for item in transaction))
            fp.write("\n")
            transaction_count += 1

    return transaction_count


def run_gminer(gminer_path: str, input_path: str, output_path: str, min_support: float, max_length: int):
    command = [
        gminer_path,
        "-i",
        input_path,
        "-o",
        output_path,
        "-s",
        str(min_support),
        "-w",
        "1",
    ]
    if max_length > 0:
        command.extend(["-l", str(max_length)])
    subprocess.run(command, check=True)


def load_itemsets(path: str):
    itemsets = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ")
            itemset = tuple(sorted(int(piece) for piece in parts[:-1]))
            support = float(parts[-1][:-1][1:])
            itemsets.append((itemset, support))
    return itemsets


def build_rules(
    itemsets,
    answer_id_lookup: Dict[int, int],
    item_to_feature: Dict[int, Tuple[str, str]],
    answer_list: Sequence[str],
    min_confidence: float,
    require_cue: bool,
):
    supports_by_itemset = {(): 1.0}
    for itemset, support in itemsets:
        supports_by_itemset[itemset] = support

    pre_rules = []
    for itemset, support_with_answer in itemsets:
        answer_item = None
        answer_index = None
        for idx, item in enumerate(itemset):
            if item in answer_id_lookup:
                answer_item = item
                answer_index = idx
                break
        if answer_item is None:
            continue
        antecedent = itemset[:answer_index] + itemset[answer_index + 1 :]
        if not antecedent:
            continue
        pre_rules.append((antecedent, answer_id_lookup[answer_item], support_with_answer))

    temp_rules: List[Rule] = []
    for antecedent, answer_id, support_with_answer in pre_rules:
        antecedent_support = supports_by_itemset.get(tuple(sorted(antecedent)))
        if antecedent_support is None or antecedent_support <= 0.0:
            continue

        confidence = support_with_answer / antecedent_support
        if confidence < min_confidence:
            continue

        has_cue = any(item_to_feature[item][0] == "cue" for item in antecedent)
        if require_cue and not has_cue:
            continue

        temp_rules.append(
            Rule(
                itemset=tuple(sorted(antecedent)),
                ans=answer_id,
                sup=antecedent_support,
                conf=confidence,
            )
        )

    filtered_rules = superset_filtering(temp_rules)
    filtered_rules = sorted(
        filtered_rules,
        key=lambda rule: (-rule.conf, -rule.sup, len(rule.itemset), rule.ans, rule.itemset),
    )

    output_rules = []
    for rule_id, rule in enumerate(filtered_rules, start=1):
        text_keywords = []
        visual_cues = []
        for item in rule.itemset:
            kind, value = item_to_feature[item]
            if kind == "kw":
                text_keywords.append(value)
            elif kind == "cue":
                visual_cues.append(value)

        text_keywords = sorted(set(text_keywords))
        visual_cues = sorted(set(visual_cues))
        if require_cue and not visual_cues:
            continue

        output_rules.append(
            {
                "rule_id": rule_id,
                "text_keywords": text_keywords,
                "visual_cues": visual_cues,
                "trigger": " ".join(text_keywords),
                "answer": answer_list[rule.ans],
                "support": round(float(rule.sup), 6),
                "confidence": round(float(rule.conf), 6),
            }
        )

    return output_rules


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compressed-jsonl",
        nargs="+",
        default=["/path/to/sage_repro_bundle/shortcut_outputs/compressed_answers_20260331_052254/shard_*.jsonl"],
    )
    parser.add_argument(
        "--detections-json",
        default="/path/to/sage_repro_bundle/shortcut_inputs/llava_mix665k_single_noocr/image_to_detection.json",
    )
    parser.add_argument("--gminer-path", default="/path/to/sage_repro_bundle/GMiner")
    parser.add_argument("--output-rules-json", required=True)
    parser.add_argument("--output-metadata-json")
    parser.add_argument("--work-dir", default="/path/to/sage_repro_bundle/tmp/rule_mining_llava")
    parser.add_argument("--min-score", type=float, default=15.0)
    parser.add_argument("--min-support", type=float, default=0.00015)
    parser.add_argument("--min-confidence", type=float, default=0.9)
    parser.add_argument("--max-rule-items", type=int, default=4)
    parser.add_argument(
        "--max-cues",
        type=int,
        default=1024,
        help="Global cap on retained visual cues after support filtering; <= 0 keeps all cues.",
    )
    parser.add_argument("--require-cue", action="store_true", default=True)
    parser.add_argument("--allow-no-cue", dest="require_cue", action="store_false")
    args = parser.parse_args()

    input_paths = resolve_input_paths(args.compressed_jsonl)
    os.makedirs(args.work_dir, exist_ok=True)

    image_to_cues = load_image_to_cues(args.detections_json)
    total_rows, answer_counts, keyword_counts, cue_counts = first_pass(
        input_paths=input_paths,
        image_to_cues=image_to_cues,
        min_score=args.min_score,
    )
    if total_rows == 0:
        raise RuntimeError("No compressed rows passed the score threshold.")

    support_count, kept_answers, kept_keywords, kept_cues = select_vocab(
        total_rows=total_rows,
        answer_counts=answer_counts,
        keyword_counts=keyword_counts,
        cue_counts=cue_counts,
        min_support=args.min_support,
        max_cues=args.max_cues,
    )

    vocabs = build_feature_vocabs(
        kept_keywords=kept_keywords,
        kept_cues=kept_cues,
        kept_answers=kept_answers,
    )

    gminer_in = os.path.join(args.work_dir, "gminer_in.txt")
    gminer_out = os.path.join(args.work_dir, "gminer_out.txt")

    transaction_count = write_gminer_input(
        input_paths=input_paths,
        image_to_cues=image_to_cues,
        min_score=args.min_score,
        kept_answers=kept_answers,
        keyword_to_item=vocabs["keyword_to_item"],
        cue_to_item=vocabs["cue_to_item"],
        answer_to_item=vocabs["answer_to_item"],
        output_path=gminer_in,
    )
    if transaction_count == 0:
        raise RuntimeError("No transactions were written for GMiner.")
    effective_min_support = min(1.0, support_count / transaction_count)
    if effective_min_support <= 0.0:
        raise RuntimeError("Effective GMiner support must be positive.")

    run_gminer(
        gminer_path=args.gminer_path,
        input_path=gminer_in,
        output_path=gminer_out,
        min_support=effective_min_support,
        max_length=args.max_rule_items + 1,
    )
    itemsets = load_itemsets(gminer_out)
    rules = build_rules(
        itemsets=itemsets,
        answer_id_lookup=vocabs["answer_id_lookup"],
        item_to_feature=vocabs["item_to_feature"],
        answer_list=vocabs["answer_list"],
        min_confidence=args.min_confidence,
        require_cue=args.require_cue,
    )

    payload = {"rules": rules}
    os.makedirs(os.path.dirname(os.path.abspath(args.output_rules_json)), exist_ok=True)
    with open(args.output_rules_json, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False)

    metadata = {
        "compressed_jsonl": input_paths,
        "detections_json": args.detections_json,
        "gminer_path": args.gminer_path,
        "min_score": args.min_score,
        "min_support": args.min_support,
        "min_confidence": args.min_confidence,
        "support_count": support_count,
        "effective_gminer_support": effective_min_support,
        "max_rule_items": args.max_rule_items,
        "max_cues": args.max_cues,
        "max_cues_effective": None if args.max_cues <= 0 else args.max_cues,
        "require_cue": args.require_cue,
        "qualified_rows": total_rows,
        "transaction_count": transaction_count,
        "kept_answers": len(kept_answers),
        "kept_keywords": len(kept_keywords),
        "kept_cues": len(kept_cues),
        "num_itemsets": len(itemsets),
        "num_rules": len(rules),
    }

    metadata_path = args.output_metadata_json
    if not metadata_path:
        base, _ = os.path.splitext(os.path.abspath(args.output_rules_json))
        metadata_path = base + ".meta.json"
    with open(metadata_path, "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, ensure_ascii=False, indent=2)

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
