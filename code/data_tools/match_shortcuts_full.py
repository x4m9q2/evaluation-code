import argparse
import heapq
import json
import time
from collections import Counter, defaultdict

from prepare_llava_shortcut_inputs import (
    normalize_text,
    normalize_question_token,
    normalize_visual_cue,
    tokenize_question_keywords,
)


def iter_wrapped_json_array(path, key, chunk_size=1 << 20):
    decoder = json.JSONDecoder()
    marker = json.dumps(key)
    with open(path, "r", encoding="utf-8") as f:
        buffer = ""
        eof = False
        started = False

        while not started:
            if not eof and len(buffer) < chunk_size:
                chunk = f.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True

            pos = buffer.find(marker)
            if pos != -1:
                array_pos = buffer.find("[", pos + len(marker))
                if array_pos != -1:
                    buffer = buffer[array_pos + 1 :]
                    started = True
                    break

            if eof:
                raise ValueError(f"Could not find array key {key!r} in {path}")

            if len(buffer) > len(marker) * 4:
                buffer = buffer[-len(marker) * 4 :]

        while True:
            buffer = buffer.lstrip()
            if buffer.startswith("]"):
                return
            if buffer.startswith(","):
                buffer = buffer[1:]
                continue
            if not buffer and eof:
                raise ValueError(f"{path} ended unexpectedly while reading {key}")

            while True:
                try:
                    item, idx = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    chunk = f.read(chunk_size)
                    if chunk:
                        buffer += chunk
                    else:
                        eof = True

            yield item
            buffer = buffer[idx:]

            if not eof and len(buffer) < chunk_size:
                chunk = f.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True


def load_rules(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    raw_rules = payload["rules"] if isinstance(payload, dict) else payload

    rules = []
    answer_cue_to_rule_idxs = defaultdict(lambda: defaultdict(list))
    answer_to_no_visual_rule_idxs = defaultdict(list)
    cue_vocab = set()

    for idx, raw_rule in enumerate(raw_rules):
        text_keywords = [
            normalized_keyword
            for keyword in raw_rule.get("text_keywords", [])
            if (normalized_keyword := normalize_question_token(keyword))
        ]
        visual_cues = [
            normalized_cue
            for cue in raw_rule.get("visual_cues", [])
            if (normalized_cue := normalize_visual_cue(cue))
        ]
        answer = normalize_text(raw_rule.get("answer", "")).lower()
        rule = {
            "rule_id": str(raw_rule.get("rule_id", "")),
            "text_keywords": text_keywords,
            "visual_cues": visual_cues,
            "answer": answer,
            "confidence": float(raw_rule.get("confidence", 0.0)),
            "support": raw_rule.get("support", 0),
        }
        rules.append(rule)

        if visual_cues:
            for cue in visual_cues:
                answer_cue_to_rule_idxs[answer][cue].append(idx)
                cue_vocab.add(cue)
        else:
            answer_to_no_visual_rule_idxs[answer].append(idx)

    return rules, answer_cue_to_rule_idxs, answer_to_no_visual_rule_idxs, cue_vocab


def load_image_cues(path, cue_vocab):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

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
                    if (normalized_cue := normalize_visual_cue(cue)) in cue_vocab
                }
            )
        )
        image_to_cues[image_id] = cues
    return image_to_cues


def build_candidate_rule_idxs(answer, image_cues, answer_cue_to_rule_idxs, answer_to_no_visual_rule_idxs):
    answer_index = answer_cue_to_rule_idxs.get(answer)
    lists = []
    if answer_index:
        for cue in image_cues:
            rule_idxs = answer_index.get(cue)
            if rule_idxs:
                lists.append(rule_idxs)

    no_visual = answer_to_no_visual_rule_idxs.get(answer)
    if no_visual:
        lists.append(no_visual)

    if not lists:
        return ()

    merged = []
    last_idx = None
    for idx in heapq.merge(*lists):
        if idx != last_idx:
            merged.append(idx)
            last_idx = idx
    return tuple(merged)


def match_rule(question_tokens, candidate_rule_idxs, rules):
    for rule_idx in candidate_rule_idxs:
        rule = rules[rule_idx]
        if all(keyword in question_tokens for keyword in rule["text_keywords"]):
            return rule_idx
    return None


def extract_answer(annotation):
    answers = annotation.get("answers") or []
    if answers:
        return normalize_text(answers[0].get("answer", "")).lower()
    return normalize_text(annotation.get("multiple_choice_answer", "")).lower()


def run(args):
    rules, answer_cue_to_rule_idxs, answer_to_no_visual_rule_idxs, cue_vocab = load_rules(args.rules_json)
    image_to_cues = load_image_cues(args.detections_json, cue_vocab)

    total = 0
    matched = 0
    no_answer_rules = 0
    no_image_cues = 0
    no_candidates = 0
    by_dataset = Counter()
    matched_by_dataset = Counter()
    cache = {}
    start = time.time()

    with open(args.output_jsonl, "w", encoding="utf-8") as out_fp:
        q_iter = iter_wrapped_json_array(args.questions_json, "questions")
        a_iter = iter_wrapped_json_array(args.annotations_json, "annotations")

        for question, annotation in zip(q_iter, a_iter):
            total += 1

            question_id = int(question["question_id"])
            annotation_qid = int(annotation["question_id"])
            if question_id != annotation_qid:
                raise ValueError(f"Question/annotation mismatch: {question_id} != {annotation_qid}")

            dataset = question.get("dataset", "")
            by_dataset[dataset] += 1

            answer = extract_answer(annotation)
            if answer not in answer_cue_to_rule_idxs and answer not in answer_to_no_visual_rule_idxs:
                no_answer_rules += 1
                continue

            image_id = int(question["image_id"])
            image_cues = image_to_cues.get(image_id, ())
            if not image_cues and answer not in answer_to_no_visual_rule_idxs:
                no_image_cues += 1
                continue

            cache_key = (image_id, answer)
            candidate_rule_idxs = cache.get(cache_key)
            if candidate_rule_idxs is None:
                candidate_rule_idxs = build_candidate_rule_idxs(
                    answer,
                    image_cues,
                    answer_cue_to_rule_idxs,
                    answer_to_no_visual_rule_idxs,
                )
                if len(cache) > args.cache_limit:
                    cache.clear()
                cache[cache_key] = candidate_rule_idxs

            if not candidate_rule_idxs:
                no_candidates += 1
                continue

            question_tokens = set(tokenize_question_keywords(question.get("question", "")))
            matched_rule_idx = match_rule(question_tokens, candidate_rule_idxs, rules)
            if matched_rule_idx is None:
                continue

            matched += 1
            matched_by_dataset[dataset] += 1
            matched_rule = rules[matched_rule_idx]
            out_fp.write(
                json.dumps(
                    {
                        "question_id": question_id,
                        "image_id": image_id,
                        "llava_id": question.get("llava_id"),
                        "image": question.get("image"),
                        "dataset": dataset,
                        "question": question.get("question"),
                        "answer": answer,
                        "text_keywords": matched_rule["text_keywords"],
                        "visual_cues": matched_rule["visual_cues"],
                        "matched_rule": {
                            "rule_id": matched_rule["rule_id"],
                            "confidence": matched_rule["confidence"],
                            "support": matched_rule["support"],
                            "answer": matched_rule["answer"],
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            if args.progress_every and total % args.progress_every == 0:
                elapsed = time.time() - start
                print(
                    json.dumps(
                        {
                            "processed": total,
                            "matched": matched,
                            "elapsed_sec": round(elapsed, 1),
                            "samples_per_sec": round(total / elapsed, 1) if elapsed else None,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    elapsed = time.time() - start
    stats = {
        "rules_json": args.rules_json,
        "questions_json": args.questions_json,
        "annotations_json": args.annotations_json,
        "detections_json": args.detections_json,
        "output_jsonl": args.output_jsonl,
        "total_questions": total,
        "matched_questions": matched,
        "matched_ratio": (matched / total) if total else 0.0,
        "no_answer_rules": no_answer_rules,
        "no_image_cues": no_image_cues,
        "no_candidates": no_candidates,
        "elapsed_sec": elapsed,
        "samples_per_sec": (total / elapsed) if elapsed else 0.0,
        "by_dataset": dict(by_dataset),
        "matched_by_dataset": dict(matched_by_dataset),
    }
    with open(args.stats_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules-json", default="/path/to/sage_repro_bundle/rules.json")
    parser.add_argument("--questions-json", default="/path/to/sage_repro_bundle/shortcut_inputs/llava_mix665k_single_noocr/questions.json")
    parser.add_argument("--annotations-json", default="/path/to/sage_repro_bundle/shortcut_inputs/llava_mix665k_single_noocr/annotations.json")
    parser.add_argument("--detections-json", default="/path/to/sage_repro_bundle/shortcut_inputs/llava_mix665k_single_noocr/image_to_detection.json")
    parser.add_argument("--output-jsonl", default="/path/to/sage_repro_bundle/shortcut_inputs/llava_mix665k_single_noocr/shortcut_matches.jsonl")
    parser.add_argument("--stats-json", default="/path/to/sage_repro_bundle/shortcut_inputs/llava_mix665k_single_noocr/shortcut_match_stats.json")
    parser.add_argument("--progress-every", type=int, default=100000)
    parser.add_argument("--cache-limit", type=int, default=200000)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
