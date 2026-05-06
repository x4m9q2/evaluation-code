import argparse
import hashlib
import json
import re
from pathlib import Path


IMAGE_CUE_RE = re.compile(
    r"(<image>|"
    r"\b(image|picture|photo|photograph|scene|visible|shown)\b|"
    r"\bin (the|this) (image|picture|photo)\b|"
    r"(图中|图片|照片|画面|这张图|这幅图))",
    re.IGNORECASE,
)

OCR_RE = re.compile(
    r"(\bocr\b|"
    r"\bread (the )?text\b|"
    r"\btext in (the )?(image|picture|photo)\b|"
    r"\bwhat does .* say\b|"
    r"\bwhat is written\b|"
    r"\bwritten on\b|"
    r"\btranscribe\b|"
    r"\btranscription\b|"
    r"\bword(s)? on\b|"
    r"\bletter(s)? on\b|"
    r"\blicense plate\b|"
    r"(识别|读出|文字|文本|写着|写的内容|牌子上|看清字))",
    re.IGNORECASE,
)

LEADING_TOKENS = {
    "the",
    "this",
    "that",
    "these",
    "those",
    "it",
    "there",
    "he",
    "she",
    "they",
    "we",
    "i",
}

BAD_END_TOKENS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
}

EXACT_BAD_ANSWERS = {
    "",
    "There are",
    "There is",
    "It is",
    "He is",
    "She is",
    "They are",
    "We are",
    "He does",
    "She does",
    "It does",
    "They do",
    "We do",
    "I do",
    "In the image,",
    "It is not visible,",
    "Yes, there is a",
    "Yes, there is an",
    "Yes, there are",
    "No, there is no",
    "No, there are no",
    "The bathtub is",
}

AGGRESSIVE_EXACT_BAD_ANSWERS = {
    "Buick, land rover, ford, chrysler, kia, subaru,",
    "Jacket, snow pants, gloves,",
    "Red, blue, beige, black, yellow, pink, green,",
    "That breed does",
    "Doesn't want to",
    "Nobody wants to",
    "Can carry around with",
    "I don't know who person is",
    "To put bike on",
    "Picture was played with",
}


def has_image_path(item):
    image = item.get("image", "")
    return isinstance(image, str) and image.strip() != ""


def strip_image_token(text):
    if not isinstance(text, str):
        return ""
    return text.replace("<image>", "").strip()


def needs_image(question, sample_has_image_context):
    if sample_has_image_context:
        return True
    return IMAGE_CUE_RE.search(question) is not None


def make_single_turn_item(sample_id, image, turn_index, question, answer):
    clean_question = strip_image_token(question)
    if clean_question:
        clean_question = "<image>\n" + clean_question
    else:
        clean_question = "<image>"
    return {
        "id": f"{sample_id}_t{turn_index}",
        "image": image,
        "conversations": [
            {"from": "human", "value": clean_question},
            {"from": "gpt", "value": answer},
        ],
    }


def iter_single_turn_items(data, max_answer_chars, stats):
    for sample in data:
        conversations = sample.get("conversations", [])
        if not conversations:
            continue

        if not has_image_path(sample):
            stats["dropped_no_image_path"] += 1
            continue

        sample_id = sample.get("id", "unknown")
        image = sample.get("image", "")

        # Follow-up turns inherit image context after splitting a multi-turn dialogue.
        sample_has_image_context = any(
            msg.get("from", "").lower() == "human"
            and IMAGE_CUE_RE.search(msg.get("value", ""))
            for msg in conversations
        )

        pair_count = len(conversations) // 2
        if len(conversations) % 2 != 0:
            stats["dropped_incomplete_last_turn"] += 1

        for turn_idx in range(pair_count):
            q = conversations[2 * turn_idx]
            a = conversations[2 * turn_idx + 1]
            stats["input_turn_pairs"] += 1

            if q.get("from", "").lower() != "human" or a.get("from", "").lower() != "gpt":
                stats["dropped_role_mismatch"] += 1
                continue

            question = q.get("value", "").strip()
            answer = a.get("value", "").strip()

            if OCR_RE.search(question) or OCR_RE.search(answer):
                stats["dropped_ocr_related"] += 1
                continue

            if len(answer) > max_answer_chars:
                stats["dropped_answer_too_long"] += 1
                continue

            if not needs_image(question, sample_has_image_context):
                stats["dropped_not_image_required"] += 1
                continue

            stats["kept_turn_pairs"] += 1
            yield make_single_turn_item(sample_id, image, turn_idx, question, answer)


def normalize_token(token):
    return token.lower().strip(",.;:!?\"'")


def get_answer(item):
    conversations = item.get("conversations", [])
    for msg in conversations:
        if msg.get("from", "").lower() == "gpt":
            return msg.get("value", "")
    return ""


def canonical_hash(item):
    payload = json.dumps(
        item,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def load_drop_hashes(path):
    if not path:
        return set()
    drop_path = Path(path)
    with drop_path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip() and not line.startswith("#")}


def has_dropped_image_prefix(item, prefixes):
    image = item.get("image", "")
    return any(isinstance(image, str) and image.startswith(prefix) for prefix in prefixes)


def is_conservative_truncated(answer):
    text = answer.strip()
    if text in EXACT_BAD_ANSWERS:
        return True

    if text.endswith(","):
        return True

    words = text.split()
    if 2 <= len(words) <= 8 and text[-1:] not in ".!?":
        first = normalize_token(words[0])
        last = normalize_token(words[-1])
        if last in {"a", "an", "the"}:
            return True
        if first in LEADING_TOKENS and last in BAD_END_TOKENS:
            return True

    return False


def is_aggressive_truncated(answer):
    text = answer.strip()
    if is_conservative_truncated(text):
        return True

    if text in AGGRESSIVE_EXACT_BAD_ANSWERS:
        return True

    return False


def is_truncated(answer, mode):
    if mode == "conservative":
        return is_conservative_truncated(answer)
    if mode == "aggressive":
        return is_aggressive_truncated(answer)
    raise ValueError(f"Unsupported mode: {mode}")


def init_preprocess_stats(input_count):
    return {
        "input_samples": input_count,
        "input_turn_pairs": 0,
        "dropped_no_image_path": 0,
        "dropped_role_mismatch": 0,
        "dropped_ocr_related": 0,
        "dropped_answer_too_long": 0,
        "dropped_not_image_required": 0,
        "dropped_incomplete_last_turn": 0,
        "kept_turn_pairs": 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build the strict no-OCR LLaVA-v1.5 pretraining JSON from "
            "llava_v1_5_mix665k.json in one pass."
        )
    )
    parser.add_argument(
        "--input",
        default="data/llava_stage1/llava_v1_5_mix665k.json",
        help="Path to raw llava_v1_5_mix665k.json.",
    )
    parser.add_argument(
        "--output",
        default="data/pretrain_llava_v1_5_mix665k_single_noocr_max200_imageonly_strict_noocr.json",
        help="Path to write the final strict no-OCR JSON list.",
    )
    parser.add_argument(
        "--stats-output",
        default="data/pretrain_llava_v1_5_mix665k_single_noocr_max200_imageonly_strict_noocr.stats.json",
        help="Path to write combined preprocessing and filtering stats.",
    )
    parser.add_argument("--max-answer-chars", type=int, default=200)
    parser.add_argument(
        "--mode",
        choices=["conservative", "aggressive"],
        default="aggressive",
        help="Truncated-answer filtering strictness.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=20,
        help="How many removed truncated examples to store in stats.",
    )
    parser.add_argument(
        "--drop-image-prefix",
        action="append",
        default=[],
        help="Drop samples whose image path starts with this prefix. Can be repeated.",
    )
    parser.add_argument(
        "--drop-hashes",
        default="code/data_tools/llava_v1_5_strict_noocr_drop_hashes.txt",
        help="Newline-separated canonical sample hashes to drop.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    stats_output_path = Path(args.stats_output) if args.stats_output else None

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    preprocess_stats = init_preprocess_stats(len(data))
    drop_hashes = load_drop_hashes(args.drop_hashes)

    kept = []
    removed_examples = []
    removed_answer_counts = {}
    removed_by_reason = {
        "image_prefix": 0,
        "drop_hash": 0,
        "truncated": 0,
    }
    filter_input_count = 0

    for item in iter_single_turn_items(data, args.max_answer_chars, preprocess_stats):
        filter_input_count += 1

        if has_dropped_image_prefix(item, args.drop_image_prefix):
            removed_by_reason["image_prefix"] += 1
            continue

        item_hash = canonical_hash(item)
        if item_hash in drop_hashes:
            removed_by_reason["drop_hash"] += 1
            continue

        answer = get_answer(item)
        if is_truncated(answer, args.mode):
            removed_by_reason["truncated"] += 1
            removed_answer_counts[answer] = removed_answer_counts.get(answer, 0) + 1
            if len(removed_examples) < args.max_examples:
                removed_examples.append(
                    {
                        "id": item.get("id"),
                        "image": item.get("image"),
                        "answer": answer,
                        "question": item.get("conversations", [{}])[0].get("value", ""),
                    }
                )
            continue

        kept.append(item)

    filter_stats = {
        "mode": args.mode,
        "input_count": filter_input_count,
        "output_count": len(kept),
        "removed_count": filter_input_count - len(kept),
        "removed_by_reason": removed_by_reason,
        "drop_image_prefix": args.drop_image_prefix,
        "drop_hashes_file": str(Path(args.drop_hashes)) if args.drop_hashes else "",
        "drop_hash_count": len(drop_hashes),
        "removed_answer_counts": dict(
            sorted(removed_answer_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        "removed_examples": removed_examples,
    }
    stats = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "preprocess": preprocess_stats,
        "filter": filter_stats,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False)

    if stats_output_path is not None:
        stats_output_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_output_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"output: {output_path}")
    if stats_output_path is not None:
        print(f"stats: {stats_output_path}")


if __name__ == "__main__":
    main()
