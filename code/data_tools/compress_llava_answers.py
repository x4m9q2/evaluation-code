import argparse
import json
import multiprocessing as mp
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import torch

# This environment ships a DeepSpeed/accelerate stack that expects
# torch.amp.custom_fwd/custom_bwd to exist, while torch 2.3.1 only exposes
# the CUDA AMP variants. Patch before importing transformers.
if hasattr(torch, "amp") and hasattr(torch, "cuda") and hasattr(torch.cuda, "amp"):
    if not hasattr(torch.amp, "custom_fwd") and hasattr(torch.cuda.amp, "custom_fwd"):
        def _custom_fwd(*args, **kwargs):
            kwargs.pop("device_type", None)
            return torch.cuda.amp.custom_fwd(*args, **kwargs)

        torch.amp.custom_fwd = _custom_fwd

    if not hasattr(torch.amp, "custom_bwd") and hasattr(torch.cuda.amp, "custom_bwd"):
        def _custom_bwd(*args, **kwargs):
            kwargs.pop("device_type", None)
            return torch.cuda.amp.custom_bwd(*args, **kwargs)

        torch.amp.custom_bwd = _custom_bwd

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from match_shortcuts_full import iter_wrapped_json_array
from prepare_llava_shortcut_inputs import normalize_text


TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
QUESTION_FAMILY_BBOX_COORD = "bbox_coord"
QUESTION_FAMILY_REGION_DESC = "region_desc"
QUESTION_FAMILY_OTHER = "other"
BBOX_COORD_PROMPT_PREFIX = "please provide the bounding box coordinate of the region this sentence describes:"
REGION_DESC_PROMPT_PREFIX = "please provide a short description for this region:"
QUESTION_AUXILIARIES = {
    "is",
    "are",
    "was",
    "were",
    "do",
    "does",
    "did",
    "can",
    "could",
    "will",
    "would",
    "should",
    "has",
    "have",
    "had",
}
CHOICE_EDGE_TRIM_WORDS = {
    "a",
    "an",
    "answer",
    "choices",
    "choice",
    "directly",
    "given",
    "here",
    "image",
    "letter",
    "option",
    "options",
    "phrase",
    "photo",
    "picture",
    "question",
    "scene",
    "shown",
    "single",
    "that",
    "the",
    "there",
    "these",
    "this",
    "those",
    "using",
    "visible",
    "word",
}
CHOICE_TRAILING_CONTEXT_WORDS = {
    "around",
    "as",
    "at",
    "behind",
    "beneath",
    "beside",
    "between",
    "by",
    "during",
    "for",
    "from",
    "in",
    "inside",
    "near",
    "of",
    "on",
    "outside",
    "over",
    "to",
    "toward",
    "under",
    "with",
}
YESNO_OR_PREFIXES = (
    "are there any ",
    "can you see any ",
    "do you see any ",
    "does the image show any ",
    "is there any ",
)
COLOR_WORDS = {
    "black",
    "blue",
    "brown",
    "gold",
    "gray",
    "green",
    "grey",
    "orange",
    "pink",
    "purple",
    "red",
    "silver",
    "tan",
    "teal",
    "white",
    "yellow",
    "beige",
    "maroon",
    "navy",
    "rainbow",
    "blonde",
}
ROOM_PHRASES = {
    "bathroom",
    "bedroom",
    "dining room",
    "garage",
    "kitchen",
    "living room",
    "office",
}
MATERIAL_WORDS = {
    "brick",
    "cement",
    "ceramic",
    "cloth",
    "concrete",
    "glass",
    "leather",
    "metal",
    "paper",
    "plastic",
    "stone",
    "tile",
    "wood",
    "wooden",
}
LOCATION_WORDS = {
    "above",
    "across",
    "around",
    "at",
    "behind",
    "below",
    "beneath",
    "beside",
    "between",
    "by",
    "in",
    "inside",
    "near",
    "next",
    "off",
    "on",
    "outside",
    "over",
    "under",
}
LOCATION_NOUN_WORDS = {
    "back",
    "background",
    "bottom",
    "center",
    "centre",
    "foreground",
    "front",
    "left",
    "middle",
    "right",
    "side",
    "top",
}
LOCATION_TAIL_SUFFIXES = (
    ("on", "the", "side"),
    ("at", "the", "side"),
    ("by", "the", "side"),
    ("to", "the", "side"),
    ("on", "side"),
    ("at", "side"),
    ("by", "side"),
)
CLAUSE_WORDS = {
    "am",
    "are",
    "be",
    "been",
    "being",
    "has",
    "have",
    "had",
    "is",
    "was",
    "were",
}
STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "both",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "just",
    "many",
    "much",
    "of",
    "on",
    "or",
    "some",
    "that",
    "the",
    "their",
    "there",
    "these",
    "this",
    "those",
    "to",
    "which",
    "while",
    "with",
}
NUMBER_ALIASES = {
    "zero": "0",
    "none": "0",
    "no": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
}


@dataclass(frozen=True)
class AnswerSpan:
    text: str
    tokens: Tuple[str, ...]
    canon_tokens: Tuple[str, ...]
    is_yesno: bool
    is_number: bool
    is_color: bool
    is_room: bool
    is_material: bool


PostprocessTask = Tuple[int, int, int, str, str, str, int, int, int, str]


@dataclass
class PreparedBatch:
    batch_rows: List[Optional[Dict]]
    active_batch: List[Tuple[int, Dict, Dict]]
    active_indices: List[int]
    active_tasks: List[PostprocessTask]


@dataclass
class PendingBatch:
    batch_rows: List[Optional[Dict]]
    active_indices: List[int]
    postprocess_results: Optional[Iterable[Dict]]


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(normalize_text(text).lower())


def canonical_token(token: str) -> str:
    if token.isdigit():
        stripped = token.lstrip("0")
        return stripped or "0"
    return NUMBER_ALIASES.get(token, token)


def canonicalize_tokens(tokens: Sequence[str]) -> Tuple[str, ...]:
    return tuple(canonical_token(token) for token in tokens)


def detect_question_family(question_text: str) -> str:
    if question_text.startswith(BBOX_COORD_PROMPT_PREFIX):
        return QUESTION_FAMILY_BBOX_COORD
    if question_text.startswith(REGION_DESC_PROMPT_PREFIX):
        return QUESTION_FAMILY_REGION_DESC
    return QUESTION_FAMILY_OTHER


def should_skip_question_family(
    question_family: str,
    skip_region_desc: bool,
    include_bbox_coord: bool,
) -> bool:
    if question_family == QUESTION_FAMILY_BBOX_COORD and not include_bbox_coord:
        return True
    if question_family == QUESTION_FAMILY_REGION_DESC and skip_region_desc:
        return True
    return False


def is_yesno_or_prompt(question_text: str) -> bool:
    padded = f" {question_text} "
    if " either " in padded or question_text.startswith(YESNO_OR_PREFIXES):
        return True
    if question_text.startswith(("is there ", "are there ", "do you see ", "can you see ")) and " any " in padded:
        return True
    return False


def detect_question_type(question_tokens: Sequence[str], question_text: str) -> str:
    if question_text.startswith("where ") or " position " in f" {question_text} " or " located " in f" {question_text} ":
        return "location"
    if question_text.startswith("can you ") or question_text.startswith("could you ") or question_text.startswith("would you "):
        return "other"
    if question_tokens and question_tokens[0] in QUESTION_AUXILIARIES:
        if " or " in f" {question_text} " and not is_yesno_or_prompt(question_text):
            return "choice"
        return "yesno"
    if " or " in f" {question_text} ":
        return "choice"
    if question_text.startswith("how many") or " number " in f" {question_text} ":
        return "count"
    if " color " in f" {question_text} " or " colour " in f" {question_text} ":
        return "color"
    if " room " in f" {question_text} ":
        return "room"
    if " material " in f" {question_text} " or " made of " in f" {question_text} ":
        return "material"
    return "other"


def normalize_choice_option(option_tokens: Sequence[str], max_option_tokens: int = 4) -> Tuple[str, ...]:
    tokens = list(option_tokens)
    while tokens and tokens[0] in CHOICE_EDGE_TRIM_WORDS:
        tokens.pop(0)
    while tokens and tokens[-1] in CHOICE_EDGE_TRIM_WORDS:
        tokens.pop()
    while tokens and tokens[-1] in CHOICE_TRAILING_CONTEXT_WORDS:
        tokens.pop()
        while tokens and tokens[-1] in CHOICE_EDGE_TRIM_WORDS:
            tokens.pop()

    if not tokens or len(tokens) > max_option_tokens:
        return ()
    if all(token in STOPWORDS for token in tokens):
        return ()
    return tuple(tokens)


def extract_choice_options(question_tokens: Sequence[str]) -> List[Tuple[str, ...]]:
    options: List[Tuple[str, ...]] = []
    for index, token in enumerate(question_tokens):
        if token != "or":
            continue

        left_window = question_tokens[max(0, index - 4) : index]
        right_window = question_tokens[index + 1 : index + 5]

        for size in range(1, len(left_window) + 1):
            option_tokens = normalize_choice_option(left_window[-size:])
            if option_tokens:
                options.append(option_tokens)

        for size in range(1, len(right_window) + 1):
            option_tokens = normalize_choice_option(right_window[:size])
            if option_tokens:
                options.append(option_tokens)

    deduped: List[Tuple[str, ...]] = []
    seen = set()
    for option in sorted(options, key=lambda item: (len(item), item)):
        if option not in seen:
            deduped.append(option)
            seen.add(option)
    return deduped


def score_choice_match(span_canon_tokens: Sequence[str], choice_options: Sequence[Tuple[str, ...]]) -> float:
    span_tokens = tuple(span_canon_tokens)
    span_content = tuple(token for token in span_tokens if token not in STOPWORDS)
    if not span_tokens:
        return 0.0

    best = 0.0
    for option in choice_options:
        option_canon = canonicalize_tokens(option)
        option_content = tuple(token for token in option_canon if token not in STOPWORDS)
        if not option_canon:
            continue

        if span_tokens == option_canon:
            return 50.0 + 5.0 * len(option_canon)

        if len(span_tokens) >= 2:
            if span_tokens == option_canon[: len(span_tokens)]:
                best = max(best, 24.0)
            if span_tokens == option_canon[-len(span_tokens) :]:
                best = max(best, 24.0)

        if not span_content or not option_content:
            continue
        shared = len(set(span_content) & set(option_content))
        if shared >= 2:
            best = max(best, 12.0 + 4.0 * shared)

    return best


def is_numeric_location_fragment(tokens: Sequence[str]) -> bool:
    if len(tokens) < 2 or not tokens[0].isdigit():
        return False

    tail_tokens = [token for token in tokens[1:] if token not in STOPWORDS]
    if not tail_tokens:
        return False
    if not any(token in LOCATION_WORDS or token in LOCATION_NOUN_WORDS for token in tail_tokens):
        return False
    return all(token in LOCATION_WORDS or token in LOCATION_NOUN_WORDS for token in tail_tokens)


def trailing_location_suffix_length(tokens: Sequence[str]) -> int:
    token_tuple = tuple(tokens)
    for suffix in LOCATION_TAIL_SUFFIXES:
        if len(token_tuple) <= len(suffix):
            continue
        if token_tuple[-len(suffix) :] != suffix:
            continue

        prefix_tokens = [
            token
            for token in token_tuple[: -len(suffix)]
            if token not in STOPWORDS and token not in CLAUSE_WORDS
        ]
        if prefix_tokens:
            return len(suffix)
    return 0


def is_location_only_fragment(tokens: Sequence[str]) -> bool:
    content_tokens = [
        token
        for token in tokens
        if token not in STOPWORDS and token not in CLAUSE_WORDS
    ]
    if not content_tokens:
        return False
    return all(token in LOCATION_WORDS or token in LOCATION_NOUN_WORDS for token in content_tokens)


def build_prompt(question: str, answer: str) -> str:
    return (
        "Extract the shortest answer span from the Answer that directly answers the Question.\n"
        "Rules:\n"
        "- Copy words only from the Answer.\n"
        "- Do not paraphrase.\n"
        "- Prefer 1 to 4 words.\n"
        "- Return only the answer span.\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        "Answer span:"
    )


def build_answer_spans(answer_text: str, max_span_tokens: int) -> List[AnswerSpan]:
    answer_tokens = tokenize(answer_text)
    spans: List[AnswerSpan] = []
    seen = set()
    for start in range(len(answer_tokens)):
        for length in range(1, min(max_span_tokens, len(answer_tokens) - start) + 1):
            tokens = tuple(answer_tokens[start : start + length])
            if not tokens or all(token in STOPWORDS for token in tokens):
                continue
            if is_numeric_location_fragment(tokens):
                continue
            text = " ".join(tokens)
            if text in seen:
                continue
            seen.add(text)
            canon_tokens = canonicalize_tokens(tokens)
            spans.append(
                AnswerSpan(
                    text=text,
                    tokens=tokens,
                    canon_tokens=canon_tokens,
                    is_yesno=text in {"yes", "no"},
                    is_number=all(token.isdigit() or canonical_token(token).isdigit() for token in tokens),
                    is_color=all(token in COLOR_WORDS or token == "and" for token in tokens)
                    and any(token in COLOR_WORDS for token in tokens),
                    is_room=text in ROOM_PHRASES,
                    is_material=any(token in MATERIAL_WORDS for token in tokens),
                )
            )
    return spans


def f1_overlap(lhs: Sequence[str], rhs: Sequence[str]) -> float:
    lhs_set = set(lhs)
    rhs_set = set(rhs)
    if not lhs_set or not rhs_set:
        return 0.0
    shared = len(lhs_set & rhs_set)
    if shared == 0:
        return 0.0
    precision = shared / len(lhs_set)
    recall = shared / len(rhs_set)
    return 2.0 * precision * recall / (precision + recall)


def choose_span(
    question_text: str,
    answer_text: str,
    model_output: str,
    max_span_tokens: int,
) -> Tuple[str, str, float]:
    question_norm = normalize_text(question_text).lower()
    question_tokens = tokenize(question_norm)
    question_type = detect_question_type(question_tokens, question_norm)
    choice_options = extract_choice_options(question_tokens)
    question_token_set = set(question_tokens)

    candidate_tokens = tuple(tokenize(model_output))
    candidate_canon = canonicalize_tokens(candidate_tokens)
    candidate_is_compliant = 0 < len(candidate_tokens) <= max_span_tokens
    spans = build_answer_spans(answer_text, max_span_tokens=max_span_tokens)
    if not spans:
        fallback = normalize_text(answer_text).lower()
        return fallback, "fallback_full_answer", 0.0

    best_span: Optional[AnswerSpan] = None
    best_score = float("-inf")
    best_method = "aligned_overlap"

    for span in spans:
        score = 0.0
        if candidate_tokens:
            if candidate_is_compliant and (span.tokens == candidate_tokens or span.canon_tokens == candidate_canon):
                score += 100.0
            score += 30.0 * f1_overlap(span.canon_tokens, candidate_canon)

        if question_type == "yesno" and span.is_yesno:
            score += 45.0
        elif question_type == "count" and span.is_number:
            score += 35.0
        elif question_type == "color" and span.is_color:
            score += 30.0
        elif question_type == "room" and span.is_room:
            score += 25.0
        elif question_type == "material" and span.is_material:
            score += 20.0
        elif question_type == "location" and span.tokens[0] in LOCATION_WORDS:
            score += 30.0

        choice_match_score = 0.0
        if question_type == "choice":
            choice_match_score = score_choice_match(span.canon_tokens, choice_options)
            score += choice_match_score

        if len(span.tokens) > 4:
            score -= 5.0
        score -= 0.35 * len(span.tokens)
        if span.tokens[-1] in STOPWORDS:
            score -= 18.0
        if span.tokens[0] in {"a", "an", "as", "it", "the", "there", "they"}:
            score -= 4.0
        if any(token in CLAUSE_WORDS for token in span.tokens):
            score -= 12.0

        new_content_tokens = [
            token
            for token in span.tokens
            if token not in STOPWORDS and token not in question_token_set and token not in CLAUSE_WORDS
        ]
        if new_content_tokens:
            score += 4.0 * len(new_content_tokens)
            if span.tokens[-1] in question_token_set and span.tokens[-1] not in STOPWORDS:
                score += 8.0
        elif any(token not in STOPWORDS for token in span.tokens):
            score -= 6.0

        if all(token in STOPWORDS for token in span.tokens):
            score -= 25.0

        content_tokens = [
            token
            for token in span.tokens
            if token not in STOPWORDS and token not in CLAUSE_WORDS
        ]
        if len(content_tokens) >= 2 and all(token in question_token_set for token in content_tokens):
            score -= 16.0

        if question_type == "choice" and choice_options and choice_match_score <= 0.0:
            if content_tokens and all(token in question_token_set for token in content_tokens):
                score -= 18.0
            else:
                score -= 9.0

        if question_type != "location" and is_location_only_fragment(span.tokens):
            score -= 28.0

        location_tail_len = trailing_location_suffix_length(span.tokens)
        if question_type != "location" and location_tail_len > 0:
            score -= 12.0 + 4.0 * location_tail_len

        if score > best_score:
            best_score = score
            best_span = span

    assert best_span is not None
    if candidate_is_compliant and (best_span.tokens == candidate_tokens or best_span.canon_tokens == candidate_canon):
        best_method = "aligned_exact"
    elif question_type != "other":
        best_method = f"aligned_{question_type}"

    return best_span.text, best_method, best_score


def load_model_and_tokenizer(
    model_name_or_path: str,
    cache_dir: Optional[str],
    device: str,
    local_files_only: bool,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    torch_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        torch_dtype=torch_dtype,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def batched(iterable: Iterable[Tuple[int, Dict, Dict]], batch_size: int):
    batch: List[Tuple[int, Dict, Dict]] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def iter_pairs(
    questions_json: str,
    annotations_json: str,
    shard_id: int,
    num_shards: int,
    limit: Optional[int],
    skip: int = 0,
):
    matched = 0
    q_iter = iter_wrapped_json_array(questions_json, "questions")
    a_iter = iter_wrapped_json_array(annotations_json, "annotations")
    for index, (question, annotation) in enumerate(zip(q_iter, a_iter)):
        if index % num_shards != shard_id:
            continue
        if limit is not None and matched >= limit:
            return
        if matched < skip:
            matched += 1
            continue
        yield index, question, annotation
        matched += 1


def count_jsonl_lines(path: str) -> int:
    count = 0
    with open(path, "rb") as fp:
        while True:
            chunk = fp.read(1 << 20)
            if not chunk:
                break
            count += chunk.count(b"\n")
    return count


def truncate_partial_jsonl_line(path: str) -> None:
    with open(path, "rb+") as fp:
        fp.seek(0, os.SEEK_END)
        file_size = fp.tell()
        if file_size == 0:
            return

        fp.seek(file_size - 1)
        if fp.read(1) == b"\n":
            return

        position = file_size - 1
        while position >= 0:
            fp.seek(position)
            if fp.read(1) == b"\n":
                fp.truncate(position + 1)
                return
            position -= 1

        fp.truncate(0)


def resolve_resume_count(output_path: str, resume: bool) -> int:
    if not resume or not os.path.exists(output_path):
        return 0
    truncate_partial_jsonl_line(output_path)
    return count_jsonl_lines(output_path)


def generate_candidates(
    batch: Sequence[Tuple[int, Dict, Dict]],
    tokenizer,
    model,
    device: str,
    max_input_length: int,
    max_new_tokens: int,
    num_beams: int,
) -> List[str]:
    prompts = []
    for _, question, annotation in batch:
        answers = annotation.get("answers") or []
        if answers:
            answer_text = answers[0].get("answer", "")
        else:
            answer_text = annotation.get("multiple_choice_answer", "")
        prompts.append(build_prompt(question.get("question", ""), answer_text))

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
        )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def build_output_row(
    global_index: int,
    question: Dict,
    answer_text: str,
    model_output: str,
    short_answer: str,
    selection_method: str,
    selection_score: float,
    shard_id: int,
    num_shards: int,
    question_family: str,
) -> Dict:
    return {
        "global_index": global_index,
        "question_id": int(question.get("question_id", 0)),
        "image_id": int(question.get("image_id", 0)),
        "question": question.get("question", ""),
        "answer": answer_text,
        "model_output": normalize_text(model_output).lower(),
        "short_answer": short_answer,
        "selection_method": selection_method,
        "selection_score": selection_score,
        "question_family": question_family,
        "shard_id": shard_id,
        "num_shards": num_shards,
    }


def build_output_row_from_values(
    global_index: int,
    question_id: int,
    image_id: int,
    question_text: str,
    answer_text: str,
    model_output: str,
    short_answer: str,
    selection_method: str,
    selection_score: float,
    shard_id: int,
    num_shards: int,
    question_family: str,
) -> Dict:
    return {
        "global_index": global_index,
        "question_id": question_id,
        "image_id": image_id,
        "question": question_text,
        "answer": answer_text,
        "model_output": normalize_text(model_output).lower(),
        "short_answer": short_answer,
        "selection_method": selection_method,
        "selection_score": selection_score,
        "question_family": question_family,
        "shard_id": shard_id,
        "num_shards": num_shards,
    }


def postprocess_active_row(task: Tuple[int, int, int, str, str, str, int, int, int, str]) -> Dict:
    (
        global_index,
        question_id,
        image_id,
        question_text,
        answer_text,
        model_output,
        max_span_tokens,
        shard_id,
        num_shards,
        question_family,
    ) = task
    short_answer, method, score = choose_span(
        question_text=question_text,
        answer_text=answer_text,
        model_output=model_output,
        max_span_tokens=max_span_tokens,
    )
    return build_output_row_from_values(
        global_index=global_index,
        question_id=question_id,
        image_id=image_id,
        question_text=question_text,
        answer_text=answer_text,
        model_output=model_output,
        short_answer=short_answer,
        selection_method=method,
        selection_score=score,
        shard_id=shard_id,
        num_shards=num_shards,
        question_family=question_family,
    )


def make_postprocess_executor(num_workers: int) -> Optional[ProcessPoolExecutor]:
    if num_workers <= 0:
        return None
    return ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=mp.get_context("spawn"),
    )


def prepare_batch(
    batch: Sequence[Tuple[int, Dict, Dict]],
    *,
    skip_region_desc: bool,
    include_bbox_coord: bool,
    max_span_tokens: int,
    shard_id: int,
    num_shards: int,
) -> PreparedBatch:
    active_batch: List[Tuple[int, Dict, Dict]] = []
    active_tasks: List[PostprocessTask] = []
    active_indices = []
    batch_rows: List[Optional[Dict]] = [None] * len(batch)

    for index, (global_index, question, annotation) in enumerate(batch):
        question_text = question.get("question", "")
        question_text_norm = normalize_text(question_text).lower()
        question_family = detect_question_family(question_text_norm)
        answers = annotation.get("answers") or []
        if answers:
            answer_text = answers[0].get("answer", "")
        else:
            answer_text = annotation.get("multiple_choice_answer", "")

        if should_skip_question_family(
            question_family,
            skip_region_desc=skip_region_desc,
            include_bbox_coord=include_bbox_coord,
        ):
            batch_rows[index] = build_output_row(
                global_index=global_index,
                question=question,
                answer_text=answer_text,
                model_output="",
                short_answer="",
                selection_method=f"skipped_{question_family}",
                selection_score=-1.0,
                shard_id=shard_id,
                num_shards=num_shards,
                question_family=question_family,
            )
            continue

        active_batch.append((global_index, question, annotation))
        active_indices.append(index)
        active_tasks.append(
            (
                global_index,
                int(question.get("question_id", 0)),
                int(question.get("image_id", 0)),
                question_text,
                answer_text,
                "",
                max_span_tokens,
                shard_id,
                num_shards,
                question_family,
            )
        )

    return PreparedBatch(
        batch_rows=batch_rows,
        active_batch=active_batch,
        active_indices=active_indices,
        active_tasks=active_tasks,
    )


def attach_model_outputs(
    active_tasks: Sequence[PostprocessTask],
    model_outputs: Sequence[str],
) -> List[PostprocessTask]:
    tasks_with_outputs: List[PostprocessTask] = []
    for task, model_output in zip(active_tasks, model_outputs):
        task_list = list(task)
        task_list[5] = model_output
        tasks_with_outputs.append(tuple(task_list))
    return tasks_with_outputs


def submit_postprocess(
    prepared: PreparedBatch,
    tasks_with_outputs: Sequence[PostprocessTask],
    executor: Optional[ProcessPoolExecutor],
    num_workers: int,
) -> PendingBatch:
    if executor is not None and tasks_with_outputs:
        chunksize = max(1, len(tasks_with_outputs) // (num_workers * 4))
        results = executor.map(
            postprocess_active_row,
            tasks_with_outputs,
            chunksize=chunksize,
        )
    elif tasks_with_outputs:
        results = [postprocess_active_row(task) for task in tasks_with_outputs]
    else:
        results = []

    return PendingBatch(
        batch_rows=prepared.batch_rows,
        active_indices=prepared.active_indices,
        postprocess_results=results,
    )


def resolve_pending_batch(pending: PendingBatch) -> List[Dict]:
    resolved_rows = list(pending.postprocess_results or [])
    for batch_index, row in zip(pending.active_indices, resolved_rows):
        pending.batch_rows[batch_index] = row
    for row in pending.batch_rows:
        assert row is not None
    return pending.batch_rows  # type: ignore[return-value]


def log_progress(
    *,
    processed: int,
    resumed: int,
    session_processed: int,
    start_time: float,
    output_path: str,
    done: bool = False,
) -> None:
    elapsed = max(time.time() - start_time, 1e-6)
    resume_fragment = f" resumed={resumed}" if resumed else ""
    prefix = "done " if done else ""
    print(
        f"{prefix}processed={processed}{resume_fragment} elapsed_sec={elapsed:.1f} "
        f"items_per_sec={session_processed / elapsed:.2f} output={output_path}",
        flush=True,
    )


def run(args):
    if args.hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
    if args.cache_dir:
        os.environ.setdefault("HF_HOME", args.cache_dir)

    tokenizer, model = load_model_and_tokenizer(
        model_name_or_path=args.model_name_or_path,
        cache_dir=args.cache_dir,
        device=args.device,
        local_files_only=args.local_files_only,
    )

    output_path = os.path.abspath(args.output_jsonl)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    resumed = resolve_resume_count(output_path, args.resume)
    if resumed and args.limit is not None and resumed >= args.limit:
        print(
            f"output already has {resumed} rows, which meets/exceeds limit={args.limit}; "
            f"nothing to do for {output_path}",
            flush=True,
        )
        return

    start_time = time.time()
    processed = resumed
    session_processed = 0
    if resumed:
        print(f"resuming output={output_path} existing_rows={resumed}", flush=True)

    postprocess_executor = make_postprocess_executor(args.postprocess_workers)
    try:
        file_mode = "a" if resumed else "w"
        next_log_at = None
        if args.log_every > 0:
            next_log_at = ((processed // args.log_every) + 1) * args.log_every

        with open(output_path, file_mode, encoding="utf-8") as out_fp:
            iterator = iter_pairs(
                questions_json=args.questions_json,
                annotations_json=args.annotations_json,
                shard_id=args.shard_id,
                num_shards=args.num_shards,
                limit=args.limit,
                skip=resumed,
            )
            pending_batch: Optional[PendingBatch] = None
            for batch in batched(iterator, args.batch_size):
                prepared = prepare_batch(
                    batch,
                    skip_region_desc=args.skip_region_desc,
                    include_bbox_coord=args.include_bbox_coord,
                    max_span_tokens=args.max_span_tokens,
                    shard_id=args.shard_id,
                    num_shards=args.num_shards,
                )

                model_outputs: List[str] = []
                if prepared.active_batch:
                    model_outputs = generate_candidates(
                        batch=prepared.active_batch,
                        tokenizer=tokenizer,
                        model=model,
                        device=args.device,
                        max_input_length=args.max_input_length,
                        max_new_tokens=args.max_new_tokens,
                        num_beams=args.num_beams,
                    )

                tasks_with_outputs = attach_model_outputs(prepared.active_tasks, model_outputs)
                current_pending = submit_postprocess(
                    prepared=prepared,
                    tasks_with_outputs=tasks_with_outputs,
                    executor=postprocess_executor,
                    num_workers=args.postprocess_workers,
                )

                if pending_batch is not None:
                    resolved_batch_rows = resolve_pending_batch(pending_batch)
                    for row in resolved_batch_rows:
                        out_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                    batch_count = len(resolved_batch_rows)
                    processed += batch_count
                    session_processed += batch_count
                    if next_log_at is not None:
                        while processed >= next_log_at:
                            log_progress(
                                processed=processed,
                                resumed=resumed,
                                session_processed=session_processed,
                                start_time=start_time,
                                output_path=output_path,
                            )
                            next_log_at += args.log_every

                pending_batch = current_pending

            if pending_batch is not None:
                resolved_batch_rows = resolve_pending_batch(pending_batch)
                for row in resolved_batch_rows:
                    out_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                batch_count = len(resolved_batch_rows)
                processed += batch_count
                session_processed += batch_count
                if next_log_at is not None:
                    while processed >= next_log_at:
                        log_progress(
                            processed=processed,
                            resumed=resumed,
                            session_processed=session_processed,
                            start_time=start_time,
                            output_path=output_path,
                        )
                        next_log_at += args.log_every
    finally:
        if postprocess_executor is not None:
            postprocess_executor.shutdown(wait=True)

    log_progress(
        processed=processed,
        resumed=resumed,
        session_processed=session_processed,
        start_time=start_time,
        output_path=output_path,
        done=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions-json",
        default="data/shortcut_inputs/llava_mix665k_single_noocr/questions.json",
    )
    parser.add_argument(
        "--annotations-json",
        default="data/shortcut_inputs/llava_mix665k_single_noocr/annotations.json",
    )
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--model-name-or-path", default="google/flan-t5-small")
    parser.add_argument("--cache-dir", default=".cache/huggingface")
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-input-length", type=int, default=192)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-span-tokens", type=int, default=5)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--log-every", type=int, default=512)
    parser.add_argument("--skip-region-desc", action="store_true")
    parser.add_argument("--include-bbox-coord", action="store_true")
    parser.add_argument("--postprocess-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError(f"shard_id must be in [0, {args.num_shards})")

    run(args)


if __name__ == "__main__":
    main()
