#!/usr/bin/env python3
"""Prepare OpenAI Batch API JSONL payloads for cross-modality QA generation."""

import argparse
import base64
import json
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPE_ROOT = REPO_ROOT / "data" / "shortcut_pipeline"

PROMPT_TEMPLATE = """You are a meticulous VQA exam designer tasked with creating challenging question-answer pairs.

Core task:
- Generate one new QA pair where the question includes ALL keywords: {keywords}
- Both question and answer must be based ONLY on visible (non-black) regions of the provided masked image
- The answer must keep the same answer type as the original answer type: {answer_type}
- The original answer is: {original_answer}
- The new answer must use DIFFERENT content from the original answer

Critical constraints:
1. Answer differentiation:
   - Original "yes" -> Your answer "no"
   - Original "red" -> Your answer "blue" (not crimson/scarlet)
   - Original "3" -> Your answer "5" (not three/03)
2. Visual dependency:
   - Question must require inspecting specific image details
   - Cannot be answered from keywords or common sense alone
3. Quality standards:
   - Avoid simple template questions
   - Add contextual conditions for specificity
   - Question: 8-20 words, Answer: concise 1-10 words

Output format: Return ONLY a JSON object:
{{
  "question": "Your question containing all keywords",
  "answer": "Answer grounded purely in visible content and satisfying the constraints"
}}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-json",
        "--split-json",
        dest="input_json",
        default=str(PIPE_ROOT / "cross_modality_qa_input.json"),
        help="Path to the merged GQA-style JSON file.",
    )
    parser.add_argument(
        "--mask-root",
        default=str(PIPE_ROOT / "output_mask"),
        help="Directory containing the masked images. Flat and split-folder layouts are both supported.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=str(PIPE_ROOT / "batch_inputs" / "cross_modality_qa_requests.jsonl"),
        help="Destination JSONL file for Batch API requests.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Maximum number of requests to generate; <= 0 keeps all.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.4",
        help="Target model for the Batch API.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=400,
        help="max_output_tokens to include in each request body.",
    )
    parser.add_argument(
        "--include-unmatched",
        action="store_true",
        help="Keep rows even if no masked image can be found; a warning is printed and the row is skipped by default.",
    )
    return parser.parse_args()


def load_records(input_path: Path) -> List[Dict[str, Any]]:
    with input_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    items = payload.get("results", [])
    if not isinstance(items, list):
        raise RuntimeError(f"{input_path} does not contain a 'results' list.")
    return items


def encode_image(image_path: Path) -> str:
    with image_path.open("rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8")


def infer_answer_type(answer_type: Optional[str], original_answer: str) -> str:
    normalized = (answer_type or "").strip().lower()
    if normalized and normalized != "unknown":
        return normalized

    answer = (original_answer or "").strip().lower()
    yes_no = {"yes", "no", "true", "false"}
    if answer in yes_no:
        return "yes/no"
    if answer.isdigit():
        return "number"
    if answer:
        return "other"
    return "unknown"


def normalize_tokens(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    normalized: List[str] = []
    for value in values:
        text = str(value).strip()
        if text:
            normalized.append(text)
    return normalized


def build_prompt(text_keywords: List[str], original_answer: str, answer_type: str) -> str:
    keywords_str = ", ".join(text_keywords) if text_keywords else "(no keywords)"
    return PROMPT_TEMPLATE.format(
        keywords=keywords_str,
        original_answer=original_answer or "(empty)",
        answer_type=answer_type or "(unknown)",
    )


def build_request_body(
    prompt: str,
    image_b64: str,
    image_path: Path,
    model: str,
    max_output_tokens: int,
) -> Dict[str, Any]:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    mime_type = mime_type or "image/png"
    return {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime_type};base64,{image_b64}",
                        "detail": "low",
                    },
                ],
            }
        ],
        "max_output_tokens": max_output_tokens,
    }


def find_mask_file(mask_root: Path, question_id: Any, image_id: Any) -> Optional[Path]:
    """Find the most likely masked image for a record."""
    qid = "" if question_id is None else str(question_id)
    iid = "" if image_id is None else str(image_id)
    candidates: List[Path] = []
    search_roots: List[Path] = [mask_root]
    nested = mask_root / "output_mask"
    if nested.exists():
        search_roots.insert(0, nested)

    names: List[str] = []
    if qid and iid:
        names.extend(
            [
                f"{qid}_{iid}.png",
                f"{qid}_{iid}.jpg",
                f"{qid}_{iid}.jpeg",
                f"{qid}_{iid}.webp",
                f"{iid}_{qid}.png",
                f"{iid}_{qid}.jpg",
                f"{iid}_{qid}.jpeg",
                f"{iid}_{qid}.webp",
            ]
        )
    if qid:
        names.extend([f"{qid}.png", f"{qid}.jpg", f"{qid}.jpeg", f"{qid}.webp"])
    if iid:
        names.extend([f"{iid}.png", f"{iid}.jpg", f"{iid}.jpeg", f"{iid}.webp"])

    for root in search_roots:
        for name in names:
            candidate = root / name
            if candidate.exists():
                return candidate

    # Fall back to a narrow glob if exact names do not match.
    if qid and iid:
        pattern_tokens = [qid, iid]
        for root in search_roots:
            for candidate in sorted(root.glob("*")):
                if candidate.is_file():
                    candidate_name = candidate.name
                    if all(token in candidate_name for token in pattern_tokens):
                        candidates.append(candidate)

    if len(candidates) == 1:
        return candidates[0]
    return None


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_json).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    mask_root = Path(args.mask_root).resolve()
    if not mask_root.exists():
        raise FileNotFoundError(f"Mask root not found: {mask_root}")

    records = load_records(input_path)
    output_path = Path(args.output_jsonl).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped_missing_mask = 0
    skipped_ineligible = 0
    skipped_other = 0

    with output_path.open("w", encoding="utf-8") as out_f:
        for item in records:
            if args.limit > 0 and written >= args.limit:
                break

            qid = item.get("question_id", item.get("id"))
            image_id = item.get("image_id")
            text_keywords = normalize_tokens(item.get("text_keywords", []))
            visual_cues = normalize_tokens(item.get("visual_cues", []))
            original_answer = str(item.get("answer", "")).strip()
            answer_type = infer_answer_type(item.get("answer_type"), original_answer)

            # Stage 2 only uses rows that have both text and visual shortcut signals.
            if not text_keywords or not visual_cues:
                skipped_ineligible += 1
                continue

            image_path = find_mask_file(mask_root, qid, image_id)
            if image_path is None:
                skipped_missing_mask += 1
                if args.include_unmatched:
                    print(f"[warn] no masked image found for question_id={qid}, image_id={image_id}")
                continue

            try:
                image_b64 = encode_image(image_path)
                prompt = build_prompt(text_keywords, original_answer, answer_type)
                body = build_request_body(
                    prompt=prompt,
                    image_b64=image_b64,
                    image_path=image_path,
                    model=args.model,
                    max_output_tokens=args.max_output_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                skipped_other += 1
                print(f"[warn] skipping {qid=} due to error: {exc}")
                continue

            request_entry = {
                "custom_id": f"{input_path.stem}-{qid}",
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
            }
            out_f.write(json.dumps(request_entry))
            out_f.write("\n")
            written += 1

    print(f"Wrote {written} requests to {output_path}")
    if skipped_ineligible:
        print(
            f"Skipped {skipped_ineligible} items because stage 2 requires both "
            "text_keywords and visual_cues."
        )
    if skipped_missing_mask:
        print(f"Skipped {skipped_missing_mask} items due to missing mask images.")
    if skipped_other:
        print(f"Skipped {skipped_other} items due to processing errors.")


if __name__ == "__main__":
    main()
