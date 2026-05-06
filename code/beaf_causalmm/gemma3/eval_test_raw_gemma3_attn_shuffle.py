import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from tqdm import tqdm
from transformers import set_seed

from causalmm_gemma3 import CausalMMGemma3


BUNDLE_ROOT = Path(__file__).resolve().parents[3]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Gemma 3 with vision attention-map shuffle CausalMM on test_raw_llava.jsonl."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(BUNDLE_ROOT / "models/Gemma-3-4B-IT"),
    )
    parser.add_argument(
        "--question-file",
        type=str,
        default=str(BUNDLE_ROOT / "outputs/beaf_causalmm/test_raw_llava.jsonl"),
    )
    parser.add_argument(
        "--answer-file",
        type=str,
        default=str(BUNDLE_ROOT / "data/eval/test_raw_with_shortcut_answer.json"),
    )
    parser.add_argument(
        "--image-folder",
        type=str,
        default=str(BUNDLE_ROOT / "data/playground_data/coco/train2014"),
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=str(BUNDLE_ROOT / "outputs/beaf_causalmm/gemma3_attn_shuffle_test_raw_results.json"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the dataset into this many shards and only run one shard in this process.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="0-based shard index to run when --num-shards > 1.",
    )
    parser.add_argument(
        "--attention-layer",
        type=int,
        default=-1,
        help="Gemma3 vision attention layer used for patch importance. -1 means the last layer.",
    )
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Forwarded to from_pretrained. Use with CUDA_VISIBLE_DEVICES to pin one process per GPU.",
    )
    parser.add_argument(
        "--progress-position",
        type=int,
        default=0,
        help="tqdm line position. Useful when running multiple shards in parallel on one terminal.",
    )
    return parser.parse_args()


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_answers(path: str) -> Dict[int, dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(item["question_id"]): item for item in data}


def clean_question(text: str) -> str:
    return text.replace("<image>", "").strip()


def resolve_image(image_folder: str, image_value: str) -> str:
    if os.path.isabs(image_value):
        return image_value
    return os.path.join(image_folder, image_value)


def load_done_ids(path: Path) -> set:
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if "question_id" in item:
                done.add(int(item["question_id"]))
    return done


def select_shard(rows: List[dict], num_shards: int, shard_index: int) -> List[dict]:
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    if num_shards == 1:
        return rows
    return [row for row in rows if int(row["source_index"]) % num_shards == shard_index]


def build_sample_seed(base_seed: int, question_id: int) -> int:
    # Keep shuffle randomness stable per sample so single-GPU and sharded runs match.
    return (int(base_seed) * 1_000_003 + int(question_id)) % (2**63 - 1)


def write_final_json(tmp_file: Path, output_file: Path) -> None:
    rows = []
    with tmp_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append(
                {
                    "question": item["question"],
                    "llm_output": item["llm_output"],
                    "correct_answer": item["correct_answer"],
                    "answer_type": item["answer_type"],
                }
            )
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


class Gemma3AttentionShuffleCausalMM(CausalMMGemma3):
    def __init__(
        self,
        model_path: str,
        attention_layer: int = -1,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
    ):
        super().__init__(model_path=model_path, torch_dtype=torch_dtype, device_map=device_map)
        self.attention_layer = attention_layer
        self._sample_seed: Optional[int] = None

    def _counterfactual_image_features_from_attention(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_outputs = self.model.vision_tower(
            pixel_values=pixel_values,
            output_hidden_states=False,
            output_attentions=True,
            return_dict=True,
        )
        vision_hidden = vision_outputs.last_hidden_state
        attention = vision_outputs.attentions[self.attention_layer]

        # Gemma3/SigLIP attention shape is [batch, heads, query_patches, key_patches].
        # This keeps the original CausalMM idea: shuffle a vision attention map,
        # use it to reweight vision features, then run Gemma3's projector to
        # produce counterfactual image soft-token embeddings.
        #
        # Full per-head 4096x4096 map manipulation is very memory heavy, so we
        # aggregate to a key-patch importance map before shuffling.
        patch_weights = attention.mean(dim=(1, 2))
        patch_weights = patch_weights / patch_weights.mean(dim=1, keepdim=True).clamp_min(
            torch.finfo(patch_weights.dtype).eps
        )

        shuffled = patch_weights.clone()
        generator = None
        if self._sample_seed is not None:
            generator = torch.Generator(device=shuffled.device)
            generator.manual_seed(int(self._sample_seed))
        for batch_idx in range(shuffled.shape[0]):
            perm = torch.randperm(shuffled.shape[1], device=shuffled.device, generator=generator)
            shuffled[batch_idx] = shuffled[batch_idx, perm]

        weighted_hidden = vision_hidden * shuffled.unsqueeze(-1)
        return self.model.multi_modal_projector(weighted_hidden)

    def _merge_image_features(
        self,
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.Tensor],
        cf_mode: str,
        vision_method: str,
    ) -> torch.Tensor:
        input_ids_for_embed = input_ids
        if self.model.config.image_token_index >= self.model.vocab_size:
            input_ids_for_embed = input_ids.clone()
            input_ids_for_embed[input_ids_for_embed == self.model.config.image_token_index] = 0

        inputs_embeds = self.model.get_input_embeddings()(input_ids_for_embed)
        if pixel_values is None:
            return inputs_embeds

        if cf_mode == "vision_attention_shuffle":
            image_features = self._counterfactual_image_features_from_attention(pixel_values)
        else:
            image_features = self.model.get_image_features(pixel_values)

        special_image_mask = (input_ids == self.model.config.image_token_index).unsqueeze(-1)
        special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
        if inputs_embeds[special_image_mask].numel() != image_features.numel():
            image_tokens_in_text = special_image_mask.sum(dim=1).sum(dim=0)[0]
            raise ValueError(
                f"Image token mismatch: got {image_tokens_in_text} soft image tokens in text, "
                f"but {image_features.shape[0] * image_features.shape[1]} image features."
            )

        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        return inputs_embeds.masked_scatter(special_image_mask, image_features)

    @torch.inference_mode()
    def generate_one(
        self,
        prompt: str,
        image_path: str,
        max_new_tokens: int,
        gamma: float,
        epsilon: float,
        temperature: float,
        top_p: float,
        sample_seed: Optional[int] = None,
    ) -> str:
        self._sample_seed = sample_seed
        inputs = self.prepare_batch_inputs(prompts=[prompt], image_paths=[image_path], systems=[None])
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
        token_type_ids = inputs.get("token_type_ids")
        pixel_values = inputs.get("pixel_values")

        prompt_len = input_ids.shape[1]
        max_cache_len = prompt_len + max_new_tokens

        normal_embeds = self._merge_image_features(
            input_ids=input_ids,
            pixel_values=pixel_values,
            cf_mode="normal",
            vision_method="none",
        )
        cf_embeds = self._merge_image_features(
            input_ids=input_ids,
            pixel_values=pixel_values,
            cf_mode="vision_attention_shuffle",
            vision_method="attention_shuffle",
        )
        self._sample_seed = None

        normal_cache = self._new_cache(batch_size=1, max_cache_len=max_cache_len)
        cf_cache = self._new_cache(batch_size=1, max_cache_len=max_cache_len)
        cache_position = torch.arange(prompt_len, device=self.device)

        normal_outputs = self._forward(
            input_ids=None,
            inputs_embeds=normal_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            past_key_values=normal_cache,
            cache_position=cache_position,
        )
        cf_outputs = self._forward(
            input_ids=None,
            inputs_embeds=cf_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            past_key_values=cf_cache,
            cache_position=cache_position,
        )

        generated = input_ids.clone()
        eos_ids = self.model.generation_config.eos_token_id or self.model.config.eos_token_id
        eos_ids = eos_ids if isinstance(eos_ids, Sequence) else [eos_ids]
        eos_ids = torch.tensor(eos_ids, device=self.device)
        pad_token_id = self.model.generation_config.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.model.config.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processor.tokenizer.pad_token_id
        unfinished = torch.ones(1, dtype=torch.bool, device=self.device)

        for step in range(max_new_tokens):
            logits = normal_outputs.logits[:, -1, :]
            cf_logits = cf_outputs.logits[:, -1, :]
            causalmm_logits = self._combine_logits(logits=logits, cf_logits=cf_logits, gamma=gamma, epsilon=epsilon)
            next_token = self._sample_next_token(causalmm_logits, temperature=temperature, top_p=top_p, top_k=None)
            next_token = torch.where(unfinished, next_token, torch.full_like(next_token, int(pad_token_id)))

            generated = torch.cat([generated, next_token[:, None]], dim=-1)
            unfinished = unfinished & ~torch.isin(next_token, eos_ids)
            if not unfinished.any():
                break

            attention_mask = torch.cat([attention_mask, unfinished.to(dtype=attention_mask.dtype).unsqueeze(1)], dim=1)
            if token_type_ids is not None:
                token_type_ids = torch.cat(
                    [token_type_ids, torch.zeros((1, 1), device=self.device, dtype=token_type_ids.dtype)],
                    dim=1,
                )

            next_input_ids = next_token[:, None]
            next_cache_position = torch.tensor([prompt_len + step], device=self.device)
            normal_outputs = self._forward(
                input_ids=next_input_ids,
                inputs_embeds=None,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                past_key_values=normal_cache,
                cache_position=next_cache_position,
            )
            cf_outputs = self._forward(
                input_ids=next_input_ids,
                inputs_embeds=None,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                past_key_values=cf_cache,
                cache_position=next_cache_position,
            )

        completion_ids = generated[:, prompt_len:]
        return self.processor.decode(completion_ids[0], skip_special_tokens=True).strip()


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.manual_seed(args.seed)

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    question_rows = load_jsonl(args.question_file)
    answer_by_id = load_answers(args.answer_file)
    if args.limit is not None:
        question_rows = question_rows[: args.limit]
    question_rows = [{**row, "source_index": idx} for idx, row in enumerate(question_rows)]
    question_rows = select_shard(question_rows, num_shards=args.num_shards, shard_index=args.shard_index)

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = output_file.with_suffix(output_file.suffix + ".jsonl")

    done_ids = load_done_ids(tmp_file) if args.resume else set()
    mode = "a" if args.resume else "w"

    runner = Gemma3AttentionShuffleCausalMM(
        model_path=args.model_path,
        attention_layer=args.attention_layer,
        torch_dtype=dtype,
        device_map=args.device_map,
    )

    pending_rows = [row for row in question_rows if int(row["question_id"]) not in done_ids]
    with tmp_file.open(mode, encoding="utf-8") as f:
        progress = tqdm(
            total=len(pending_rows),
            desc=f"Gemma3+vision-attn-shuffle [{args.shard_index + 1}/{args.num_shards}]",
            position=args.progress_position,
            dynamic_ncols=True,
            leave=True,
        )
        for row in pending_rows:
            question_id = int(row["question_id"])
            answer_item = answer_by_id.get(question_id)
            if answer_item is None:
                raise KeyError(f"question_id {question_id} not found in {args.answer_file}")

            question = clean_question(row["text"])
            prompt = question + "\nAnswer with only the final answer, using as few words as possible."
            image_path = resolve_image(args.image_folder, row["image"])
            text = runner.generate_one(
                prompt=prompt,
                image_path=image_path,
                max_new_tokens=args.max_new_tokens,
                gamma=args.gamma,
                epsilon=args.epsilon,
                temperature=args.temperature,
                top_p=args.top_p,
                sample_seed=build_sample_seed(args.seed, question_id),
            )

            f.write(
                json.dumps(
                    {
                        "source_index": int(row["source_index"]),
                        "question_id": question_id,
                        "question": answer_item["question"],
                        "llm_output": text,
                        "correct_answer": answer_item["answer"],
                        "answer_type": answer_item["answer_type"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            f.flush()
            progress.update(1)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        progress.close()

    write_final_json(tmp_file, output_file)
    print(output_file)


if __name__ == "__main__":
    main()
