import contextlib
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from transformers.cache_utils import HybridCache
from transformers.models.gemma3 import modeling_gemma3
from transformers.models.gemma3.modeling_gemma3 import repeat_kv


_ORIGINAL_EAGER_ATTENTION = modeling_gemma3.eager_attention_forward
_CAUSALMM_ATTENTION_METHOD: Optional[str] = None


def _normalise_allowed(values: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    values = values.masked_fill(~allowed, 0)
    denom = values.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(values.dtype).eps)
    return values / denom


def _shuffle_attention(attention: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    shuffled = attention.clone()
    flat_attention = shuffled.reshape(-1, shuffled.shape[-1])
    flat_allowed = allowed.expand_as(shuffled).reshape(-1, shuffled.shape[-1])
    for row_idx in range(flat_attention.shape[0]):
        valid_idx = torch.where(flat_allowed[row_idx])[0]
        if valid_idx.numel() > 1:
            perm = valid_idx[torch.randperm(valid_idx.numel(), device=valid_idx.device)]
            flat_attention[row_idx, valid_idx] = flat_attention[row_idx, perm]
    return _normalise_allowed(shuffled, allowed.expand_as(shuffled))


def apply_counterfactual_attention(
    attention: torch.Tensor,
    allowed: torch.Tensor,
    method: str,
) -> torch.Tensor:
    allowed = allowed.expand_as(attention)
    method = method.lower()

    if method in {"none", "normal"}:
        return attention
    if method == "reverse":
        max_values = attention.masked_fill(~allowed, 0).max(dim=-1, keepdim=True).values
        return (max_values - attention).masked_fill(~allowed, 0)
    if method == "reverse_and_normalize":
        max_values = attention.masked_fill(~allowed, 0).max(dim=-1, keepdim=True).values
        return _normalise_allowed(max_values - attention, allowed)
    if method == "random":
        return _normalise_allowed(torch.rand_like(attention), allowed)
    if method == "uniform":
        return _normalise_allowed(torch.ones_like(attention), allowed)
    if method == "shuffle":
        return _shuffle_attention(attention, allowed)

    raise ValueError(
        "Unknown causal attention method "
        f"{method!r}; choose from none, reverse, reverse_and_normalize, random, uniform, shuffle."
    )


def causalmm_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    **kwargs,
):
    method = _CAUSALMM_ATTENTION_METHOD
    if method is None:
        return _ORIGINAL_EAGER_ATTENTION(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            softcap=softcap,
            **kwargs,
        )

    if scaling is None:
        scaling = module.head_dim**-0.5

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
        allowed = causal_mask == 0
    else:
        allowed = torch.ones(
            attn_weights.shape[0],
            1,
            attn_weights.shape[-2],
            attn_weights.shape[-1],
            dtype=torch.bool,
            device=attn_weights.device,
        )

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = apply_counterfactual_attention(attn_weights, allowed, method)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


@contextlib.contextmanager
def causalmm_attention(method: str):
    global _CAUSALMM_ATTENTION_METHOD
    previous_method = _CAUSALMM_ATTENTION_METHOD
    previous_forward = modeling_gemma3.eager_attention_forward
    _CAUSALMM_ATTENTION_METHOD = method
    modeling_gemma3.eager_attention_forward = causalmm_eager_attention_forward
    try:
        yield
    finally:
        _CAUSALMM_ATTENTION_METHOD = previous_method
        modeling_gemma3.eager_attention_forward = previous_forward


@dataclass
class GenerationResult:
    text: str
    sequences: torch.LongTensor
    prompt_tokens: int
    completion_tokens: int


class CausalMMGemma3:
    def __init__(
        self,
        model_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
    ):
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation="sdpa",
        ).eval()
        self.model.config.text_config._attn_implementation = "eager"
        self.model.language_model.config._attn_implementation = "eager"

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def build_messages(self, prompt: str, image: Optional[Image.Image] = None, system: Optional[str] = None):
        messages = []
        if system:
            messages.append({"role": "system", "content": [{"type": "text", "text": system}]})

        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})
        return messages

    def prepare_inputs(self, prompt: str, image_path: Optional[str] = None, system: Optional[str] = None) -> Dict[str, torch.Tensor]:
        image = Image.open(image_path).convert("RGB") if image_path else None
        messages = self.build_messages(prompt=prompt, image=image, system=system)
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        prepared = {}
        for key, value in inputs.items():
            if not torch.is_tensor(value):
                continue
            if key == "pixel_values":
                prepared[key] = value.to(self.device, dtype=self.dtype)
            else:
                prepared[key] = value.to(self.device)
        return prepared

    def prepare_batch_inputs(
        self,
        prompts: Sequence[str],
        image_paths: Optional[Sequence[Optional[str]]] = None,
        systems: Optional[Sequence[Optional[str]]] = None,
    ) -> Dict[str, torch.Tensor]:
        image_paths = image_paths if image_paths is not None else [None] * len(prompts)
        systems = systems if systems is not None else [None] * len(prompts)
        conversations = []

        for prompt, image_path, system in zip(prompts, image_paths, systems):
            image = Image.open(image_path).convert("RGB") if image_path else None
            conversations.append(self.build_messages(prompt=prompt, image=image, system=system))

        inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )

        prepared = {}
        for key, value in inputs.items():
            if not torch.is_tensor(value):
                continue
            if key == "pixel_values":
                prepared[key] = value.to(self.device, dtype=self.dtype)
            else:
                prepared[key] = value.to(self.device)
        return prepared

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

        image_features = self.model.get_image_features(pixel_values)
        if cf_mode in {"vision", "both"}:
            image_features = self._edit_image_features(image_features, vision_method)

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

    def _edit_image_features(self, image_features: torch.Tensor, method: str) -> torch.Tensor:
        method = method.lower()
        if method in {"none", "normal"}:
            return image_features
        if method == "shuffle":
            edited = image_features.clone()
            for batch_idx in range(edited.shape[0]):
                perm = torch.randperm(edited.shape[1], device=edited.device)
                edited[batch_idx] = edited[batch_idx, perm]
            return edited
        if method == "uniform":
            return image_features.mean(dim=1, keepdim=True).expand_as(image_features)
        if method == "reverse":
            max_values = image_features.max(dim=1, keepdim=True).values
            return max_values - image_features
        if method == "random":
            return torch.randn_like(image_features) * image_features.std().clamp_min(1e-6) + image_features.mean()

        raise ValueError(f"Unknown vision method {method!r}; choose from none, shuffle, uniform, reverse, random.")

    def _new_cache(self, batch_size: int, max_cache_len: int) -> HybridCache:
        return HybridCache(
            self.model.config.text_config,
            max_batch_size=batch_size,
            max_cache_len=max_cache_len,
            device=self.device,
            dtype=self.dtype,
        )

    def _forward(
        self,
        input_ids: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor],
        past_key_values: HybridCache,
        cache_position: torch.Tensor,
        attention_method: Optional[str] = None,
    ):
        kwargs = {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "past_key_values": past_key_values,
            "cache_position": cache_position,
            "use_cache": True,
            "logits_to_keep": 1,
            "return_dict": True,
        }
        if attention_method:
            with causalmm_attention(attention_method):
                return self.model(**kwargs)
        return self.model(**kwargs)

    def _combine_logits(
        self,
        logits: torch.Tensor,
        cf_logits: torch.Tensor,
        gamma: float,
        epsilon: float,
    ) -> torch.Tensor:
        diff_logits = (1.0 + gamma) * logits - gamma * cf_logits
        if epsilon > 0:
            cutoff = math.log(epsilon) + logits.max(dim=-1, keepdim=True).values
            diff_logits = diff_logits.masked_fill(logits < cutoff, -float("inf"))
        return diff_logits

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: Optional[int],
    ) -> torch.LongTensor:
        if temperature <= 0:
            return logits.argmax(dim=-1)

        scores = logits / temperature
        if top_k is not None and top_k > 0:
            top_k = min(top_k, scores.shape[-1])
            threshold = torch.topk(scores, top_k, dim=-1).values[..., -1, None]
            scores = scores.masked_fill(scores < threshold, -float("inf"))

        if top_p is not None and 0 < top_p < 1:
            sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_scores, dim=-1)
            cumulative_probs = sorted_probs.cumsum(dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            scores = scores.scatter(
                dim=-1,
                index=sorted_indices,
                src=sorted_scores.masked_fill(sorted_indices_to_remove, -float("inf")),
            )

        probs = torch.softmax(scores, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(1)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        system: Optional[str] = None,
        max_new_tokens: int = 128,
        gamma: float = 1.0,
        epsilon: float = 0.1,
        temperature: float = 0.2,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        cf_mode: str = "language",
        attention_method: str = "reverse_and_normalize",
        vision_method: str = "shuffle",
    ) -> GenerationResult:
        return self.generate_batch(
            prompts=[prompt],
            image_paths=[image_path],
            systems=[system],
            max_new_tokens=max_new_tokens,
            gamma=gamma,
            epsilon=epsilon,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            cf_mode=cf_mode,
            attention_method=attention_method,
            vision_method=vision_method,
        )[0]

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: Sequence[str],
        image_paths: Optional[Sequence[Optional[str]]] = None,
        systems: Optional[Sequence[Optional[str]]] = None,
        max_new_tokens: int = 128,
        gamma: float = 1.0,
        epsilon: float = 0.1,
        temperature: float = 0.2,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        cf_mode: str = "language",
        attention_method: str = "reverse_and_normalize",
        vision_method: str = "shuffle",
    ) -> List[GenerationResult]:
        cf_mode = cf_mode.lower()
        if cf_mode not in {"language", "vision", "both"}:
            raise ValueError("cf_mode must be one of: language, vision, both")

        inputs = self.prepare_batch_inputs(prompts=prompts, image_paths=image_paths, systems=systems)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
        initial_attention_mask = attention_mask.clone()
        token_type_ids = inputs.get("token_type_ids")
        pixel_values = inputs.get("pixel_values")

        prompt_len = input_ids.shape[1]
        batch_size = input_ids.shape[0]
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
            cf_mode=cf_mode,
            vision_method=vision_method,
        )

        normal_cache = self._new_cache(batch_size=batch_size, max_cache_len=max_cache_len)
        cf_cache = self._new_cache(batch_size=batch_size, max_cache_len=max_cache_len)
        cache_position = torch.arange(prompt_len, device=self.device)

        normal_outputs = self._forward(
            input_ids=None,
            inputs_embeds=normal_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            past_key_values=normal_cache,
            cache_position=cache_position,
        )
        cf_attention_method = attention_method if cf_mode in {"language", "both"} else None
        cf_outputs = self._forward(
            input_ids=None,
            inputs_embeds=cf_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            past_key_values=cf_cache,
            cache_position=cache_position,
            attention_method=cf_attention_method,
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
        unfinished = torch.ones(batch_size, dtype=torch.bool, device=self.device)

        for step in range(max_new_tokens):
            logits = normal_outputs.logits[:, -1, :]
            cf_logits = cf_outputs.logits[:, -1, :]
            causalmm_logits = self._combine_logits(logits=logits, cf_logits=cf_logits, gamma=gamma, epsilon=epsilon)
            next_token = self._sample_next_token(
                causalmm_logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            next_token = torch.where(
                unfinished,
                next_token,
                torch.full_like(next_token, int(pad_token_id)),
            )

            generated = torch.cat([generated, next_token[:, None]], dim=-1)
            unfinished = unfinished & ~torch.isin(next_token, eos_ids)
            if not unfinished.any():
                break

            next_attention = unfinished.to(dtype=attention_mask.dtype).unsqueeze(1)
            attention_mask = torch.cat([attention_mask, next_attention], dim=1)
            if token_type_ids is not None:
                token_type_ids = torch.cat(
                    [token_type_ids, torch.zeros((batch_size, 1), device=self.device, dtype=token_type_ids.dtype)],
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
                attention_method=cf_attention_method,
            )

        completion_ids = generated[:, prompt_len:]
        prompt_token_counts = initial_attention_mask.sum(dim=1).tolist()
        results = []
        for batch_idx in range(batch_size):
            text = self.processor.decode(completion_ids[batch_idx], skip_special_tokens=True).strip()
            results.append(
                GenerationResult(
                    text=text,
                    sequences=generated[batch_idx : batch_idx + 1],
                    prompt_tokens=int(prompt_token_counts[batch_idx]),
                    completion_tokens=int(completion_ids[batch_idx].shape[0]),
                )
            )
        return results
