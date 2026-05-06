import math
import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from transformers import Trainer
from transformers.trainer import (
    ALL_LAYERNORM_LAYERS,
    get_parameter_names,
    has_length,
    is_sagemaker_mp_enabled,
    logger,
)

from transformers.trainer_utils import seed_worker

from llava.eval.m4c_evaluator import EvalAIAnswerProcessor


def split_to_even_chunks(indices, lengths, num_chunks):
    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks
    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")
    return chunks


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]
    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    def __init__(self, batch_size: int, world_size: int, lengths: Optional[List[int]] = None, generator=None):
        if lengths is None:
            raise ValueError("Lengths must be provided.")
        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class RepeatBatchSampler(Sampler):
    def __init__(self, sampler, batch_size: int, drop_last: bool, repeat_count: int):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.repeat_count = max(int(repeat_count), 1)

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                for _ in range(self.repeat_count):
                    for repeated_idx in batch:
                        yield repeated_idx
                batch = []
        if batch and not self.drop_last:
            for _ in range(self.repeat_count):
                for repeated_idx in batch:
                    yield repeated_idx

    def __len__(self):
        if self.drop_last:
            base_indices = (len(self.sampler) // self.batch_size) * self.batch_size
        else:
            base_indices = len(self.sampler)
        return base_indices * self.repeat_count


class LLaVAGRPOTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.ref_model = kwargs.pop("ref_model", None)
        super().__init__(*args, **kwargs)
        self.answer_processor = EvalAIAnswerProcessor()
        self._rollout_cache = None

        if self.ref_model is None:
            raise ValueError("ref_model must be provided for KL penalty.")
        self.ref_model.requires_grad_(False)
        self.ref_model.eval()

        if self._get_repeat_budget() > 1 and self.args.gradient_accumulation_steps > 1:
            raise ValueError(
                "grpo_update_epochs > 1 cannot be combined with gradient_accumulation_steps > 1 "
                "in the current trainer: repeated rollout passes happen before any optimizer step, "
                "so ratio_mean stays near 1.0 and the policy barely updates. "
                "Set gradient_accumulation_steps=1 or grpo_update_epochs=1."
            )

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if getattr(self.args, "group_by_length", False):
            lengths = self.train_dataset.lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
            )
        return super()._get_train_sampler()

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            sampler = self._get_train_sampler()
            repeat_count = self._get_repeat_budget()
            if repeat_count > 1:
                dataloader_params["batch_size"] = self._train_batch_size
                dataloader_params["sampler"] = RepeatBatchSampler(
                    sampler=sampler,
                    batch_size=self._train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    repeat_count=repeat_count,
                )
                dataloader_params["drop_last"] = self.args.dataloader_drop_last
            else:
                dataloader_params["batch_size"] = self._train_batch_size
                dataloader_params["sampler"] = sampler
                dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
        else:
            dataloader_params["batch_size"] = self._train_batch_size

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            return {}

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        model = self.model
        was_training = model.training
        model.eval()
        self._reset_rollout_cache()

        pair_metric_keys = [
            "ori_acc",
            "as_acc",
            "sep",
            "cross",
            "empty_rate",
            "long_rate",
            "repeat_rate",
            "format_rate",
        ]
        group_metric_avg_keys = [
            "group_reward_std",
            "low_var_group_rate",
        ]
        group_metric_max_keys = [
            "group_reward_std_max",
            "active_group_reward_std_max",
        ]

        pair_metric_sums = {key: 0.0 for key in pair_metric_keys}
        group_metric_sums = {key: 0.0 for key in group_metric_avg_keys}
        group_metric_max = {key: 0.0 for key in group_metric_max_keys}

        total_pairs = 0
        total_groups = 0
        skipped_batches = 0
        reward_sum = 0.0
        reward_sq_sum = 0.0
        max_groups = max(int(getattr(self.args, "grpo_eval_max_groups", 0)), 0)

        try:
            for inputs in eval_dataloader:
                if max_groups > 0 and total_groups >= max_groups:
                    break

                inputs = self._prepare_inputs(inputs)
                skip_batch = inputs.pop("skip_batch", None)
                if skip_batch is not None:
                    skip_value = bool(skip_batch.item()) if isinstance(skip_batch, torch.Tensor) else bool(skip_batch)
                    if skip_value:
                        skipped_batches += 1
                        continue

                batch_group_count = len(inputs["group_ids"])
                if max_groups > 0 and total_groups + batch_group_count > max_groups:
                    break

                sampled_completions, completion_texts, prompt_indices = self._sample_completions(model, inputs)
                expanded_answers = [inputs["answers"][idx] for idx in prompt_indices]
                pair_rewards, _, reward_metrics = self._compute_group_rewards(completion_texts, expanded_answers)

                pair_rewards = pair_rewards.detach().float().cpu()
                pair_count = int(pair_rewards.numel())
                if pair_count == 0:
                    skipped_batches += 1
                    continue

                total_pairs += pair_count
                total_groups += batch_group_count
                reward_sum += float(pair_rewards.sum().item())
                reward_sq_sum += float(pair_rewards.pow(2).sum().item())

                for key in pair_metric_keys:
                    pair_metric_sums[key] += float(reward_metrics.get(key, 0.0)) * pair_count
                for key in group_metric_avg_keys:
                    group_metric_sums[key] += float(reward_metrics.get(key, 0.0)) * batch_group_count
                for key in group_metric_max_keys:
                    group_metric_max[key] = max(group_metric_max[key], float(reward_metrics.get(key, 0.0)))
        finally:
            self._reset_rollout_cache()
            if was_training:
                model.train()

        metrics: Dict[str, float] = {
            f"{metric_key_prefix}/num_groups": float(total_groups),
            f"{metric_key_prefix}/num_pairs": float(total_pairs),
            f"{metric_key_prefix}/skipped_batches": float(skipped_batches),
        }

        if total_pairs > 0:
            reward_mean = reward_sum / total_pairs
            reward_var = max(reward_sq_sum / total_pairs - reward_mean * reward_mean, 0.0)
            metrics[f"{metric_key_prefix}/reward_mean"] = reward_mean
            metrics[f"{metric_key_prefix}/reward_std"] = math.sqrt(reward_var)
            for key in pair_metric_keys:
                metrics[f"{metric_key_prefix}/{key}"] = pair_metric_sums[key] / total_pairs
        else:
            metrics[f"{metric_key_prefix}/reward_mean"] = 0.0
            metrics[f"{metric_key_prefix}/reward_std"] = 0.0
            for key in pair_metric_keys:
                metrics[f"{metric_key_prefix}/{key}"] = 0.0

        if total_groups > 0:
            for key in group_metric_avg_keys:
                metrics[f"{metric_key_prefix}/{key}"] = group_metric_sums[key] / total_groups
        else:
            for key in group_metric_avg_keys:
                metrics[f"{metric_key_prefix}/{key}"] = 0.0

        for key, value in group_metric_max.items():
            metrics[f"{metric_key_prefix}/{key}"] = value

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def _make_rollout_cache_key(self, inputs) -> tuple:
        return tuple(inputs["group_ids"])

    def _reset_rollout_cache(self):
        self._rollout_cache = None

    def _get_repeat_budget(self) -> int:
        return max(int(getattr(self.args, "grpo_update_epochs", 1)), 1)

    def _select_reuse_count(self, reward_metrics) -> int:
        repeat_budget = self._get_repeat_budget()
        low_var_threshold = max(float(getattr(self.args, "grpo_low_var_threshold", 0.0)), 0.0)
        high_var_threshold = max(float(getattr(self.args, "grpo_high_var_threshold", 0.0)), low_var_threshold)
        reward_std = float(reward_metrics.get("active_group_reward_std_max", 0.0))

        if reward_metrics.get("all_groups_low_var", 0.0) >= 1.0:
            return 0

        if not bool(getattr(self.args, "grpo_adaptive_group_reuse", False)) or repeat_budget <= 1:
            return repeat_budget

        if high_var_threshold <= low_var_threshold:
            return repeat_budget

        scaled = (reward_std - low_var_threshold) / max(high_var_threshold - low_var_threshold, 1e-8)
        scaled = min(max(scaled, 0.0), 1.0)
        desired_uses = 1 + int(round(scaled * (repeat_budget - 1)))
        return min(max(desired_uses, 1), repeat_budget)

    def _make_zero_loss(self, return_outputs=False):
        loss = torch.zeros((), device=self.args.device, requires_grad=True)
        return (loss, None) if return_outputs else loss

    def create_optimizer(self):
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model
        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if getattr(self.args, "mm_projector_lr", None) is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if n in decay_parameters and n not in projector_parameters and p.requires_grad
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if n not in decay_parameters and n not in projector_parameters and p.requires_grad
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if n in decay_parameters and n in projector_parameters and p.requires_grad
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if n not in decay_parameters and n in projector_parameters and p.requires_grad
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if n in decay_parameters and p.requires_grad],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if n not in decay_parameters and p.requires_grad],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def _select_image(self, images, index: int):
        if isinstance(images, torch.Tensor):
            return images[index : index + 1]
        return [images[index]]

    def _truncate_completion_text(self, text: str) -> str:
        text = text.strip()
        if not text:
            return text
        text = text.split("\n")[0].strip()
        return text

    def _normalize_answer(self, text: str) -> str:
        return self.answer_processor(self._truncate_completion_text(text))

    def _top_p_sample(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        filtered_logits = sorted_logits.masked_fill(sorted_mask, torch.finfo(sorted_logits.dtype).min)
        filtered_probs = F.softmax(filtered_logits, dim=-1)
        sampled_index = torch.multinomial(filtered_probs, num_samples=1)
        next_token = sorted_indices.gather(-1, sampled_index)
        return next_token.squeeze(-1)

    @torch.no_grad()
    def _sample_single_completion(
        self,
        model,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        clip_input_ids: torch.Tensor,
        clip_attn_mask: torch.Tensor,
        image,
        image_size,
    ) -> torch.Tensor:
        input_ids = prompt_input_ids.unsqueeze(0)
        attention_mask = prompt_attention_mask.unsqueeze(0)
        clip_input_ids = clip_input_ids.unsqueeze(0)
        clip_attn_mask = clip_attn_mask.unsqueeze(0)
        generated_tokens = []
        eos_token_id = self.tokenizer.eos_token_id

        for _ in range(self.args.grpo_max_new_tokens):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                clip_input_ids=clip_input_ids,
                clip_attn_mask=clip_attn_mask,
                images=image,
                image_sizes=[image_size],
                use_cache=False,
                return_dict=True,
            )
            next_token_logits = outputs.logits[:, -1, :]
            if eos_token_id is not None and len(generated_tokens) < self.args.grpo_min_new_tokens:
                next_token_logits[:, eos_token_id] = torch.finfo(next_token_logits.dtype).min
            if self.args.grpo_do_sample:
                temperature = max(float(self.args.grpo_temperature), 1e-5)
                next_token_logits = next_token_logits / temperature
                if self.args.grpo_top_p < 1.0:
                    next_token = self._top_p_sample(next_token_logits, self.args.grpo_top_p)
                else:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_token = next_token_logits.argmax(dim=-1)

            generated_tokens.append(next_token)
            input_ids = torch.cat([input_ids, next_token[:, None]], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)],
                dim=1,
            )
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        if not generated_tokens:
            return torch.empty(0, dtype=prompt_input_ids.dtype, device=prompt_input_ids.device)
        return torch.cat(generated_tokens, dim=0)

    def _sample_completions(self, model, inputs):
        prompt_input_ids = inputs["prompt_input_ids"]
        prompt_attention_mask = inputs["prompt_attention_mask"]
        clip_input_ids = inputs["clip_input_ids"]
        clip_attn_mask = inputs["clip_attn_mask"]
        images = inputs["images"]
        image_sizes = inputs["image_sizes"]

        batch_size = prompt_input_ids.size(0)
        if batch_size % 2 != 0:
            raise ValueError("Batch size must be even: expected flattened (ori, as) pairs.")

        completions = []
        completion_texts = []
        prompt_indices = []

        # order:
        # pair0_sample0_ori, pair0_sample0_as,
        # pair0_sample1_ori, pair0_sample1_as, ...
        for pair_start in range(0, batch_size, 2):
            ori_idx = pair_start
            as_idx = pair_start + 1

            for _ in range(self.args.grpo_group_size):
                for prompt_idx in (ori_idx, as_idx):
                    prompt_len = int(prompt_attention_mask[prompt_idx].sum().item())
                    image = self._select_image(images, prompt_idx)

                    completion_ids = self._sample_single_completion(
                        model=model,
                        prompt_input_ids=prompt_input_ids[prompt_idx, :prompt_len],
                        prompt_attention_mask=prompt_attention_mask[prompt_idx, :prompt_len],
                        clip_input_ids=clip_input_ids[prompt_idx],
                        clip_attn_mask=clip_attn_mask[prompt_idx],
                        image=image,
                        image_size=image_sizes[prompt_idx],
                    )

                    completions.append(completion_ids)
                    completion_texts.append(
                        self.tokenizer.decode(
                            completion_ids.detach().cpu().tolist(),
                            skip_special_tokens=True
                        ).strip()
                    )
                    prompt_indices.append(prompt_idx)

        return completions, completion_texts, prompt_indices    

    def _build_policy_inputs(self, inputs, completions, prompt_indices):
        prompt_input_ids = inputs["prompt_input_ids"]
        prompt_attention_mask = inputs["prompt_attention_mask"]
        clip_input_ids = inputs["clip_input_ids"]
        clip_attn_mask = inputs["clip_attn_mask"]
        images = inputs["images"]
        image_sizes = inputs["image_sizes"]

        full_sequences = []
        full_masks = []
        gen_token_masks = []

        expanded_clip_input_ids = []
        expanded_clip_attn_mask = []
        expanded_image_sizes = []

        if isinstance(images, torch.Tensor):
            expanded_images = []
        else:
            expanded_images = []

        for prompt_idx, completion_ids in zip(prompt_indices, completions):
            prompt_len = int(prompt_attention_mask[prompt_idx].sum().item())
            prompt_ids = prompt_input_ids[prompt_idx, :prompt_len]

            full_ids = torch.cat([prompt_ids, completion_ids], dim=0)
            full_mask = torch.ones_like(full_ids)
            gen_mask = torch.zeros_like(full_ids, dtype=torch.bool)
            if completion_ids.numel() > 0:
                gen_mask[prompt_len:] = True

            full_sequences.append(full_ids)
            full_masks.append(full_mask)
            gen_token_masks.append(gen_mask)

            expanded_clip_input_ids.append(clip_input_ids[prompt_idx])
            expanded_clip_attn_mask.append(clip_attn_mask[prompt_idx])
            expanded_image_sizes.append(image_sizes[prompt_idx])

            if isinstance(images, torch.Tensor):
                expanded_images.append(images[prompt_idx : prompt_idx + 1])
            else:
                expanded_images.append(images[prompt_idx])

        pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

        full_input_ids = torch.nn.utils.rnn.pad_sequence(
            full_sequences, batch_first=True, padding_value=pad_token_id
        )
        full_attention_mask = torch.nn.utils.rnn.pad_sequence(
            full_masks, batch_first=True, padding_value=0
        )
        full_gen_mask = torch.nn.utils.rnn.pad_sequence(
            gen_token_masks, batch_first=True, padding_value=False
        )

        expanded_clip_input_ids = torch.stack(expanded_clip_input_ids, dim=0)
        expanded_clip_attn_mask = torch.stack(expanded_clip_attn_mask, dim=0)

        if isinstance(images, torch.Tensor):
            expanded_images = torch.cat(expanded_images, dim=0)

        return (
            full_input_ids,
            full_attention_mask,
            full_gen_mask,
            expanded_clip_input_ids,
            expanded_clip_attn_mask,
            expanded_images,
            expanded_image_sizes,
        )

    def _has_repetition(self, text: str) -> bool:
        words = text.lower().split()
        if len(words) < 6:
            return False
        for i in range(len(words) - 2):
            tri = words[i:i + 3]
            for j in range(i + 3, len(words) - 2):
                if words[j:j + 3] == tri:
                    return True
        return False

    def _compute_seq_log_probs(self, logits, full_input_ids, full_gen_mask):
        shift_logits = logits[:, :-1, :]
        shift_labels = full_input_ids[:, 1:]
        shift_gen_mask = full_gen_mask[:, 1:]

        token_log_probs = F.log_softmax(shift_logits, dim=-1)

        safe_shift_labels = shift_labels.masked_fill(~shift_gen_mask, 0)
        safe_shift_labels = safe_shift_labels.masked_fill(safe_shift_labels < 0, 0)
        safe_shift_labels = safe_shift_labels.masked_fill(
            safe_shift_labels >= shift_logits.size(-1), 0
        )
        safe_shift_labels = safe_shift_labels.long()

        gathered_log_probs = token_log_probs.gather(
            dim=-1,
            index=safe_shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        token_counts = shift_gen_mask.sum(dim=-1).clamp_min(1).to(gathered_log_probs.dtype)
        seq_log_probs = (gathered_log_probs * shift_gen_mask).sum(dim=-1) / token_counts
        return seq_log_probs, gathered_log_probs, shift_gen_mask, token_counts

    def _is_yesno(self, text: str) -> bool:
        text = text.strip().lower()
        return text in {"yes", "no"}

    def _looks_bad_answer(self, text: str) -> bool:
        text = text.strip()
        if not text:
            return False

        bad_patterns = [
            "the answer is",
            "in the image",
            "are you",
            "again,",
            "an answer",
            "the last one",
        ]
        lower = text.lower()
        if any(p in lower for p in bad_patterns):
            return True

        if "\n" in text:
            return True

        compact = "".join(text.split())
        if compact.isdigit() and len(compact) > 4:
            return True

        bad_chars = set("{}[]|\\")
        if any(ch in text for ch in bad_chars):
            return True

        return False
        
    def _compute_group_rewards(self, predictions: List[str], answers: List[str]):
        block_size = 2 * self.args.grpo_group_size
        if len(predictions) % block_size != 0:
            raise ValueError(
                f"Predictions must be divisible by 2 * grpo_group_size ({block_size})."
            )

        all_pair_rewards = []
        all_pair_advantages = []
        low_var_threshold = max(float(getattr(self.args, "grpo_low_var_threshold", 0.0)), 0.0)

        metrics = {
            "ori_acc": [],
            "as_acc": [],
            "sep": [],
            "cross": [],
            "empty_rate": [],
            "long_rate": [],
            "repeat_rate": [],
            "format_rate": [],
            "group_reward_std": [],
            "active_group_reward_std": [],
            "low_var_group_rate": [],
        }

        for base in range(0, len(predictions), block_size):
            local_pair_rewards = []

            for offset in range(0, block_size, 2):
                idx = base + offset

                pred_ori_raw = predictions[idx].strip()
                pred_as_raw = predictions[idx + 1].strip()
                pred_ori = self._normalize_answer(pred_ori_raw)
                pred_as = self._normalize_answer(pred_as_raw)
                ans_ori = self._normalize_answer(answers[idx])
                ans_as = self._normalize_answer(answers[idx + 1])

                r_ori = float(pred_ori == ans_ori)
                r_as = float(pred_as == ans_as)

                r_sep = float(
                    (pred_ori == ans_ori)
                    and (pred_as == ans_as)
                    and (pred_ori != pred_as)
                )

                r_cross = -float(pred_ori == ans_as) - float(pred_as == ans_ori)

                pred_ori_words = pred_ori_raw.replace("\n", " ").split()
                pred_as_words = pred_as_raw.replace("\n", " ").split()

                empty_penalty = float(pred_ori_raw == "") + float(pred_as_raw == "")
                length_penalty = float(len(pred_ori_words) > self.args.grpo_max_answer_words) + float(
                    len(pred_as_words) > self.args.grpo_max_answer_words
                )
                repeat_penalty = float(self._has_repetition(pred_ori_raw.replace("\n", " "))) + float(
                    self._has_repetition(pred_as_raw.replace("\n", " "))
                )

                format_penalty = 0.0

                if ans_ori in {"yes", "no"} and not self._is_yesno(pred_ori):
                    format_penalty += 1.0
                if ans_as in {"yes", "no"} and not self._is_yesno(pred_as):
                    format_penalty += 1.0

                format_penalty += float(self._looks_bad_answer(pred_ori_raw))
                format_penalty += float(self._looks_bad_answer(pred_as_raw))

                reward = (
                    self.args.grpo_lambda_ori * r_ori
                    + self.args.grpo_lambda_as * r_as
                    + self.args.grpo_lambda_sep * r_sep
                    + self.args.grpo_lambda_cross * r_cross
                    - self.args.grpo_empty_penalty * empty_penalty
                    - self.args.grpo_length_penalty * length_penalty
                    - self.args.grpo_repeat_penalty * repeat_penalty
                    - 1.0 * format_penalty
                )

                local_pair_rewards.append(reward)

                metrics["ori_acc"].append(r_ori)
                metrics["as_acc"].append(r_as)
                metrics["sep"].append(r_sep)
                metrics["cross"].append(r_cross)
                metrics["empty_rate"].append(empty_penalty / 2.0)
                metrics["long_rate"].append(length_penalty / 2.0)
                metrics["repeat_rate"].append(repeat_penalty / 2.0)
                metrics["format_rate"].append(format_penalty / 2.0)

            local_pair_rewards = torch.tensor(
                local_pair_rewards,
                device=self.args.device,
                dtype=torch.float32,
            )
            local_reward_std = float(local_pair_rewards.std(unbiased=False).item()) if local_pair_rewards.numel() > 1 else 0.0
            low_var_group = bool(low_var_threshold > 0.0 and local_reward_std < low_var_threshold)

            if local_pair_rewards.numel() > 1:
                local_pair_advantages = (
                    local_pair_rewards - local_pair_rewards.mean()
                ) / (local_pair_rewards.std(unbiased=False) + self.args.grpo_reward_eps)
            else:
                local_pair_advantages = local_pair_rewards

            if low_var_group:
                local_pair_advantages = torch.zeros_like(local_pair_advantages)

            all_pair_rewards.append(local_pair_rewards)
            all_pair_advantages.append(local_pair_advantages)
            metrics["group_reward_std"].append(local_reward_std)
            metrics["active_group_reward_std"].append(0.0 if low_var_group else local_reward_std)
            metrics["low_var_group_rate"].append(float(low_var_group))

        all_pair_rewards = torch.cat(all_pair_rewards, dim=0)
        all_pair_advantages = torch.cat(all_pair_advantages, dim=0)

        metric_means = {
            key: float(sum(values) / max(len(values), 1))
            for key, values in metrics.items()
        }
        metric_means["group_reward_std_max"] = max(metrics["group_reward_std"]) if metrics["group_reward_std"] else 0.0
        metric_means["active_group_reward_std_max"] = max(metrics["active_group_reward_std"]) if metrics["active_group_reward_std"] else 0.0
        metric_means["all_groups_low_var"] = float(all(value > 0.0 for value in metrics["low_var_group_rate"])) if metrics["low_var_group_rate"] else 0.0

        return all_pair_rewards, all_pair_advantages, metric_means

    def compute_loss(self, model, inputs, return_outputs=False):
        skip_batch = inputs.pop("skip_batch", None)
        if skip_batch is not None:
            skip_value = bool(skip_batch.item()) if isinstance(skip_batch, torch.Tensor) else bool(skip_batch)
            if skip_value:
                logger.warning("Skipping current GRPO batch: CLIP EOS token was truncated or missing.")
                self._reset_rollout_cache()
                return self._make_zero_loss(return_outputs)

        group_ids = inputs["group_ids"]
        pair_roles = inputs["pair_roles"]
        answers = inputs["answers"]
        cache_key = self._make_rollout_cache_key(inputs)

        same_cached_key = self._rollout_cache is not None and self._rollout_cache["key"] == cache_key
        cache_hit = same_cached_key and self._rollout_cache["remaining_reuses"] > 0
        skip_repeated_batch = same_cached_key and not cache_hit and self._rollout_cache.get("skip_remaining", 0) > 0

        if skip_repeated_batch:
            self._rollout_cache["skip_remaining"] -= 1
            if self.args.local_rank in (-1, 0):
                self.log(
                    {
                        "grpo/skipped_low_var_batch": 1.0,
                        "grpo/reuse_target": float(self._rollout_cache.get("desired_uses", 0)),
                        "grpo/reuse_remaining": float(self._rollout_cache["remaining_reuses"]),
                        "grpo/skip_remaining": float(self._rollout_cache["skip_remaining"]),
                        "grpo/low_var_group_rate": self._rollout_cache["reward_metrics"].get("low_var_group_rate", 0.0),
                        "grpo/group_reward_std": self._rollout_cache["reward_metrics"].get("group_reward_std", 0.0),
                        "grpo/group_reward_std_max": self._rollout_cache["reward_metrics"].get("group_reward_std_max", 0.0),
                    }
                )
            return self._make_zero_loss(return_outputs)

        if cache_hit:
            self._rollout_cache["remaining_reuses"] -= 1
            rollout = self._rollout_cache
        else:
            was_training = model.training
            model.eval()
            sampled_completions, completion_texts, prompt_indices = self._sample_completions(model, inputs)
            if was_training:
                model.train()
            if self.args.local_rank in (-1, 0):
                print("\n" + "=" * 80)
                print(f"[GRPO COMPLETIONS] step={self.state.global_step}")
                for i, text in enumerate(completion_texts):
                    print(f"[completion {i}] {repr(text)}")
                print("=" * 80 + "\n")

            expanded_answers = [answers[idx] for idx in prompt_indices]
            pair_rewards, pair_advantages, reward_metrics = self._compute_group_rewards(
                completion_texts, expanded_answers
            )
            expanded_advantages = pair_advantages.repeat_interleave(2).detach()
            desired_uses = self._select_reuse_count(reward_metrics)
            repeat_budget = self._get_repeat_budget()

            (
                full_input_ids,
                full_attention_mask,
                full_gen_mask,
                full_clip_input_ids,
                full_clip_attn_mask,
                full_images,
                full_image_sizes,
            ) = self._build_policy_inputs(inputs, sampled_completions, prompt_indices)

            ref_device = next(self.ref_model.parameters()).device
            if full_input_ids.device != ref_device:
                raise RuntimeError(
                    f"ref_model is on {ref_device}, but inputs are on {full_input_ids.device}"
                )

            model_was_training = model.training
            model.eval()
            with torch.no_grad():
                old_outputs = model(
                    input_ids=full_input_ids,
                    attention_mask=full_attention_mask,
                    clip_input_ids=full_clip_input_ids,
                    clip_attn_mask=full_clip_attn_mask,
                    images=full_images,
                    image_sizes=full_image_sizes,
                    use_cache=False,
                    return_dict=True,
                )
                ref_outputs = self.ref_model(
                    input_ids=full_input_ids,
                    attention_mask=full_attention_mask,
                    clip_input_ids=full_clip_input_ids,
                    clip_attn_mask=full_clip_attn_mask,
                    images=full_images,
                    image_sizes=full_image_sizes,
                    use_cache=False,
                    return_dict=True,
                )
            if model_was_training:
                model.train()

            old_seq_log_probs, _, _, _ = self._compute_seq_log_probs(
                old_outputs.logits, full_input_ids, full_gen_mask
            )
            ref_seq_log_probs, ref_gathered_log_probs, _, _ = self._compute_seq_log_probs(
                ref_outputs.logits, full_input_ids, full_gen_mask
            )

            future_skip_count = repeat_budget - desired_uses if desired_uses > 0 else max(repeat_budget - 1, 0)
            rollout = {
                "key": cache_key,
                "desired_uses": desired_uses,
                "remaining_reuses": max(desired_uses - 1, 0),
                "skip_remaining": max(future_skip_count, 0),
                "group_ids": list(group_ids),
                "pair_roles": list(pair_roles),
                "completion_texts": completion_texts,
                "pair_rewards": pair_rewards.detach(),
                "reward_metrics": reward_metrics,
                "expanded_advantages": expanded_advantages,
                "full_input_ids": full_input_ids,
                "full_attention_mask": full_attention_mask,
                "full_gen_mask": full_gen_mask,
                "full_clip_input_ids": full_clip_input_ids,
                "full_clip_attn_mask": full_clip_attn_mask,
                "full_images": full_images,
                "full_image_sizes": full_image_sizes,
                "old_seq_log_probs": old_seq_log_probs.detach(),
                "ref_seq_log_probs": ref_seq_log_probs.detach(),
                "ref_gathered_log_probs": ref_gathered_log_probs.detach(),
            }
            self._rollout_cache = rollout

            if desired_uses == 0:
                if self.args.local_rank in (-1, 0):
                    self.log(
                        {
                            "grpo/skipped_low_var_batch": 1.0,
                            "grpo/reuse_target": 0.0,
                            "grpo/reuse_remaining": 0.0,
                            "grpo/skip_remaining": float(rollout["skip_remaining"]),
                            "grpo/low_var_group_rate": reward_metrics.get("low_var_group_rate", 0.0),
                            "grpo/group_reward_std": reward_metrics.get("group_reward_std", 0.0),
                            "grpo/group_reward_std_max": reward_metrics.get("group_reward_std_max", 0.0),
                        }
                    )
                return self._make_zero_loss(return_outputs)

        outputs = model(
            input_ids=rollout["full_input_ids"],
            attention_mask=rollout["full_attention_mask"],
            clip_input_ids=rollout["full_clip_input_ids"],
            clip_attn_mask=rollout["full_clip_attn_mask"],
            images=rollout["full_images"],
            image_sizes=rollout["full_image_sizes"],
            use_cache=False,
            return_dict=True,
        )

        seq_log_probs, gathered_log_probs, shift_gen_mask, token_counts = self._compute_seq_log_probs(
            outputs.logits, rollout["full_input_ids"], rollout["full_gen_mask"]
        )

        log_ratio = seq_log_probs - rollout["old_seq_log_probs"]
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(
            ratio,
            1.0 - self.args.grpo_clip_epsilon,
            1.0 + self.args.grpo_clip_epsilon,
        )
        surrogate_unclipped = ratio * rollout["expanded_advantages"]
        surrogate_clipped = clipped_ratio * rollout["expanded_advantages"]
        policy_loss = -torch.min(surrogate_unclipped, surrogate_clipped).mean()

        ref_to_policy_log_ratio = rollout["ref_gathered_log_probs"] - gathered_log_probs
        token_kl = torch.exp(ref_to_policy_log_ratio) - ref_to_policy_log_ratio - 1.0
        seq_kl = (token_kl * shift_gen_mask).sum(dim=-1) / token_counts
        kl_loss = seq_kl.mean()

        loss = policy_loss + self.args.grpo_kl_coef * kl_loss

        clip_fraction = ((ratio - 1.0).abs() > self.args.grpo_clip_epsilon).float().mean()
        approx_kl = ((ratio - 1.0) - log_ratio).mean()

        if self.args.local_rank in (-1, 0):
            self.log(
                {
                    "grpo/loss": float(loss.detach().item()),
                    "grpo/policy_loss": float(policy_loss.detach().item()),
                    "grpo/kl_loss": float(kl_loss.detach().item()),
                    "grpo/reward_mean": float(rollout["pair_rewards"].mean().detach().item()),
                    "grpo/reward_std": float(rollout["pair_rewards"].std(unbiased=False).detach().item()) if rollout["pair_rewards"].numel() > 1 else 0.0,
                    "grpo/ratio_mean": float(ratio.mean().detach().item()),
                    "grpo/clip_fraction": float(clip_fraction.detach().item()),
                    "grpo/approx_kl": float(approx_kl.detach().item()),
                    "grpo/ref_logp_mean": float(rollout["ref_seq_log_probs"].mean().detach().item()),
                    "grpo/reuse_target": float(rollout.get("desired_uses", self._get_repeat_budget())),
                    "grpo/reuse_remaining": float(rollout["remaining_reuses"]),
                    "grpo/skip_remaining": float(rollout.get("skip_remaining", 0)),
                    "grpo/skipped_low_var_batch": 0.0,
                    "grpo/ori_acc": rollout["reward_metrics"]["ori_acc"],
                    "grpo/as_acc": rollout["reward_metrics"]["as_acc"],
                    "grpo/sep": rollout["reward_metrics"]["sep"],
                    "grpo/cross": rollout["reward_metrics"]["cross"],
                    "grpo/empty_rate": rollout["reward_metrics"]["empty_rate"],
                    "grpo/long_rate": rollout["reward_metrics"]["long_rate"],
                    "grpo/repeat_rate": rollout["reward_metrics"]["repeat_rate"],
                    "grpo/format_rate": rollout["reward_metrics"]["format_rate"],
                    "grpo/low_var_group_rate": rollout["reward_metrics"].get("low_var_group_rate", 0.0),
                    "grpo/group_reward_std": rollout["reward_metrics"].get("group_reward_std", 0.0),
                    "grpo/group_reward_std_max": rollout["reward_metrics"].get("group_reward_std_max", 0.0),
                    "grpo/active_group_reward_std_max": rollout["reward_metrics"].get("active_group_reward_std_max", 0.0),
                    "grpo/gen_len": float(token_counts.mean().detach().item()),
                }
            )

        aux_outputs = {
            "group_ids": rollout["group_ids"],
            "pair_roles": rollout["pair_roles"],
            "predictions": rollout["completion_texts"],
            "rewards": rollout["pair_rewards"],
        }
        if return_outputs:
            return loss, aux_outputs
        return loss
