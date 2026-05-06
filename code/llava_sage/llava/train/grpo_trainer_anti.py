from typing import List

import torch

from transformers.trainer import logger

from llava.train.grpo_trainer import LLaVAGRPOTrainer as BaseGRPOTrainer


class LLaVAAntiShortcutGRPOTrainer(BaseGRPOTrainer):
    def _sample_completions(self, model, inputs):
        prompt_input_ids = inputs["prompt_input_ids"]
        prompt_attention_mask = inputs["prompt_attention_mask"]
        clip_input_ids = inputs["clip_input_ids"]
        clip_attn_mask = inputs["clip_attn_mask"]
        images = inputs["images"]
        image_sizes = inputs["image_sizes"]

        batch_size = prompt_input_ids.size(0)
        completions = []
        completion_texts = []
        prompt_indices = []

        for prompt_idx in range(batch_size):
            for _ in range(self.args.grpo_group_size):
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
                        skip_special_tokens=True,
                    ).strip()
                )
                prompt_indices.append(prompt_idx)

        return completions, completion_texts, prompt_indices

    def _compute_group_rewards(self, predictions: List[str], anti_answers: List[str], original_answers: List[str]):
        block_size = self.args.grpo_group_size
        if len(predictions) % block_size != 0:
            raise ValueError(
                f"Predictions must be divisible by grpo_group_size ({block_size})."
            )

        all_rewards = []
        all_advantages = []
        low_var_threshold = max(float(getattr(self.args, "grpo_low_var_threshold", 0.0)), 0.0)

        reward_match_as = float(getattr(self.args, "grpo_reward_match_as", 1.0))
        reward_other = float(getattr(self.args, "grpo_reward_other", 0.2))
        reward_shortcut = float(getattr(self.args, "grpo_reward_shortcut", -1.0))

        metrics = {
            "as_acc": [],
            "other_diff_rate": [],
            "shortcut_rate": [],
            "format_rate": [],
            "group_reward_std": [],
            "active_group_reward_std": [],
            "low_var_group_rate": [],
        }

        for base in range(0, len(predictions), block_size):
            local_rewards = []

            for offset in range(block_size):
                idx = base + offset

                pred_raw = predictions[idx].strip()
                pred = self._normalize_answer(pred_raw)
                ans_as = self._normalize_answer(anti_answers[idx])
                ans_ori = self._normalize_answer(original_answers[idx])

                as_hit = float(pred == ans_as)
                shortcut_hit = float((pred != ans_as) and (pred == ans_ori))
                other_diff_hit = float((pred != ans_as) and (pred != ans_ori))

                format_penalty = 0.0
                if ans_as in {"yes", "no"} and not self._is_yesno(pred):
                    format_penalty += 1.0
                format_penalty += float(self._looks_bad_answer(pred_raw))

                reward = (
                    reward_match_as * as_hit
                    + reward_other * other_diff_hit
                    + reward_shortcut * shortcut_hit
                    - 1.0 * format_penalty
                )

                local_rewards.append(reward)
                metrics["as_acc"].append(as_hit)
                metrics["other_diff_rate"].append(other_diff_hit)
                metrics["shortcut_rate"].append(shortcut_hit)
                metrics["format_rate"].append(format_penalty)

            local_rewards = torch.tensor(
                local_rewards,
                device=self.args.device,
                dtype=torch.float32,
            )
            local_reward_std = float(local_rewards.std(unbiased=False).item()) if local_rewards.numel() > 1 else 0.0
            low_var_group = bool(low_var_threshold > 0.0 and local_reward_std < low_var_threshold)

            if local_rewards.numel() > 1:
                local_advantages = (
                    local_rewards - local_rewards.mean()
                ) / (local_rewards.std(unbiased=False) + self.args.grpo_reward_eps)
            else:
                local_advantages = local_rewards

            if low_var_group:
                local_advantages = torch.zeros_like(local_advantages)

            all_rewards.append(local_rewards)
            all_advantages.append(local_advantages)
            metrics["group_reward_std"].append(local_reward_std)
            metrics["active_group_reward_std"].append(0.0 if low_var_group else local_reward_std)
            metrics["low_var_group_rate"].append(float(low_var_group))

        all_rewards = torch.cat(all_rewards, dim=0)
        all_advantages = torch.cat(all_advantages, dim=0)

        metric_means = {
            key: float(sum(values) / max(len(values), 1))
            for key, values in metrics.items()
        }
        metric_means["group_reward_std_max"] = max(metrics["group_reward_std"]) if metrics["group_reward_std"] else 0.0
        metric_means["active_group_reward_std_max"] = max(metrics["active_group_reward_std"]) if metrics["active_group_reward_std"] else 0.0
        metric_means["all_groups_low_var"] = float(all(value > 0.0 for value in metrics["low_var_group_rate"])) if metrics["low_var_group_rate"] else 0.0

        return all_rewards, all_advantages, metric_means

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
            "as_acc",
            "other_diff_rate",
            "shortcut_rate",
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
                expanded_original_answers = [inputs["original_answers"][idx] for idx in prompt_indices]
                sample_rewards, _, reward_metrics = self._compute_group_rewards(
                    completion_texts,
                    expanded_answers,
                    expanded_original_answers,
                )

                sample_rewards = sample_rewards.detach().float().cpu()
                pair_count = int(sample_rewards.numel())
                if pair_count == 0:
                    skipped_batches += 1
                    continue

                total_pairs += pair_count
                total_groups += batch_group_count
                reward_sum += float(sample_rewards.sum().item())
                reward_sq_sum += float(sample_rewards.pow(2).sum().item())

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

        if total_pairs > 0:
            reward_mean = reward_sum / total_pairs
            reward_var = max(reward_sq_sum / total_pairs - reward_mean * reward_mean, 0.0)
            reward_std = reward_var ** 0.5
        else:
            reward_mean = 0.0
            reward_std = 0.0

        metrics = {
            f"{metric_key_prefix}/num_groups": float(total_groups),
            f"{metric_key_prefix}/num_pairs": float(total_pairs),
            f"{metric_key_prefix}/skipped_batches": float(skipped_batches),
            f"{metric_key_prefix}/reward_mean": float(reward_mean),
            f"{metric_key_prefix}/reward_std": float(reward_std),
        }

        if total_pairs > 0:
            for key in pair_metric_keys:
                metrics[f"{metric_key_prefix}/{key}"] = pair_metric_sums[key] / total_pairs
        else:
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

    def compute_loss(self, model, inputs, return_outputs=False):
        skip_batch = inputs.pop("skip_batch", None)
        if skip_batch is not None:
            skip_value = bool(skip_batch.item()) if isinstance(skip_batch, torch.Tensor) else bool(skip_batch)
            if skip_value:
                logger.warning("Skipping current GRPO batch: CLIP EOS token was truncated or missing.")
                self._reset_rollout_cache()
                return self._make_zero_loss(return_outputs)

        group_ids = inputs["group_ids"]
        answers = inputs["answers"]
        original_answers = inputs["original_answers"]
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
            expanded_original_answers = [original_answers[idx] for idx in prompt_indices]
            sample_rewards, sample_advantages, reward_metrics = self._compute_group_rewards(
                completion_texts,
                expanded_answers,
                expanded_original_answers,
            )
            expanded_advantages = sample_advantages.detach()
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
                "completion_texts": completion_texts,
                "sample_rewards": sample_rewards.detach(),
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
                    "grpo/reward_mean": float(rollout["sample_rewards"].mean().detach().item()),
                    "grpo/reward_std": float(rollout["sample_rewards"].std(unbiased=False).detach().item()) if rollout["sample_rewards"].numel() > 1 else 0.0,
                    "grpo/ratio_mean": float(ratio.mean().detach().item()),
                    "grpo/clip_fraction": float(clip_fraction.detach().item()),
                    "grpo/approx_kl": float(approx_kl.detach().item()),
                    "grpo/ref_logp_mean": float(rollout["ref_seq_log_probs"].mean().detach().item()),
                    "grpo/reuse_target": float(rollout.get("desired_uses", self._get_repeat_budget())),
                    "grpo/reuse_remaining": float(rollout["remaining_reuses"]),
                    "grpo/skip_remaining": float(rollout.get("skip_remaining", 0)),
                    "grpo/skipped_low_var_batch": 0.0,
                    "grpo/as_acc": rollout["reward_metrics"]["as_acc"],
                    "grpo/other_diff_rate": rollout["reward_metrics"]["other_diff_rate"],
                    "grpo/shortcut_rate": rollout["reward_metrics"]["shortcut_rate"],
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
            "predictions": rollout["completion_texts"],
            "rewards": rollout["sample_rewards"],
        }
        return (loss, aux_outputs) if return_outputs else loss
