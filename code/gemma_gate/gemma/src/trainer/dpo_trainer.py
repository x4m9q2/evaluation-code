import os
import torch
from torch import nn
from typing import Union
import torch.nn.functional as F

from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    ALL_LAYERNORM_LAYERS,
    is_peft_available,
    WEIGHTS_NAME,
    TRAINING_ARGS_NAME,
    SAFE_WEIGHTS_NAME,
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
)
import safetensors
from peft import PeftModel
from typing import Optional
from transformers.modeling_utils import PreTrainedModel
from peft import PeftModel
from trl import DPOTrainer
from trl.trainer.utils import pad_to_length, flush_left, selective_log_softmax
from train.train_utils import get_peft_state_non_lora_maybe_zero_3

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

class GemmaDPOTrainer(DPOTrainer):

    def __init__(self, processing_class, *args, **kwargs):
        super(GemmaDPOTrainer, self).__init__(processing_class=processing_class, *args, **kwargs)
        self.processor = processing_class
        self._debug_forward_counter = 0

    def _debug_rank(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        return int(os.environ.get("LOCAL_RANK", 0))

    def _debug_tensor_finite(self, name, tensor):
        if tensor is None:
            return f"{name}=None"
        finite = torch.isfinite(tensor.detach())
        return (
            f"{name}: finite={bool(finite.all().item())} "
            f"nan={int(torch.isnan(tensor.detach()).sum().item())} "
            f"inf={int(torch.isinf(tensor.detach()).sum().item())} "
            f"shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}"
        )

    def _debug_forward_output(self, tag, output):
        if os.environ.get("DPO_DEBUG_FINITE", "0") != "1":
            return
        rank = self._debug_rank()
        self._debug_forward_counter += 1
        parts = [
            self._debug_tensor_finite("chosen_logps", output.get("chosen_logps")),
            self._debug_tensor_finite("rejected_logps", output.get("rejected_logps")),
            self._debug_tensor_finite("chosen_avg_logps", output.get("chosen_avg_logps")),
            self._debug_tensor_finite("rejected_avg_logps", output.get("rejected_avg_logps")),
            self._debug_tensor_finite("mean_chosen_logits", output.get("mean_chosen_logits")),
            self._debug_tensor_finite("mean_rejected_logits", output.get("mean_rejected_logits")),
        ]
        print(f"[dpo-debug][rank={rank}][forward={self._debug_forward_counter}][{tag}] " + " | ".join(parts), flush=True)

    def _prepare_dataset(
        self,
        dataset,
        processing_class,
        args,
        dataset_name
    ):
        return dataset

    def create_optimizer(self):
        """Mirror the SFT optimizer grouping so DPO can keep module-specific LRs."""
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = {name for name in decay_parameters if "bias" not in name}
            named_parameters = list(opt_model.named_parameters())

            def matches_module_name(param_name: str, module_name: str) -> bool:
                return param_name == module_name or param_name.startswith(f"{module_name}.")

            lr_mapper = {}
            if self.args.projector_lr is not None:
                lr_mapper["multi_modal_projector"] = self.args.projector_lr
            if self.args.vision_lr is not None:
                lr_mapper["vision_tower"] = self.args.vision_lr
            if self.args.gate_lr is not None:
                lr_mapper["gate"] = self.args.gate_lr

            optimizer_grouped_parameters = []

            def add_group(group_name, params, weight_decay, lr=None):
                params = list(params)
                if not params:
                    return
                group = {
                    "params": params,
                    "weight_decay": weight_decay,
                    "param_group_name": group_name,
                }
                if lr is not None:
                    group["lr"] = lr
                optimizer_grouped_parameters.append(group)

            if lr_mapper:
                special_lr_parameter_names = {
                    name
                    for name, param in named_parameters
                    if param.requires_grad
                    and any(matches_module_name(name, module_keyword) for module_keyword in lr_mapper)
                }
                add_group(
                    "base_decay",
                    (
                        param
                        for name, param in named_parameters
                        if name in decay_parameters and name not in special_lr_parameter_names and param.requires_grad
                    ),
                    self.args.weight_decay,
                )
                add_group(
                    "base_no_decay",
                    (
                        param
                        for name, param in named_parameters
                        if name not in decay_parameters and name not in special_lr_parameter_names and param.requires_grad
                    ),
                    0.0,
                )
                for module_keyword, lr in lr_mapper.items():
                    module_parameter_names = {
                        name
                        for name, param in named_parameters
                        if param.requires_grad and matches_module_name(name, module_keyword)
                    }
                    add_group(
                        f"{module_keyword}_decay",
                        (
                            param
                            for name, param in named_parameters
                            if name in decay_parameters and name in module_parameter_names and param.requires_grad
                        ),
                        self.args.weight_decay,
                        lr=lr,
                    )
                    add_group(
                        f"{module_keyword}_no_decay",
                        (
                            param
                            for name, param in named_parameters
                            if name not in decay_parameters and name in module_parameter_names and param.requires_grad
                        ),
                        0.0,
                        lr=lr,
                    )
            else:
                add_group(
                    "decay",
                    (param for name, param in named_parameters if name in decay_parameters and param.requires_grad),
                    self.args.weight_decay,
                )
                add_group(
                    "no_decay",
                    (param for name, param in named_parameters if name not in decay_parameters and param.requires_grad),
                    0.0,
                )

            optimizer_cls, optimizer_kwargs = DPOTrainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug("bitsandbytes: will optimize embedding weights in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    @staticmethod
    def concatenated_inputs(
        batch: dict[str, Union[list, torch.LongTensor]], padding_value: int
    ) -> dict[str, torch.LongTensor]:

        concatenated_batch = {}

        concatenated_batch['prompt_input_ids'] = torch.cat([batch["prompt_input_ids"], batch["prompt_input_ids"]], dim=0)
        concatenated_batch['prompt_attention_mask'] = torch.cat([batch["prompt_attention_mask"], batch["prompt_attention_mask"]], dim=0)

        if 'pixel_values' in batch:
            concatenated_batch['pixel_values'] = torch.cat([batch["pixel_values"], batch["pixel_values"]], dim=0)
            concatenated_batch['token_type_ids'] = torch.cat([batch["token_type_ids"], batch["token_type_ids"]], dim=0)

        if 'gate_input_ids' in batch:
            concatenated_batch['gate_input_ids'] = torch.cat([batch["gate_input_ids"], batch["gate_input_ids"]], dim=0)
            concatenated_batch['gate_attention_mask'] = torch.cat([batch["gate_attention_mask"], batch["gate_attention_mask"]], dim=0)

        max_completion_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        concatenated_batch['completion_input_ids'] = torch.cat(
            (
                pad_to_length(batch["chosen_input_ids"], max_completion_length, pad_value=padding_value),
                pad_to_length(batch["rejected_input_ids"], max_completion_length, pad_value=padding_value),
            ),
        )

        concatenated_batch['completion_attention_mask'] = torch.cat(
            (
                pad_to_length(batch["chosen_attention_mask"], max_completion_length, pad_value=0),
                pad_to_length(batch["rejected_attention_mask"], max_completion_length, pad_value=0),
            ),
        )

        return concatenated_batch
    

    def concatenated_forward(self, model: nn.Module, batch: dict[str, Union[list, torch.LongTensor]]):

        num_examples = batch['prompt_input_ids'].shape[0]
        
        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.padding_value)

        model_kwargs = {}

        if self.aux_loss_enabled:
            model_kwargs['output_router_logits'] = True

        # Add image/video values to model kwargs
        if 'pixel_values' in batch:
            model_kwargs['pixel_values'] = concatenated_batch['pixel_values']
            if 'gate_input_ids' in concatenated_batch:
                model_kwargs['gate_input_ids'] = concatenated_batch['gate_input_ids']
                model_kwargs['gate_attention_mask'] = concatenated_batch['gate_attention_mask']

        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]
        
        input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
        attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
        loss_mask = torch.cat(
            (torch.zeros_like(prompt_attention_mask), completion_attention_mask), dim=1
        )
        token_type_ids = None
        pass_token_type_ids = False
        if 'token_type_ids' in concatenated_batch:
            completion_token_type_ids = torch.zeros_like(completion_input_ids)
            token_type_ids = torch.cat((concatenated_batch['token_type_ids'], completion_token_type_ids), dim=1)

        # Flush left to reduce the memory usage
        # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
        #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
        if token_type_ids is not None and not getattr(self.args, "disable_token_type_ids", False):
            attention_mask, input_ids, loss_mask, token_type_ids = flush_left(
                attention_mask, input_ids, loss_mask, token_type_ids
            )
            model_kwargs["token_type_ids"] = token_type_ids
            pass_token_type_ids = True
        else:
            attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)

        model_kwargs["attention_mask"] = attention_mask

        if os.environ.get("DPO_DEBUG_FINITE", "0") == "1":
            rank = self._debug_rank()
            print(
                f"[dpo-debug][rank={rank}] calling {model.__class__.__name__} "
                f"input_ids={tuple(input_ids.shape)} attention_mask={tuple(attention_mask.shape)} "
                f"pixel_values={tuple(model_kwargs['pixel_values'].shape) if 'pixel_values' in model_kwargs else None} "
                f"pass_token_type_ids={pass_token_type_ids} disable_token_type_ids={getattr(self.args, 'disable_token_type_ids', False)}",
                flush=True,
            )

        outputs = model(input_ids, **model_kwargs)
        logits = outputs.logits

        labels = torch.roll(input_ids, shifts=-1, dims=1)
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()

        if logits.shape[:2] != labels.shape[:2]:
            # for llava, the returned logits include the image tokens (placed before the text tokens)
            seq_len = labels.shape[1]
            logits = logits[:, -seq_len:]

        # Compute the log probabilities of the labels
        labels[~loss_mask] = 0  # dummy token; we'll ignore the losses on these tokens later
        per_token_logps = selective_log_softmax(logits, labels)
        per_token_logps[~loss_mask] = 0
        per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)

        all_logps = per_token_logps.sum(-1)
        all_avg_logps = all_logps / loss_mask.sum(-1).clamp_min(1)

        output = {}

        if self.use_weighting:
            with torch.no_grad():
                # Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827
                logprobs = F.log_softmax(logits, dim=-1)
                weights_adjustment_factor = torch.logsumexp(2 * logprobs, dim=-1)  # same as sum(probs**2) in log space
                per_token_logps_adjusted = per_token_logps - weights_adjustment_factor
                all_weights = (per_token_logps_adjusted * loss_mask).sum(-1) / loss_mask.sum(-1)
                chosen_weights = all_weights[:num_examples]
                rejected_weights = all_weights[num_examples:]
                output["policy_weights"] = torch.clamp(torch.exp(chosen_weights + rejected_weights), max=1)

        if self.args.rpo_alpha is not None:
            # Only use the chosen logits for the RPO loss
            chosen_logits = logits[:num_examples]
            chosen_labels = labels[:num_examples]

            # Compute the log probabilities of the labels
            output["nll_loss"] = F.cross_entropy(
                torch.flatten(chosen_logits, end_dim=1), torch.flatten(chosen_labels, end_dim=1), ignore_index=0
            )

        if self.loss_type == "ipo":
            all_logps = all_logps / loss_mask.sum(-1)
            all_avg_logps = all_logps

        output["chosen_logps"] = all_logps[:num_examples]
        output["rejected_logps"] = all_logps[num_examples:]
        output["chosen_avg_logps"] = all_avg_logps[:num_examples]
        output["rejected_avg_logps"] = all_avg_logps[num_examples:]
        output["mean_chosen_logits"] = logits[:num_examples][loss_mask[:num_examples]].mean()
        output["mean_rejected_logits"] = logits[num_examples:][loss_mask[num_examples:]].mean()

        if self.aux_loss_enabled:
            output["aux_loss"] = outputs.aux_loss

        return output

    def _napo_loss_type(self):
        return str(getattr(self.args, "napo_loss_type", "none")).lower()

    def _compute_napo_q(self, logits: torch.Tensor) -> torch.Tensor:
        loss_type = self._napo_loss_type()
        if loss_type == "lq":
            q = torch.as_tensor(getattr(self.args, "napo_q", 1.0), device=logits.device, dtype=logits.dtype)
            return q.clamp(min=0.001, max=1.0)
        if loss_type == "dyn_lq":
            alpha = float(getattr(self.args, "napo_alpha", 0.5))
            with torch.no_grad():
                try:
                    global_mean = self.accelerator.gather_for_metrics(logits.detach()).mean()
                except Exception:
                    global_mean = logits.detach().mean()
                q = 2 * (1 - torch.sigmoid(torch.as_tensor(alpha, device=logits.device, dtype=logits.dtype) * global_mean))
                return q.clamp(min=0.001, max=1.0)
        raise ValueError(f"Unsupported napo_loss_type={loss_type!r}; expected none, lq, or dyn_lq.")

    def dpo_loss(
        self,
        chosen_logps: torch.FloatTensor,
        rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
    ):
        loss_type = self._napo_loss_type()
        if loss_type in ("none", "", "dpo", "sigmoid"):
            return super().dpo_loss(chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps)

        logratios = chosen_logps - rejected_logps
        if self.reference_free:
            ref_logratios = torch.zeros_like(logratios)
        else:
            ref_logratios = ref_chosen_logps - ref_rejected_logps

        logits = logratios.to(self.accelerator.device) - ref_logratios.to(self.accelerator.device)
        q_source = getattr(self, "_napo_q_source_logits", None)
        q = self._compute_napo_q(logits if q_source is None else q_source.to(logits.device))
        probs = torch.sigmoid(self.beta * logits)
        losses = (1 - probs.pow(q)) / q

        chosen_rewards = self.beta * (chosen_logps.to(self.accelerator.device) - ref_chosen_logps.to(self.accelerator.device)).detach()
        rejected_rewards = self.beta * (rejected_logps.to(self.accelerator.device) - ref_rejected_logps.to(self.accelerator.device)).detach()
        self._last_napo_q = float(q.detach().float().item()) if q.numel() == 1 else float(q.detach().float().mean().item())
        return losses, chosen_rewards, rejected_rewards

    def _compute_ref_model_output(self, batch):
        with torch.no_grad():
            if self.ref_model is None:
                with self.null_ref_context():
                    output = self.concatenated_forward(self.model, batch)
                    self._debug_forward_output("ref-null-context", output)
                    return output
            output = self.concatenated_forward(self.ref_model, batch)
            self._debug_forward_output("ref-model", output)
            return output

    def get_batch_loss_metrics(self, model, batch, train_eval="train"):
        metrics = {}
        model_output = self.concatenated_forward(model, batch)
        self._debug_forward_output("policy", model_output)

        if getattr(self.args, "disable_ref_model", False):
            ref_chosen_logps = model_output["chosen_logps"].detach()
            ref_rejected_logps = model_output["rejected_logps"].detach()
            ref_chosen_avg_logps = model_output.get("chosen_avg_logps", ref_chosen_logps).detach()
            ref_rejected_avg_logps = model_output.get("rejected_avg_logps", ref_rejected_logps).detach()
            if os.environ.get("DPO_DEBUG_FINITE", "0") == "1":
                self._debug_forward_output(
                    "ref-disabled-policy-detached",
                    {
                        "chosen_logps": ref_chosen_logps,
                        "rejected_logps": ref_rejected_logps,
                        "chosen_avg_logps": ref_chosen_avg_logps,
                        "rejected_avg_logps": ref_rejected_avg_logps,
                        "mean_chosen_logits": model_output.get("mean_chosen_logits"),
                        "mean_rejected_logits": model_output.get("mean_rejected_logits"),
                    },
                )
        elif "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            ref_chosen_logps = batch["ref_chosen_logps"]
            ref_rejected_logps = batch["ref_rejected_logps"]
            ref_chosen_avg_logps = batch.get("ref_chosen_avg_logps", ref_chosen_logps)
            ref_rejected_avg_logps = batch.get("ref_rejected_avg_logps", ref_rejected_logps)
        else:
            ref_model_output = self._compute_ref_model_output(batch)
            ref_chosen_logps = ref_model_output["chosen_logps"]
            ref_rejected_logps = ref_model_output["rejected_logps"]
            ref_chosen_avg_logps = ref_model_output.get("chosen_avg_logps", ref_chosen_logps)
            ref_rejected_avg_logps = ref_model_output.get("rejected_avg_logps", ref_rejected_logps)

        if self._napo_loss_type() == "dyn_lq" and getattr(self.args, "napo_dyn_q_use_average", False):
            pi_avg_logratios = model_output["chosen_avg_logps"] - model_output["rejected_avg_logps"]
            ref_avg_logratios = ref_chosen_avg_logps - ref_rejected_avg_logps
            self._napo_q_source_logits = pi_avg_logratios - ref_avg_logratios
        else:
            self._napo_q_source_logits = None

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            model_output["chosen_logps"], model_output["rejected_logps"], ref_chosen_logps, ref_rejected_logps
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if self.args.rpo_alpha is not None:
            losses = losses + self.args.rpo_alpha * model_output["nll_loss"]

        if self.use_weighting:
            losses = losses * model_output["policy_weights"]

        if self.aux_loss_enabled:
            losses = losses + self.aux_loss_coef * model_output["aux_loss"]

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = self.accelerator.gather_for_metrics(chosen_rewards).mean().item()
        metrics[f"{prefix}rewards/rejected"] = self.accelerator.gather_for_metrics(rejected_rewards).mean().item()
        metrics[f"{prefix}rewards/accuracies"] = self.accelerator.gather_for_metrics(reward_accuracies).mean().item()
        metrics[f"{prefix}rewards/margins"] = (
            self.accelerator.gather_for_metrics(chosen_rewards - rejected_rewards).mean().item()
        )
        metrics[f"{prefix}logps/chosen"] = (
            self.accelerator.gather_for_metrics(model_output["chosen_logps"]).detach().mean().item()
        )
        metrics[f"{prefix}logps/rejected"] = (
            self.accelerator.gather_for_metrics(model_output["rejected_logps"]).detach().mean().item()
        )
        metrics[f"{prefix}logits/chosen"] = (
            self.accelerator.gather_for_metrics(model_output["mean_chosen_logits"]).detach().mean().item()
        )
        metrics[f"{prefix}logits/rejected"] = (
            self.accelerator.gather_for_metrics(model_output["mean_rejected_logits"]).detach().mean().item()
        )
        if self._napo_loss_type() not in ("none", "", "dpo", "sigmoid"):
            metrics[f"{prefix}napo/q"] = getattr(self, "_last_napo_q", 0.0)
        if self.args.rpo_alpha is not None:
            metrics[f"{prefix}nll_loss"] = (
                self.accelerator.gather_for_metrics(model_output["nll_loss"]).detach().mean().item()
            )
        if self.aux_loss_enabled:
            metrics[f"{prefix}aux_loss"] = (
                self.accelerator.gather_for_metrics(model_output["aux_loss"]).detach().mean().item()
            )

        self._napo_q_source_logits = None
        return losses.mean(), metrics

    def _save_checkpoint(self, model, trial):
        if self.args.lora_enable:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            if self.hp_search_backend is None and trial is None:
                self.store_flos()

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            self.save_model(output_dir, _internal_call=True)

            non_lora_weights = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters(), require_grad_only=False)
            torch.save(non_lora_weights, os.path.join(output_dir, "non_lora_state_dict.bin"))

            if not self.args.save_only_model:
                # Save optimizer and scheduler
                self._save_optimizer_and_scheduler(output_dir)
                # Save RNG state
                self._save_rng_state(output_dir)

            # Save the Trainer state
            if self.args.should_save:
                # Update the `TrainerControl` state to where we are currently
                self.state.stateful_callbacks["TrainerControl"] = self.control.state()
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

            if self.args.push_to_hub:
                self._push_from_checkpoint(output_dir)

            # Maybe delete some older checkpoints.
            if self.args.should_save:
                # Solely rely on numerical checkpoint id for rotation.
                # mtime is not reliable especially on some fuse fs in cloud environments.
                self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)

        else:
            super(GemmaDPOTrainer, self)._save_checkpoint(model, trial)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
            # If we are executing this function, we are the process zero, so we don't check for that.
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Saving model checkpoint to {output_dir}")

            supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
            # Save a trained model and configuration using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            if not isinstance(self.model, supported_classes):
                if state_dict is None:
                    state_dict = self.model.state_dict()

                if isinstance(self.accelerator.unwrap_model(self.model), supported_classes):
                    self.accelerator.unwrap_model(self.model).save_pretrained(
                        output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                    )
                else:
                    logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                    if self.args.save_safetensors:
                        safetensors.torch.save_file(
                            state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                        )
                    else:
                        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
            else:
                self.model.save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )

            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)

            if self.processor is not None:
                self.processor.save_pretrained(output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
