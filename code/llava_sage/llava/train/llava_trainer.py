import os
import inspect
import torch
import torch.nn as nn

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import List, Optional


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

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


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        scale = float(getattr(self.args, "lr_scheduler_total_steps_scale", 1.0) or 1.0)
        if scale != 1.0:
            scaled_steps = max(1, int(round(num_training_steps * scale)))
            if self.args.local_rank in (-1, 0):
                logger.info(
                    "Scaling LR scheduler horizon from %s to %s steps (scale=%s).",
                    num_training_steps,
                    scaled_steps,
                    scale,
                )
            num_training_steps = scaled_steps
        return super().create_scheduler(num_training_steps, optimizer=optimizer)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

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
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def training_step(self, model: nn.Module, inputs, num_items_in_batch=None) -> torch.Tensor:
        debug_batch_meta = os.environ.get("DEBUG_BATCH_META", "0") == "1"
        batch_question_ids = inputs.get("question_ids")
        batch_data_sources = inputs.get("data_sources")
        batch_mask_supervisions = inputs.get("mask_supervisions")
        skip_batch = inputs.pop("skip_batch", None)
        if skip_batch is not None:
            if isinstance(skip_batch, torch.Tensor):
                skip_batch = bool(skip_batch.item())
            else:
                skip_batch = bool(skip_batch)
        debug_skip = os.environ.get("DEBUG_SKIP_BATCH", "0") == "1"
        if debug_skip and (self.args.local_rank in (-1, 0)):
            print(f"[debug] skip_batch={skip_batch}")

        labels = inputs.get("labels", None)
        if labels is not None and isinstance(labels, torch.Tensor):
            valid_shift_labels = int(labels[..., 1:].ne(-100).sum().item())
            if valid_shift_labels == 0:
                logger.warning("Skipping current batch: no valid shift labels for causal LM loss.")
                return torch.zeros((), device=self.args.device, requires_grad=True)

        if skip_batch:
            logger.warning("Skipping current batch: CLIP EOS token was truncated or missing.")
            return torch.zeros((), device=self.args.device, requires_grad=True)

        debug_img_token = os.environ.get("DEBUG_IMG_TOKEN", "0") == "1"
        if debug_img_token and (self.args.local_rank in (-1, 0)):
            image_token_count = -1
            has_images = "images" in inputs
            if "input_ids" in inputs and isinstance(inputs["input_ids"], torch.Tensor):
                image_token_count = int(inputs["input_ids"].eq(-200).sum().item())
            print(f"[debug-img] image_token_count={image_token_count} has_images={has_images}")

        parent_training_step = super().training_step
        if "num_items_in_batch" in inspect.signature(parent_training_step).parameters:
            loss = parent_training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        else:
            loss = parent_training_step(model, inputs)

        report_to = self.args.report_to
        if isinstance(report_to, str):
            report_to_targets = [report_to]
        else:
            report_to_targets = list(report_to) if report_to is not None else []
        should_log_wandb = ("wandb" in report_to_targets) and (self.args.local_rank in (-1, 0))
        if should_log_wandb:
            base_model = model.module if hasattr(model, "module") else model
            llava_model = getattr(base_model, "model", None)
            gate_module = getattr(llava_model, "gate", None)
            gate_abs_mean = getattr(gate_module, "last_output_abs_mean", None) if gate_module is not None else None
            gate_patch_var = getattr(gate_module, "last_patch_var", None) if gate_module is not None else None
            mask_patch_suppression_loss = getattr(llava_model, "last_mask_patch_suppression_loss", None)
            mask_patch_coverage_mean = getattr(llava_model, "last_mask_patch_coverage_mean", None)
            weighted_mask_patch_suppression_loss = getattr(llava_model, "last_weighted_mask_patch_suppression_loss", None)
            gate_l1_loss = getattr(llava_model, "last_gate_l1_loss", None)
            weighted_gate_l1_loss = getattr(llava_model, "last_weighted_gate_l1_loss", None)
            last_lm_loss = getattr(llava_model, "last_lm_loss", None)
            if gate_abs_mean is not None and gate_patch_var is not None:
                self.log({
                    "gate/output_abs_mean": float(gate_abs_mean),
                    "gate/patch_var": float(gate_patch_var),
                })
            if gate_l1_loss is not None:
                gate_payload = {"gate/l1_loss": float(gate_l1_loss)}
                if weighted_gate_l1_loss is not None:
                    gate_payload["gate/weighted_l1_loss"] = float(weighted_gate_l1_loss)
                if last_lm_loss is not None and weighted_gate_l1_loss is not None and last_lm_loss != 0:
                    gate_payload["gate/weighted_vs_lm_ratio"] = float(weighted_gate_l1_loss / last_lm_loss)
                self.log(gate_payload)
            if mask_patch_suppression_loss is not None:
                log_payload = {"mask_patch/suppression_loss": float(mask_patch_suppression_loss)}
                if mask_patch_coverage_mean is not None:
                    log_payload["mask_patch/coverage_mean"] = float(mask_patch_coverage_mean)
                if weighted_mask_patch_suppression_loss is not None:
                    log_payload["mask_patch/weighted_suppression_loss"] = float(weighted_mask_patch_suppression_loss)
                if last_lm_loss is not None and weighted_mask_patch_suppression_loss is not None and last_lm_loss != 0:
                    log_payload["mask_patch/weighted_vs_lm_ratio"] = float(weighted_mask_patch_suppression_loss / last_lm_loss)
                self.log(log_payload)

        debug_mask_patch = os.environ.get("DEBUG_MASK_PATCH_LOSS", "0") == "1"
        if debug_mask_patch and (self.args.local_rank in (-1, 0)):
            base_model = model.module if hasattr(model, "module") else model
            llava_model = getattr(base_model, "model", None)
            mask_patch_suppression_loss = getattr(llava_model, "last_mask_patch_suppression_loss", None)
            mask_patch_coverage_mean = getattr(llava_model, "last_mask_patch_coverage_mean", None)
            weighted_mask_patch_suppression_loss = getattr(llava_model, "last_weighted_mask_patch_suppression_loss", None)
            gate_l1_loss = getattr(llava_model, "last_gate_l1_loss", None)
            weighted_gate_l1_loss = getattr(llava_model, "last_weighted_gate_l1_loss", None)
            last_lm_loss = getattr(llava_model, "last_lm_loss", None)
            ratio = None
            if weighted_mask_patch_suppression_loss is not None and last_lm_loss not in (None, 0):
                ratio = weighted_mask_patch_suppression_loss / last_lm_loss
            print(
                f"[debug-mask] suppression_loss={mask_patch_suppression_loss} "
                f"weighted={weighted_mask_patch_suppression_loss} "
                f"gate_l1={gate_l1_loss} "
                f"weighted_gate_l1={weighted_gate_l1_loss} "
                f"lm_loss={last_lm_loss} "
                f"ratio={ratio} "
                f"coverage_mean={mask_patch_coverage_mean}"
            )

        debug_loss_composition = os.environ.get("DEBUG_LOSS_COMPOSITION", "0") == "1"
        if debug_loss_composition and (self.args.local_rank in (-1, 0)):
            batch_summary = ""
            if debug_batch_meta:
                qids = batch_question_ids.detach().cpu().tolist() if isinstance(batch_question_ids, torch.Tensor) else None
                batch_summary = (
                    f" qids={qids} "
                    f"data_sources={batch_data_sources} "
                    f"mask_supervisions={batch_mask_supervisions}"
                )
            base_model = model.module if hasattr(model, "module") else model
            llava_model = getattr(base_model, "model", None)
            mask_patch_suppression_loss = getattr(llava_model, "last_mask_patch_suppression_loss", None)
            mask_patch_coverage_mean = getattr(llava_model, "last_mask_patch_coverage_mean", None)
            weighted_mask_patch_suppression_loss = getattr(llava_model, "last_weighted_mask_patch_suppression_loss", None)
            gate_l1_loss = getattr(llava_model, "last_gate_l1_loss", None)
            weighted_gate_l1_loss = getattr(llava_model, "last_weighted_gate_l1_loss", None)
            last_lm_loss = getattr(llava_model, "last_lm_loss", None)
            total_loss = float(loss.detach().float().item()) if isinstance(loss, torch.Tensor) else loss
            mask_ratio = None
            if weighted_mask_patch_suppression_loss is not None and last_lm_loss not in (None, 0):
                mask_ratio = weighted_mask_patch_suppression_loss / last_lm_loss
            gate_ratio = None
            if weighted_gate_l1_loss is not None and last_lm_loss not in (None, 0):
                gate_ratio = weighted_gate_l1_loss / last_lm_loss
            print(
                f"[loss-composition] total_loss={total_loss} "
                f"lm_loss={last_lm_loss} "
                f"mask_patch_loss={mask_patch_suppression_loss} "
                f"weighted_mask_patch_loss={weighted_mask_patch_suppression_loss} "
                f"mask_weighted_vs_lm_ratio={mask_ratio} "
                f"gate_l1_loss={gate_l1_loss} "
                f"weighted_gate_l1_loss={weighted_gate_l1_loss} "
                f"gate_weighted_vs_lm_ratio={gate_ratio} "
                f"coverage_mean={mask_patch_coverage_mean}"
                f"{batch_summary}"
            )

        debug_grad = os.environ.get("DEBUG_GRAD", "0") == "1"
        if debug_grad and (self.args.local_rank in (-1, 0)):
            named_params = model.named_parameters() if hasattr(model, "named_parameters") else []
            for n, p in named_params:
                if ("model.mm_projector." in n or "model.gate." in n) and p.requires_grad:
                    if p.grad is None:
                        print(f"[debug-grad] {n}: grad=None")
                    else:
                        g = p.grad.detach().float()
                        print(
                            f"[debug-grad] {n}: grad_l2={float(torch.linalg.vector_norm(g).item())} "
                            f"grad_mean_abs={float(g.abs().mean().item())}"
                        )
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        debug_batch_meta = os.environ.get("DEBUG_BATCH_META", "0") == "1"
        batch_question_ids = inputs.pop("question_ids", None)
        batch_data_sources = inputs.pop("data_sources", None)
        batch_answer_types = inputs.pop("answer_types", None)
        batch_mask_supervisions = inputs.pop("mask_supervisions", None)
        inputs.pop("skip_batch", None)
        labels = inputs.get("labels")
        outputs = model(**inputs)
        if isinstance(outputs, dict):
            loss = outputs["loss"] if "loss" in outputs else outputs[0]
            logits = outputs.get("logits", None)
        else:
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            logits = outputs.logits if hasattr(outputs, "logits") else None

        base_model = model.module if hasattr(model, "module") else model
        llava_model = getattr(base_model, "model", None)
        if llava_model is not None:
            llava_model.last_lm_loss = float(loss.detach().float().item()) if isinstance(loss, torch.Tensor) else None
            llava_model.last_weighted_mask_patch_suppression_loss = None
            llava_model.last_weighted_gate_l1_loss = None
        mask_patch_loss_weight = float(getattr(self.args, "mask_patch_loss_weight", 0.0))
        gate_l1_loss_weight = float(getattr(self.args, "gate_l1_loss_weight", 0.0))
        mask_patch_suppression_loss = getattr(
            llava_model, "current_mask_patch_suppression_loss", None
        )
        if mask_patch_suppression_loss is not None and mask_patch_loss_weight > 0:
            weighted_mask_patch_suppression_loss = mask_patch_loss_weight * mask_patch_suppression_loss
            if llava_model is not None:
                llava_model.last_weighted_mask_patch_suppression_loss = float(
                    weighted_mask_patch_suppression_loss.detach().float().item()
                )
            loss = loss + weighted_mask_patch_suppression_loss

        gate_module = getattr(llava_model, "gate", None) if llava_model is not None else None
        gate_l1_loss = getattr(gate_module, "current_gate_l1_loss", None) if gate_module is not None else None
        if llava_model is not None:
            llava_model.last_gate_l1_loss = (
                float(gate_l1_loss.detach().float().item()) if isinstance(gate_l1_loss, torch.Tensor) else None
            )
        if gate_l1_loss is not None and gate_l1_loss_weight > 0:
            weighted_gate_l1_loss = gate_l1_loss_weight * gate_l1_loss
            if llava_model is not None:
                llava_model.last_weighted_gate_l1_loss = float(
                    weighted_gate_l1_loss.detach().float().item()
                )
            loss = loss + weighted_gate_l1_loss

        debug_loss = os.environ.get("DEBUG_LOSS", "0") == "1"
        debug_numerics = os.environ.get("DEBUG_NUMERICS", "0") == "1"
        if debug_loss and (self.args.local_rank in (-1, 0)):
            if isinstance(loss, torch.Tensor):
                loss_value = float(loss.detach().float().cpu().item())
                print(f"[debug] raw_loss={loss_value}")
                if not torch.isfinite(loss.detach()).all():
                    valid_labels = -1
                    shift_valid_labels = -1
                    label_min = None
                    label_max = None
                    if labels is not None and isinstance(labels, torch.Tensor):
                        valid_mask = labels.ne(-100)
                        valid_labels = int(valid_mask.sum().item())
                        shift_valid_labels = int(labels[..., 1:].ne(-100).sum().item())
                        if valid_labels > 0:
                            valid_vals = labels[valid_mask]
                            label_min = int(valid_vals.min().item())
                            label_max = int(valid_vals.max().item())
                    if logits is not None and isinstance(logits, torch.Tensor):
                        logits_f = logits.detach().float()
                        nan_cnt = int(torch.isnan(logits_f).sum().item())
                        inf_cnt = int(torch.isinf(logits_f).sum().item())
                        finite_vals = logits_f[torch.isfinite(logits_f)]
                        if finite_vals.numel() > 0:
                            logit_min = float(finite_vals.min().item())
                            logit_max = float(finite_vals.max().item())
                        else:
                            logit_min = None
                            logit_max = None
                        print(
                            f"[debug] nan_loss_stats valid_labels={valid_labels} "
                            f"shift_valid_labels={shift_valid_labels} "
                            f"label_min={label_min} label_max={label_max} "
                            f"logits_nan={nan_cnt} logits_inf={inf_cnt} "
                            f"logit_min={logit_min} logit_max={logit_max}"
                        )
                    else:
                        print(
                            f"[debug] nan_loss_stats valid_labels={valid_labels} "
                            f"shift_valid_labels={shift_valid_labels} "
                            f"label_min={label_min} label_max={label_max} logits=None"
                        )
                    # Check whether NaN comes from DeepSpeed engine wrapper.
                    if hasattr(model, "module"):
                        with torch.no_grad():
                            base_outputs = model.module(**inputs)
                        base_loss = base_outputs.loss if hasattr(base_outputs, "loss") else base_outputs[0]
                        if isinstance(base_loss, torch.Tensor):
                            print(f"[debug] base_model_loss={float(base_loss.detach().float().cpu().item())}")

        if debug_numerics and (self.args.local_rank in (-1, 0)) and isinstance(loss, torch.Tensor):
            loss_f = loss.detach().float()
            if (not torch.isfinite(loss_f).all()) or float(loss_f.item()) == 0.0:
                batch_summary = ""
                if debug_batch_meta:
                    qids = batch_question_ids.detach().cpu().tolist() if isinstance(batch_question_ids, torch.Tensor) else None
                    batch_summary = (
                        f" qids={qids} "
                        f"data_sources={batch_data_sources} "
                        f"mask_supervisions={batch_mask_supervisions}"
                    )
                valid_labels = -1
                shift_valid_labels = -1
                if labels is not None and isinstance(labels, torch.Tensor):
                    valid_labels = int(labels.ne(-100).sum().item())
                    shift_valid_labels = int(labels[..., 1:].ne(-100).sum().item())
                if logits is not None and isinstance(logits, torch.Tensor):
                    logits_f = logits.detach().float()
                    finite_vals = logits_f[torch.isfinite(logits_f)]
                    logit_min = float(finite_vals.min().item()) if finite_vals.numel() > 0 else None
                    logit_max = float(finite_vals.max().item()) if finite_vals.numel() > 0 else None
                    print(
                        f"[debug-numerics-loss] loss={float(loss_f.item())} "
                        f"valid_labels={valid_labels} shift_valid_labels={shift_valid_labels} "
                        f"logits_nan={int(torch.isnan(logits_f).sum().item())} "
                        f"logits_inf={int(torch.isinf(logits_f).sum().item())} "
                        f"logit_min={logit_min} logit_max={logit_max} "
                        f"last_lm_loss={getattr(llava_model, 'last_lm_loss', None)} "
                        f"mask_patch_loss={getattr(llava_model, 'last_mask_patch_suppression_loss', None)} "
                        f"weighted_mask_patch_loss={getattr(llava_model, 'last_weighted_mask_patch_suppression_loss', None)}"
                        f"{batch_summary}"
                    )
                else:
                    print(
                        f"[debug-numerics-loss] loss={float(loss_f.item())} "
                        f"valid_labels={valid_labels} shift_valid_labels={shift_valid_labels} logits=None "
                        f"last_lm_loss={getattr(llava_model, 'last_lm_loss', None)} "
                        f"mask_patch_loss={getattr(llava_model, 'last_mask_patch_suppression_loss', None)} "
                        f"weighted_mask_patch_loss={getattr(llava_model, 'last_weighted_mask_patch_suppression_loss', None)}"
                        f"{batch_summary}"
                    )

        if return_outputs:
            return (loss, outputs)
        return loss

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler', 'model.gate.']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            save_checkpoint = super(LLaVATrainer, self)._save_checkpoint
            if "metrics" in inspect.signature(save_checkpoint).parameters:
                save_checkpoint(model, trial, metrics)
            else:
                save_checkpoint(model, trial)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)
