import os
import torch
import torch.nn as nn

from transformers import Trainer
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
    ExportableState,
    SaveStrategy
)
import safetensors
from peft import PeftModel
from typing import Optional
import numpy as np
from transformers.processing_utils import ProcessorMixin
from transformers.modeling_utils import PreTrainedModel
from peft import PeftModel
from train.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3

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

class GemmaSFTTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super(GemmaSFTTrainer, self).__init__(*args, **kwargs)
        self._last_loss_breakdown = {}

    @staticmethod
    def _unwrap_model(model):
        while hasattr(model, "module"):
            model = model.module
        return model

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        ce_loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        loss = ce_loss
        raw_model = self._unwrap_model(model)

        breakdown = {
            "loss_ce": float(ce_loss.detach().float().item()),
        }

        gate_module = getattr(raw_model, "gate", None)
        gate_l1_loss = getattr(gate_module, "current_gate_l1_loss", None) if gate_module is not None else None
        gate_l1_loss_weight = float(getattr(self.args, "gate_l1_loss_weight", 0.0))
        if gate_l1_loss is not None and gate_l1_loss_weight > 0:
            weighted_gate_l1_loss = gate_l1_loss_weight * gate_l1_loss
            loss = loss + weighted_gate_l1_loss
            breakdown["loss_gate_l1_raw"] = float(gate_l1_loss.detach().float().item())
            breakdown["loss_gate_l1_weighted"] = float(weighted_gate_l1_loss.detach().float().item())
        else:
            breakdown["loss_gate_l1_raw"] = 0.0
            breakdown["loss_gate_l1_weighted"] = 0.0

        mask_patch_suppression_loss = getattr(raw_model, "current_mask_patch_suppression_loss", None)
        mask_patch_loss_weight = float(getattr(self.args, "mask_patch_loss_weight", 0.0))
        if mask_patch_suppression_loss is not None and mask_patch_loss_weight > 0:
            weighted_mask_patch_loss = mask_patch_loss_weight * mask_patch_suppression_loss
            loss = loss + weighted_mask_patch_loss
            breakdown["loss_mask_raw"] = float(mask_patch_suppression_loss.detach().float().item())
            breakdown["loss_mask_weighted"] = float(weighted_mask_patch_loss.detach().float().item())
        else:
            breakdown["loss_mask_raw"] = 0.0
            breakdown["loss_mask_weighted"] = 0.0

        breakdown["loss_total"] = float(loss.detach().float().item())
        self._last_loss_breakdown = breakdown

        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        if isinstance(logs, dict) and self._last_loss_breakdown:
            merged_logs = dict(logs)
            for key, value in self._last_loss_breakdown.items():
                merged_logs.setdefault(key, value)
            logs = merged_logs
        return super().log(logs, start_time=start_time)

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
            def matches_module_name(param_name: str, module_name: str) -> bool:
                return param_name == module_name or param_name.startswith(f"{module_name}.")

            lr_mapper = {}
            if self.args.projector_lr is not None:
                lr_mapper["multi_modal_projector"] = self.args.projector_lr
            if self.args.vision_lr is not None:
                lr_mapper["vision_tower"] = self.args.vision_lr
            if self.args.gate_lr is not None:
                lr_mapper["gate"] = self.args.gate_lr
            if len(lr_mapper) > 0:
                special_lr_parameters = [
                    name
                    for name, _ in opt_model.named_parameters()
                    if any(matches_module_name(name, module_keyword) for module_keyword in lr_mapper)
                ]
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
                for module_keyword, lr in lr_mapper.items():
                    module_parameters = [
                        name
                        for name, _ in opt_model.named_parameters()
                        if matches_module_name(name, module_keyword)
                    ]
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in module_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in module_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": lr,
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
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
    
    def _maybe_log_save_evaluate(
        self,
        tr_loss,
        grad_norm,
        model,
        trial,
        epoch,
        ignore_keys_for_eval,
        start_time,
        learning_rate=None
    ):
        """
        Overridden method from `Trainer` to do custom logging of each param group.
        """
        # 1) Call the parent version, passing *all* arguments
        super()._maybe_log_save_evaluate(
            tr_loss,
            grad_norm,
            model,
            trial,
            epoch,
            ignore_keys_for_eval,
            start_time,
            learning_rate=learning_rate
        )

        if self.control.should_log:
            logs = {}

            if self.lr_scheduler is not None:
                scheduler_lrs = self.lr_scheduler.get_last_lr()
                logs["learning_rate_base"] = (
                    scheduler_lrs[0] if isinstance(scheduler_lrs, list) else scheduler_lrs
                )

            if self.optimizer is not None:
                for i, param_group in enumerate(self.optimizer.param_groups):
                    group_name = param_group.get("param_group_name", f"group_{i}")
                    logs[f"learning_rate_{group_name}"] = param_group["lr"]

            self.log(logs)

    def _save_checkpoint(self, model, trial):
       # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        if self.args.lora_enable:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            if self.hp_search_backend is None and trial is None:
                self.store_flos()

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            self.save_model(output_dir, _internal_call=True)
            non_lora_weights = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters(), require_grad_only=False)
            torch.save(non_lora_weights, os.path.join(output_dir, "non_lora_state_dict.bin"))

            if self.args.save_strategy in [SaveStrategy.STEPS, SaveStrategy.EPOCH] and self.state.best_global_step:
                best_checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.best_global_step}"
                best_checkpoint_dir = os.path.join(run_dir, best_checkpoint_folder)

                if os.path.exists(best_checkpoint_dir):
                    self.state.best_model_checkpoint = best_checkpoint_dir

            if not self.args.save_only_model:
                # Save optimizer and scheduler
                self._save_optimizer_and_scheduler(output_dir)
                self._save_scaler(output_dir)
                # Save RNG state
                self._save_rng_state(output_dir)

            # Save the Trainer state
            if self.args.should_save:
                # Update `ExportableState` callbacks and `TrainerControl` state to where we are currently
                for cb in [
                    cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
                ]:
                    cb_name = cb.__class__.__name__
                    cb_state = cb.state()
                    if isinstance(self.state.stateful_callbacks[cb_name], list):
                        self.state.stateful_callbacks[cb_name].append(cb_state)
                    else:
                        self.state.stateful_callbacks[cb_name] = cb_state
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

            if self.args.push_to_hub:
                self._push_from_checkpoint(output_dir)
        else:
            super(GemmaSFTTrainer, self)._save_checkpoint(model, trial)

    # def training_step(self, model, inputs):
    #     for name, param in model.named_parameters():
    #         if 'vision_model' in name and param.requires_grad:
    #             print(f"Training parameter {name}")
            
    #         elif 'img_projection' in name and param.requires_grad:
    #             print(f"Training parameter {name}")
    #     return super().training_step(model, inputs)
