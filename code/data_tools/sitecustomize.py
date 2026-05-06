"""Local runtime compatibility patches for this repo's training scripts."""

try:
    import contextlib
    import torch

    # DeepSpeed in this env expects torch.amp.custom_fwd/custom_bwd, while
    # torch 2.3.1+cu121 only exposes the CUDA AMP variants.
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

    # Accelerate in this env may still try to enter `model.no_sync()` during
    # gradient accumulation. DeepSpeed ZeRO-1/2/3 forbids calling `step()`
    # from that context, so skip no_sync whenever gradients are partitioned.
    try:
        from accelerate import Accelerator

        _orig_accumulate = Accelerator.accumulate

        @contextlib.contextmanager
        def _accumulate_zerostage_compat(self, model):
            self._do_sync()
            if self.sync_gradients:
                context = contextlib.nullcontext
            else:
                use_no_sync = True
                try:
                    # DeepSpeed engines in this environment call `engine.step()`
                    # during backward, so entering `no_sync()` is incompatible
                    # with gradient accumulation across ZeRO stages.
                    if hasattr(model, "zero_optimization_partition_gradients"):
                        use_no_sync = False
                    elif model.__class__.__module__.startswith("deepspeed"):
                        use_no_sync = False
                except Exception:
                    use_no_sync = True
                context = self.no_sync if use_no_sync else contextlib.nullcontext

            with context(model):
                yield

        Accelerator.accumulate = _accumulate_zerostage_compat
        Accelerator._llava_zero_accumulate_patch = _orig_accumulate
    except Exception:
        pass
except Exception:
    # Keep interpreter startup resilient; training scripts will surface the
    # real error later if something else is wrong.
    pass
