import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM


def is_gate_key(key: str) -> bool:
    return key.startswith("gate.") or key.startswith("model.gate.") or ".gate." in key


def is_projector_key(key: str) -> bool:
    return (
        key.startswith("mm_projector.")
        or key.startswith("model.mm_projector.")
        or ".mm_projector." in key
    )


def resolve_adapter_path(path: Path) -> Path:
    if path.is_file():
        return path
    candidate = path / "mm_projector.bin"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Adapter path must be a file or a directory containing mm_projector.bin: {path}")


def normalize_adapter_state(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Accept adapter files saved with or without the top-level `model.` prefix."""
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("base_model."):
            new_key = new_key[len("base_model.") :]
        if new_key.startswith("model.model."):
            new_key = "model." + new_key[len("model.model.") :]
        if new_key.startswith("mm_projector.") or new_key.startswith("gate."):
            new_key = "model." + new_key
        normalized[new_key] = value
    return normalized


def parse_gate_flag(flag: str, adapter_has_gate: bool) -> bool:
    if flag == "auto":
        return adapter_has_gate
    if flag == "true":
        if not adapter_has_gate:
            raise ValueError("--force-gate true was requested, but the adapter has no gate weights.")
        return True
    if flag == "false":
        return False
    raise ValueError(f"Unsupported --force-gate value: {flag}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble a new LLaVA base model by injecting projector/gate weights into a base checkpoint."
    )
    parser.add_argument("--base-model-path", required=True, help="Path to the source LLaVA base model.")
    parser.add_argument(
        "--adapter-path",
        required=True,
        help="Path to mm_projector.bin, or a checkpoint directory containing mm_projector.bin.",
    )
    parser.add_argument("--output-path", required=True, help="Directory to save the assembled model.")
    parser.add_argument(
        "--vision-tower-path",
        required=True,
        help="Existing vision tower path used while assembling the model.",
    )
    parser.add_argument(
        "--vision-tower-config-path",
        default=None,
        help="Vision tower path written into the saved config. Defaults to --vision-tower-path.",
    )
    parser.add_argument(
        "--max-shard-size",
        default="5GB",
        help="Shard size used by save_pretrained.",
    )
    parser.add_argument(
        "--force-gate",
        choices=("auto", "true", "false"),
        default="auto",
        help="Whether the assembled config should enable the SAGE gate. auto enables it only when adapter gate weights exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_model_path = Path(args.base_model_path)
    adapter_path = resolve_adapter_path(Path(args.adapter_path))
    output_path = Path(args.output_path)
    vision_tower_path = Path(args.vision_tower_path)
    vision_tower_config_path = args.vision_tower_config_path or str(vision_tower_path)

    if not base_model_path.exists():
        raise FileNotFoundError(base_model_path)
    if not vision_tower_path.exists():
        raise FileNotFoundError(vision_tower_path)
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"Output directory already exists and is not empty: {output_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading adapter weights from {adapter_path}")
    raw_adapter_state = torch.load(str(adapter_path), map_location="cpu")
    if not isinstance(raw_adapter_state, dict):
        raise TypeError(f"Expected a state-dict in {adapter_path}, got {type(raw_adapter_state)!r}")
    adapter_state = normalize_adapter_state(raw_adapter_state)
    adapter_has_gate = any(is_gate_key(key) for key in adapter_state)
    adapter_has_projector = any(is_projector_key(key) for key in adapter_state)
    if not adapter_has_projector:
        raise ValueError(f"No mm_projector weights were found in {adapter_path}")
    use_dual_input_gate = parse_gate_flag(args.force_gate, adapter_has_gate)

    config = LlavaConfig.from_pretrained(str(base_model_path))
    config.mm_vision_tower = str(vision_tower_path)
    config.image_aspect_ratio = "pad"
    config.model_type = "llava_llama"
    config.architectures = ["LlavaLlamaForCausalLM"]
    config.tune_mm_mlp_adapter = False
    config.use_dual_input_gate = bool(use_dual_input_gate)

    print(f"Loading base model from {base_model_path}")
    model = LlavaLlamaForCausalLM.from_pretrained(
        str(base_model_path),
        config=config,
        low_cpu_mem_usage=False,
        torch_dtype=torch.float16,
    )

    missing, unexpected = model.load_state_dict(adapter_state, strict=False)
    print(f"load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
    if unexpected:
        preview = unexpected[:20]
        raise RuntimeError(f"Unexpected adapter keys were not loaded: {preview}")

    model.config.use_dual_input_gate = bool(use_dual_input_gate)
    model.get_model().use_dual_input_gate = bool(use_dual_input_gate)
    model.config.mm_vision_tower = vision_tower_config_path

    tokenizer = AutoTokenizer.from_pretrained(str(base_model_path), use_fast=False)

    print(f"Saving assembled model to {output_path}")
    state_dict_to_save = model.state_dict()
    if not use_dual_input_gate:
        state_dict_to_save = {
            key: value for key, value in state_dict_to_save.items() if not is_gate_key(key)
        }
    model.save_pretrained(
        str(output_path),
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
        state_dict=state_dict_to_save,
    )
    tokenizer.save_pretrained(str(output_path))

    generation_config_path = base_model_path / "generation_config.json"
    if generation_config_path.exists():
        with generation_config_path.open("r", encoding="utf-8") as f:
            generation_config = json.load(f)
        with (output_path / "generation_config.json").open("w", encoding="utf-8") as f:
            json.dump(generation_config, f, ensure_ascii=False, indent=2)

    with (output_path / "assembly_meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "base_model_path": str(base_model_path),
                "adapter_path": str(adapter_path),
                "vision_tower_path": str(vision_tower_path),
                "vision_tower_config_path": vision_tower_config_path,
                "adapter_has_projector": adapter_has_projector,
                "adapter_has_gate": adapter_has_gate,
                "use_dual_input_gate": bool(use_dual_input_gate),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("done")


if __name__ == "__main__":
    main()
