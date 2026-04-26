import importlib
import json
import os
from dataclasses import fields

from src.eval.configs.base import EvalBaseConfig
from transformers import AutoConfig


def detect_tensor_parallel_size() -> int:
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is not None:
        visible = [device.strip() for device in cuda_visible_devices.split(",") if device.strip()]
        if visible:
            return len(visible)

    try:
        import torch

        count = torch.cuda.device_count()
        if count > 0:
            return count
    except Exception:
        pass

    return 1


def load_config_class(module_suffix: str, class_name: str):
    module_name = module_suffix if module_suffix.startswith("src.") else f"src.eval.configs.{module_suffix}"
    module = importlib.import_module(module_name)
    config_cls = getattr(module, class_name)
    return config_cls()

def dataset_name_to_slug(dataset_name: str) -> str:
    return dataset_name.rstrip("/").split("/")[-1]


def load_model_signature(model_name_or_path: str):
    config_path = os.path.join(model_name_or_path, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            model_config = json.load(f)
    else:
        model_config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            local_files_only=True,
        ).to_dict()

    keys = ("model_type", "hidden_size", "num_hidden_layers", "num_attention_heads")
    return {key: model_config.get(key) for key in keys}


def validate_checkpoint_matches_model(model_config, checkpoint_name: str):
    try:
        expected = load_model_signature(model_config.model_name_or_path)
    except Exception:
        return

    try:
        actual = load_model_signature(checkpoint_name)
    except Exception:
        return
    if actual != expected:
        raise ValueError(
            f"Checkpoint {checkpoint_name} does not match model {type(model_config).__name__}. "
            f"Expected {expected}, got {actual}."
        )


def load_config(dataset_name: str, model_name: str, algorithm_name: str, fine_tuned_dataset_name: str | None):
    base_config = EvalBaseConfig()
    dataset_config = load_config_class("datasets", dataset_name)
    model_config = load_config_class("models", model_name)
    algorithm_config = load_config_class("policies", algorithm_name)
    fine_tuned_dataset_config = (
        load_config_class("policies", fine_tuned_dataset_name) if fine_tuned_dataset_name is not None else None
    )
    return base_config, dataset_config, model_config, fine_tuned_dataset_config, algorithm_config


def build_policy_run_name(algorithm_name: str, fine_tuned_dataset: str | None) -> str:
    if algorithm_name == "Baseline" or not fine_tuned_dataset:
        return algorithm_name
    return f"{fine_tuned_dataset}_{algorithm_name}"


def apply_config_overrides(target_config: EvalBaseConfig, source_config, skip_fields: set[str] | None = None):
    skip_fields = skip_fields or set()
    target_field_names = {field.name for field in fields(target_config)}
    for field in fields(source_config):
        if field.name in skip_fields:
            continue
        if field.name in target_field_names:
            setattr(target_config, field.name, getattr(source_config, field.name))
    return target_config


def update_config(
    base_config: EvalBaseConfig,
    dataset_config,
    model_config,
    fine_tuned_dataset_config,
    algorithm_config,
    model_name: str,
    algorithm_name: str,
    checkpoint_name: str | None = None,
    inferencer_name: str | None = None,
    confidence_mode: str | None = None,
    tensor_parallel_size: int | None = None,
):
    config = EvalBaseConfig()
    apply_config_overrides(config, base_config)
    apply_config_overrides(config, dataset_config)
    apply_config_overrides(config, model_config, skip_fields={"fresh"})
    if fine_tuned_dataset_config is not None:
        apply_config_overrides(config, fine_tuned_dataset_config)
    apply_config_overrides(config, algorithm_config)

    config.dataset_config_name = type(dataset_config).__name__
    config.dataset_cls = type(dataset_config).__name__
    config.model_config_name = model_name
    run_name = build_policy_run_name(algorithm_name, config.fine_tuned_dataset)
    config.policy_name = run_name
    config.checkpoint_name = checkpoint_name
    if checkpoint_name is not None:
        config.model_name_or_path = checkpoint_name
    if inferencer_name is not None:
        config.inferencer_name = inferencer_name
    if confidence_mode is not None:
        config.confidence_mode = confidence_mode
    if config.inferencer_name == "self_consistency":
        config.num_generations = config.self_consistency_num_generations
        config.temperature = config.self_consistency_temperature
    output_path = os.path.join(model_name, run_name, config.inferencer_name, config.confidence_mode)
    dataset_store_name = type(dataset_config).__name__
    config.name = run_name
    config.store_name = os.path.join(
        config.logs_root,
        dataset_store_name,
        output_path,
    )
    config.log_path = os.path.join(
        config.logs_root,
        dataset_store_name,
        output_path,
    )
    config.tensor_parallel_size = tensor_parallel_size if tensor_parallel_size is not None else detect_tensor_parallel_size()
    return config


def build_eval_config(
    dataset_name: str,
    model_name: str,
    fine_tuned_algorithm_name: str | None = None,
    fine_tuned_dataset_name: str | None = None,
    checkpoint_name: str | None = None,
    inferencer_name: str | None = None,
    confidence_mode: str | None = None,
    tensor_parallel_size: int | None = None,
):
    if not model_name:
        raise ValueError("Evaluation requires --model <Class>.")
    if fine_tuned_algorithm_name is None:
        fine_tuned_algorithm_name = "Baseline"
    if fine_tuned_algorithm_name != "Baseline" and checkpoint_name is None:
        raise ValueError("Non-baseline evaluation requires --checkpoint <path>.")
    if fine_tuned_algorithm_name != "Baseline" and fine_tuned_dataset_name is None:
        raise ValueError("Non-baseline evaluation requires --fine_tuned_dataset <Hotpot|BigMath|GSM8K>.")
    if tensor_parallel_size is not None and tensor_parallel_size < 1:
        raise ValueError("--tensor_parallel_size must be >= 1.")

    base_config, dataset_config, model_config, fine_tuned_dataset_config, algorithm_config = load_config(
        dataset_name, model_name, fine_tuned_algorithm_name, fine_tuned_dataset_name
    )
    if checkpoint_name is not None:
        validate_checkpoint_matches_model(model_config, checkpoint_name)
    return update_config(
        base_config,
        dataset_config,
        model_config,
        fine_tuned_dataset_config,
        algorithm_config,
        model_name=model_name,
        algorithm_name=fine_tuned_algorithm_name,
        checkpoint_name=checkpoint_name,
        inferencer_name=inferencer_name,
        confidence_mode=confidence_mode,
        tensor_parallel_size=tensor_parallel_size,
    )
