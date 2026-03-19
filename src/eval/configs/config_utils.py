import importlib
import json
import os
from dataclasses import fields

from src.eval.configs.base import EvalBaseConfig
from transformers import AutoConfig


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


def load_config(dataset_name: str, model_name: str, policy_name: str):
    base_config = EvalBaseConfig()
    dataset_config = load_config_class("datasets", dataset_name)
    model_config = load_config_class("models", model_name)
    policy_config = load_config_class("policies", policy_name)
    return base_config, dataset_config, model_config, policy_config


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
    policy_config,
    model_name: str,
    policy_name: str,
    checkpoint_name: str | None = None,
):
    run_name = policy_name
    output_path = os.path.join(model_name, policy_name)
    config = EvalBaseConfig()
    apply_config_overrides(config, base_config)
    apply_config_overrides(config, dataset_config)
    apply_config_overrides(config, model_config, skip_fields={"fresh"})
    apply_config_overrides(config, policy_config)

    config.dataset_config_name = type(dataset_config).__name__
    config.dataset_cls = type(dataset_config).__name__
    config.model_config_name = model_name
    config.policy_name = policy_name
    config.checkpoint_name = checkpoint_name
    config.name = run_name
    config.store_name = os.path.join(
        config.logs_root,
        dataset_name_to_slug(dataset_config.dataset_name),
        output_path,
    )
    config.log_path = os.path.join(
        config.logs_root,
        dataset_name_to_slug(dataset_config.dataset_name),
        output_path,
    )
    if checkpoint_name is not None:
        config.model_name_or_path = checkpoint_name
    return config


def build_eval_config(
    dataset_name: str,
    model_name: str,
    policy_name: str | None = None,
    checkpoint_name: str | None = None,
):
    if not model_name:
        raise ValueError("Evaluation requires --model <Class>.")
    if policy_name is None:
        policy_name = "Baseline"
    if policy_name != "Baseline" and checkpoint_name is None:
        raise ValueError("Non-baseline evaluation requires --checkpoint <path>.")

    base_config, dataset_config, model_config, policy_config = load_config(
        dataset_name, model_name, policy_name
    )
    if checkpoint_name is not None:
        validate_checkpoint_matches_model(model_config, checkpoint_name)
    return update_config(
        base_config,
        dataset_config,
        model_config,
        policy_config,
        model_name=model_name,
        policy_name=policy_name,
        checkpoint_name=checkpoint_name,
    )
