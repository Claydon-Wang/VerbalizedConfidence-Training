import importlib
import os
from dataclasses import asdict, fields

from src.train.configs.config_schema import GRPOScriptArguments, GRPOConfig, ModelConfig


def load_config_class(module_suffix, class_name):
    module_name = module_suffix if module_suffix.startswith("src.") else f"src.train.configs.train.{module_suffix}"
    module = importlib.import_module(module_name)
    config_cls = getattr(module, class_name)
    return config_cls()


def config_to_dict(config_obj):
    if hasattr(config_obj, "to_config_dict"):
        return config_obj.to_config_dict()
    return {key: value for key, value in asdict(config_obj).items() if value is not None}


def update_config(config_dict, dataset_name, method_name, model_name):
    dataset_slug = dataset_name.lower()
    method_slug = method_name.lower()
    model_slug = model_name.replace("_", "-").lower()
    config_dict["run_name"] = f"{dataset_slug}-{method_slug}-{model_slug}"
    logs_root = config_dict.pop("logs_root", "logs/train")
    config_dict["output_dir"] = os.path.join(logs_root, dataset_slug, method_slug, model_name.lower())
    return config_dict


def build_train_config(dataset_name, method_name, model_name):
    dataset_obj = load_config_class("datasets", dataset_name)
    method_obj = load_config_class("methods", method_name)
    model_obj = load_config_class("models", model_name)

    config_dict = config_to_dict(dataset_obj)
    config_dict.update(config_to_dict(method_obj))
    config_dict.update(config_to_dict(model_obj))
    return update_config(config_dict, dataset_name, method_name, model_name)


def split_config_dict(config_dict):
    script_keys = {field.name for field in fields(GRPOScriptArguments)}
    training_keys = {field.name for field in fields(GRPOConfig)}
    model_keys = {field.name for field in fields(ModelConfig)}

    script_values = {key: value for key, value in config_dict.items() if key in script_keys}
    training_values = {key: value for key, value in config_dict.items() if key in training_keys}
    model_values = {key: value for key, value in config_dict.items() if key in model_keys}

    unknown_keys = sorted(set(config_dict) - script_keys - training_keys - model_keys)
    if unknown_keys:
        raise ValueError(f"Unknown config keys: {', '.join(unknown_keys)}")

    return (
        GRPOScriptArguments(**script_values),
        GRPOConfig(**training_values),
        ModelConfig(**model_values),
    )
