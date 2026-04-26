from src.train.datasets.gsm8k import GSM8K
from src.train.datasets.hotpot import Hotpot
from src.train.datasets.math import Math


DATASET_REGISTRY = {
    "GSM8K": GSM8K,
    "Hotpot": Hotpot,
    "HotpotQAEasy": Hotpot,
    "HotpotQAMedium": Hotpot,
    "HotpotQAHard": Hotpot,
    "Math": Math,
}


def build_dataset(config):
    dataset_cls = DATASET_REGISTRY.get(config.dataset_cls)
    if dataset_cls is None:
        raise KeyError(f"Unknown training dataset class: {config.dataset_cls}")
    return dataset_cls(config).dataset
