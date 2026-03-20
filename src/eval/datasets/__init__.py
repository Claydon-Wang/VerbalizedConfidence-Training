from src.eval.datasets.big_math_digits import BigMathDigits
from src.eval.datasets.commonsenseqa import CommonsenseQA
from src.eval.datasets.gpqa import GPQA
from src.eval.datasets.gsm8k import GSM8K
from src.eval.datasets.hotpot import Hotpot, HotpotVanilla
from src.eval.datasets.math500 import Math500
from src.eval.datasets.mmlu_pro import MMLUPro
from src.eval.datasets.simpleqa import SimpleQA
from src.eval.datasets.trivia import Trivia


DATASET_REGISTRY = {
    "BigMathDigits": BigMathDigits,
    "CommonsenseQA": CommonsenseQA,
    "GPQA": GPQA,
    "GSM8K": GSM8K,
    "Hotpot": Hotpot,
    "HotpotVanilla": HotpotVanilla,
    "Math500": Math500,
    "MMLUPro": MMLUPro,
    "SimpleQA": SimpleQA,
    "Trivia": Trivia,
    "TriviaQA": Trivia,
}


def build_dataset(config):
    dataset_cls = DATASET_REGISTRY[config.dataset_cls]
    return dataset_cls(config)
