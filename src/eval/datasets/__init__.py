from src.eval.datasets.big_math_digits import BigMathDigits
from src.eval.datasets.boolq import BoolQ
from src.eval.datasets.commonsenseqa import CommonsenseQA
from src.eval.datasets.gpqa import GPQA
from src.eval.datasets.gsm8k import GSM8K
from src.eval.datasets.hotpot import HotpotRLCR_Eval, HotpotRLCR_Train, HotpotVanilla
from src.eval.datasets.math500 import Math500
from src.eval.datasets.mmlu_pro import MMLUPro
from src.eval.datasets.scienceqa import ScienceQA
from src.eval.datasets.simpleqa import SimpleQA
from src.eval.datasets.squad import SQuAD
from src.eval.datasets.trivia import Trivia


DATASET_REGISTRY = {
    "BigMathDigits": BigMathDigits,
    "BoolQ": BoolQ,
    "CommonsenseQA": CommonsenseQA,
    "GPQA": GPQA,
    "GSM8K": GSM8K,
    "HotpotRLCR_Eval": HotpotRLCR_Eval,
    "HotpotRLCR_Train": HotpotRLCR_Train,
    "HotpotVanilla": HotpotVanilla,
    "Math500": Math500,
    "MMLUPro": MMLUPro,
    "ScienceQA": ScienceQA,
    "SimpleQA": SimpleQA,
    "SQuAD": SQuAD,
    "Trivia": Trivia,
    "TriviaQA": Trivia,
}


def build_dataset(config):
    dataset_cls = DATASET_REGISTRY[config.dataset_cls]
    return dataset_cls(config)
