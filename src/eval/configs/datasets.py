from dataclasses import dataclass


@dataclass
class EvalDatasetConfig:
    dataset_name: str
    dataset_config: str | None = None
    split: str = "test"
    answer_verifier_name: str = "rule_verifier"
    answer_verifier_args: dict = None
    pass_k_vals: list = None

    def __post_init__(self):
        if self.answer_verifier_args is None:
            self.answer_verifier_args = {}
        if self.pass_k_vals is None:
            self.pass_k_vals = []


@dataclass
class TriviaQA(EvalDatasetConfig):
    dataset_name: str = "claytonwang/trivia_eval"
    answer_verifier_name: str = "llm_verifier"


@dataclass
class CommonsenseQA(EvalDatasetConfig):
    dataset_name: str = "claytonwang/commonsenseqa_eval"
    answer_verifier_name: str = "llm_verifier"


@dataclass
class GPQA(EvalDatasetConfig):
    dataset_name: str = "claytonwang/gpqa_eval"
    answer_verifier_name: str = "llm_verifier"


@dataclass
class GSM8K(EvalDatasetConfig):
    dataset_name: str = "claytonwang/gsm8k_eval"
    answer_verifier_name: str = "rule_verifier"


@dataclass
class HotpotRLCR_Eval(EvalDatasetConfig):
    dataset_name: str = "mehuldamani/hotpot_qa"
    answer_verifier_name: str = "rule_verifier"


@dataclass
class HotpotRLCR_Train(EvalDatasetConfig):
    dataset_name: str = "mehuldamani/hotpot_qa"
    split: str = "train"
    answer_verifier_name: str = "rule_verifier"


@dataclass
class HotpotVanilla(EvalDatasetConfig):
    dataset_name: str = "claytonwang/hotpot_qa_vanilla_eval"
    answer_verifier_name: str = "rule_verifier"


@dataclass
class BigMathDigits(EvalDatasetConfig):
    dataset_name: str = "mehuldamani/big-math-digits"
    answer_verifier_name: str = "rule_verifier"


@dataclass
class BigMath_Eval(EvalDatasetConfig):
    dataset_name: str = "mehuldamani/big-math-digits"
    split: str = "test"
    answer_verifier_name: str = "rule_verifier"


@dataclass
class BigMath_Train(EvalDatasetConfig):
    dataset_name: str = "mehuldamani/big-math-digits"
    split: str = "train"
    answer_verifier_name: str = "rule_verifier"


@dataclass
class Math500(EvalDatasetConfig):
    dataset_name: str = "HuggingFaceH4/MATH-500"
    answer_verifier_name: str = "rule_verifier"


@dataclass
class AIME2024(EvalDatasetConfig):
    dataset_name: str = "Maxwell-Jia/AIME_2024"
    split: str = "train"
    answer_verifier_name: str = "rule_verifier"


@dataclass
class MMLUPro(EvalDatasetConfig):
    dataset_name: str = "TIGER-Lab/MMLU-Pro"
    answer_verifier_name: str = "rule_verifier"


@dataclass
class SimpleQA(EvalDatasetConfig):
    dataset_name: str = "basicv8vc/SimpleQA"
    answer_verifier_name: str = "llm_verifier"


@dataclass
class ScienceQA(EvalDatasetConfig):
    dataset_name: str = "derek-thomas/ScienceQA"
    answer_verifier_name: str = "rule_verifier"


@dataclass
class SQuAD(EvalDatasetConfig):
    dataset_name: str = "rajpurkar/squad"
    split: str = "validation"
    answer_verifier_name: str = "rule_verifier"


@dataclass
class BoolQ(EvalDatasetConfig):
    dataset_name: str = "google/boolq"
    split: str = "validation"
    answer_verifier_name: str = "rule_verifier"
