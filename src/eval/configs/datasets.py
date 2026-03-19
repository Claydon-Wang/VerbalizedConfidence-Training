from dataclasses import dataclass


@dataclass
class EvalDatasetConfig:
    dataset_name: str
    dataset_config: str | None = None
    split: str = "test"
    check_fn: str = "confidence_verifier"
    check_fn_args: dict = None
    pass_k_vals: list = None

    def __post_init__(self):
        if self.check_fn_args is None:
            self.check_fn_args = {}
        if self.pass_k_vals is None:
            self.pass_k_vals = []


@dataclass
class Trivia(EvalDatasetConfig):
    dataset_name: str = "claytonwang/trivia_eval"
    check_fn: str = "llm_confidence_verifier"


@dataclass
class CommonsenseQA(EvalDatasetConfig):
    dataset_name: str = "claytonwang/commonsenseqa_eval"
    check_fn: str = "llm_confidence_verifier"


@dataclass
class GPQA(EvalDatasetConfig):
    dataset_name: str = "claytonwang/gpqa_eval"
    check_fn: str = "llm_confidence_verifier"


@dataclass
class GSM8K(EvalDatasetConfig):
    dataset_name: str = "claytonwang/gsm8k_eval"


@dataclass
class Hotpot(EvalDatasetConfig):
    dataset_name: str = "mehuldamani/hotpot_qa"


@dataclass
class HotpotVanilla(EvalDatasetConfig):
    dataset_name: str = "claytonwang/hotpot_qa_vanilla_eval"


@dataclass
class BigMathDigits(EvalDatasetConfig):
    dataset_name: str = "mehuldamani/big-math-digits"


@dataclass
class Math500(EvalDatasetConfig):
    dataset_name: str = "HuggingFaceH4/MATH-500"


@dataclass
class SimpleQA(EvalDatasetConfig):
    dataset_name: str = "basicv8vc/SimpleQA"
    check_fn: str = "llm_confidence_verifier"
