from dataclasses import dataclass


@dataclass
class BaseFineTunedDatasetConfig:
    fine_tuned_dataset: str | None = None


@dataclass
class Hotpot(BaseFineTunedDatasetConfig):
    fine_tuned_dataset: str | None = "hotpot"


@dataclass
class BigMath(BaseFineTunedDatasetConfig):
    fine_tuned_dataset: str | None = "big_math"


@dataclass
class GSM8K(BaseFineTunedDatasetConfig):
    fine_tuned_dataset: str | None = "gsm8k"


@dataclass
class BaseAlgorithmConfig:
    task_spec: str = "generation"
    correctness_fn: str | None = None
    fine_tuned_algorithm: str | None = None
    response_prompt_name: str | None = None
    inferencer_name: str | None = "verbalized_confidence"


@dataclass
class Baseline(BaseAlgorithmConfig):
    pass


@dataclass
class RLVR(BaseAlgorithmConfig):
    fine_tuned_algorithm: str | None = "rlvr"


@dataclass
class RLCR(BaseAlgorithmConfig):
    fine_tuned_algorithm: str | None = "rlcr"


@dataclass
class RLCRSplit(BaseAlgorithmConfig):
    fine_tuned_algorithm: str | None = "rlcr_split"


@dataclass
class CoCA(BaseAlgorithmConfig):
    fine_tuned_algorithm: str | None = "coca"


@dataclass
class DCPO(BaseAlgorithmConfig):
    fine_tuned_algorithm: str | None = "dcpo"
