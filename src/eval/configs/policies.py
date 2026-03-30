from dataclasses import dataclass


@dataclass
class BasePolicyConfig:
    task_spec: str = "generation"
    correctness_fn: str | None = None
    fine_tuned_dataset: str | None = None
    fine_tuned_algorithm: str | None = None
    response_prompt_name: str | None = None
    inferencer_name: str | None = None


@dataclass
class Baseline(BasePolicyConfig):
    pass


@dataclass
class HotpotRLVR(BasePolicyConfig):
    fine_tuned_dataset: str | None = "hotpot"
    fine_tuned_algorithm: str | None = "rlvr"


@dataclass
class HotpotRLCR(BasePolicyConfig):
    fine_tuned_dataset: str | None = "hotpot"
    fine_tuned_algorithm: str | None = "rlcr"


@dataclass
class HotpotRLCRContrastive(BasePolicyConfig):
    fine_tuned_dataset: str | None = "hotpot"
    fine_tuned_algorithm: str | None = "rlcr_contrastive"


@dataclass
class HotpotBAR(BasePolicyConfig):
    fine_tuned_dataset: str | None = "hotpot"
    fine_tuned_algorithm: str | None = "bar"


@dataclass
class HotpotCoCA(BasePolicyConfig):
    fine_tuned_dataset: str | None = "hotpot"
    fine_tuned_algorithm: str | None = "coca"


@dataclass
class HotpotCoCABayesian(BasePolicyConfig):
    fine_tuned_dataset: str | None = "hotpot"
    fine_tuned_algorithm: str | None = "coca_bayesian"


@dataclass
class HotpotCOCADifficulty(BasePolicyConfig):
    fine_tuned_dataset: str | None = "hotpot"
    fine_tuned_algorithm: str | None = "coca_difficulty"


@dataclass
class MathRLVR(BasePolicyConfig):
    fine_tuned_dataset: str | None = "math"
    fine_tuned_algorithm: str | None = "rlvr"


@dataclass
class MathRLCR(BasePolicyConfig):
    fine_tuned_dataset: str | None = "math"
    fine_tuned_algorithm: str | None = "rlcr"


@dataclass
class MathRLCRContrastive(BasePolicyConfig):
    fine_tuned_dataset: str | None = "math"
    fine_tuned_algorithm: str | None = "rlcr_contrastive"


@dataclass
class MathBAR(BasePolicyConfig):
    fine_tuned_dataset: str | None = "math"
    fine_tuned_algorithm: str | None = "bar"


@dataclass
class MathCoCA(BasePolicyConfig):
    fine_tuned_dataset: str | None = "math"
    fine_tuned_algorithm: str | None = "coca"


@dataclass
class MathCoCABayesian(BasePolicyConfig):
    fine_tuned_dataset: str | None = "math"
    fine_tuned_algorithm: str | None = "coca_bayesian"


@dataclass
class MathCOCADifficulty(BasePolicyConfig):
    fine_tuned_dataset: str | None = "math"
    fine_tuned_algorithm: str | None = "coca_difficulty"


@dataclass
class MathRLCRSFT(BasePolicyConfig):
    fine_tuned_dataset: str | None = "math"
    fine_tuned_algorithm: str | None = "rlcr_sft"
