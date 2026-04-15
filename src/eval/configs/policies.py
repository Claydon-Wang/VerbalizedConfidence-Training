from dataclasses import dataclass


@dataclass
class BasePolicyConfig:
    task_spec: str = "generation"
    correctness_fn: str | None = None
    fine_tuned_dataset: str | None = None
    fine_tuned_algorithm: str | None = None
    response_prompt_name: str | None = None
    inferencer_name: str | None = "verbalized_confidence"


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
class HotpotRLCRSplit(BasePolicyConfig):
    fine_tuned_dataset: str | None = "hotpot"
    fine_tuned_algorithm: str | None = "rlcr_split"


@dataclass
class HotpotRLCRSplitBatch(BasePolicyConfig):
    fine_tuned_dataset: str | None = "hotpot"
    fine_tuned_algorithm: str | None = "rlcr_split_batch"


@dataclass
class HotpotRLCRSplitGlobal(BasePolicyConfig):
    fine_tuned_dataset: str | None = "hotpot"
    fine_tuned_algorithm: str | None = "rlcr_split_global"


@dataclass
class HotpotRLCRSplitGlobalDebias(BasePolicyConfig):
    fine_tuned_dataset: str | None = "hotpot"
    fine_tuned_algorithm: str | None = "rlcr_split_global_debias"


@dataclass
class HotpotRLCRSplitGlobalDebiasNoStd(BasePolicyConfig):
    fine_tuned_dataset: str | None = "hotpot"
    fine_tuned_algorithm: str | None = "rlcr_split_global_debias_nostd"


@dataclass
class HotpotRLCRSplitGlobalRW(BasePolicyConfig):
    fine_tuned_dataset: str | None = "hotpot"
    fine_tuned_algorithm: str | None = "rlcr_split_global_rw"


@dataclass
class HotpotRLCRSplitDAB(BasePolicyConfig):
    fine_tuned_dataset: str | None = "hotpot"
    fine_tuned_algorithm: str | None = "rlcr_split_dab"


@dataclass
class HotpotRLCRSplitCal(BasePolicyConfig):
    fine_tuned_dataset: str | None = "hotpot"
    fine_tuned_algorithm: str | None = "rlcr_split_cal"


@dataclass
class HotpotRLCRSplitDA(BasePolicyConfig):
    fine_tuned_dataset: str | None = "hotpot"
    fine_tuned_algorithm: str | None = "rlcr_split_da"


@dataclass
class HotpotRLCRalpha(BasePolicyConfig):
    fine_tuned_dataset: str | None = "hotpot"
    fine_tuned_algorithm: str | None = "rlcralpha"


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
class HotpotCOCAalpha(BasePolicyConfig):
    fine_tuned_dataset: str | None = "hotpot"
    fine_tuned_algorithm: str | None = "cocaalpha"


@dataclass
class HotpotCoCABayesian(BasePolicyConfig):
    fine_tuned_dataset: str | None = "hotpot"
    fine_tuned_algorithm: str | None = "coca_bayesian"


@dataclass
class HotpotBRPC(BasePolicyConfig):
    fine_tuned_dataset: str | None = "hotpot"
    fine_tuned_algorithm: str | None = "brpc"
    inferencer_name: str | None = "brpc"


@dataclass
class HotpotCOCADifficulty(BasePolicyConfig):
    fine_tuned_dataset: str | None = "hotpot"
    fine_tuned_algorithm: str | None = "coca_difficulty"


@dataclass
class HotpotCOCADA(BasePolicyConfig):
    fine_tuned_dataset: str | None = "hotpot"
    fine_tuned_algorithm: str | None = "coca_da"


@dataclass
class MathRLVR(BasePolicyConfig):
    fine_tuned_dataset: str | None = "math"
    fine_tuned_algorithm: str | None = "rlvr"


@dataclass
class MathRLCR(BasePolicyConfig):
    fine_tuned_dataset: str | None = "math"
    fine_tuned_algorithm: str | None = "rlcr"


@dataclass
class MathRLCRSplit(BasePolicyConfig):
    fine_tuned_dataset: str | None = "math"
    fine_tuned_algorithm: str | None = "rlcr_split"


@dataclass
class MathRLCRSplitBatch(BasePolicyConfig):
    fine_tuned_dataset: str | None = "math"
    fine_tuned_algorithm: str | None = "rlcr_split_batch"


@dataclass
class MathRLCRSplitGlobal(BasePolicyConfig):
    fine_tuned_dataset: str | None = "math"
    fine_tuned_algorithm: str | None = "rlcr_split_global"


@dataclass
class MathRLCRSplitGlobalDebias(BasePolicyConfig):
    fine_tuned_dataset: str | None = "math"
    fine_tuned_algorithm: str | None = "rlcr_split_global_debias"


@dataclass
class MathRLCRSplitGlobalDebiasNoStd(BasePolicyConfig):
    fine_tuned_dataset: str | None = "math"
    fine_tuned_algorithm: str | None = "rlcr_split_global_debias_nostd"


@dataclass
class MathRLCRSplitGlobalRW(BasePolicyConfig):
    fine_tuned_dataset: str | None = "math"
    fine_tuned_algorithm: str | None = "rlcr_split_global_rw"


@dataclass
class MathRLCRSplitDAB(BasePolicyConfig):
    fine_tuned_dataset: str | None = "math"
    fine_tuned_algorithm: str | None = "rlcr_split_dab"


@dataclass
class MathRLCRSplitCal(BasePolicyConfig):
    fine_tuned_dataset: str | None = "math"
    fine_tuned_algorithm: str | None = "rlcr_split_cal"


@dataclass
class MathRLCRSplitDA(BasePolicyConfig):
    fine_tuned_dataset: str | None = "math"
    fine_tuned_algorithm: str | None = "rlcr_split_da"


@dataclass
class MathRLCRalpha(BasePolicyConfig):
    fine_tuned_dataset: str | None = "math"
    fine_tuned_algorithm: str | None = "rlcralpha"


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
class MathCOCAalpha(BasePolicyConfig):
    fine_tuned_dataset: str | None = "math"
    fine_tuned_algorithm: str | None = "cocaalpha"


@dataclass
class MathCoCABayesian(BasePolicyConfig):
    fine_tuned_dataset: str | None = "math"
    fine_tuned_algorithm: str | None = "coca_bayesian"


@dataclass
class MathBRPC(BasePolicyConfig):
    fine_tuned_dataset: str | None = "math"
    fine_tuned_algorithm: str | None = "brpc"
    inferencer_name: str | None = "brpc"


@dataclass
class MathCOCADifficulty(BasePolicyConfig):
    fine_tuned_dataset: str | None = "math"
    fine_tuned_algorithm: str | None = "coca_difficulty"


@dataclass
class MathCOCADA(BasePolicyConfig):
    fine_tuned_dataset: str | None = "math"
    fine_tuned_algorithm: str | None = "coca_da"


@dataclass
class MathRLCRSFT(BasePolicyConfig):
    fine_tuned_dataset: str | None = "math"
    fine_tuned_algorithm: str | None = "rlcr_sft"
