from dataclasses import dataclass


@dataclass
class BasePolicyConfig:
    sys_prompt_name: str
    inferencer_name: str = "verbalized_confidence"
    task_spec: str = "generation"
    correctness_fn: str | None = None


@dataclass
class Baseline(BasePolicyConfig):
    sys_prompt_name: str = "think_answer_confidence"


@dataclass
class HotpotRLVR(BasePolicyConfig):
    sys_prompt_name: str = "think_answer"


@dataclass
class HotpotRLCR(BasePolicyConfig):
    sys_prompt_name: str = "think_answer_analysis_confidence_detailed"


@dataclass
class MathRLVR(BasePolicyConfig):
    sys_prompt_name: str = "think_answer"


@dataclass
class MathRLCR(BasePolicyConfig):
    sys_prompt_name: str = "think_answer_analysis_confidence"


@dataclass
class MathRLCRSFT(BasePolicyConfig):
    sys_prompt_name: str = "think_answer_analysis_confidence"
