from dataclasses import dataclass, field


@dataclass
class RLVR:
    trainer_name: str = "rlvr"
    format_pattern: str = "think_answer"
    sys_prompt_name: str = "think_answer"
    optimization_rewards: dict[str, float] = field(default_factory=lambda: {"format": 0.5, "accuracy": 0.5})
    monitoring_rewards: list[str] = field(
        default_factory=lambda: ["brier", "mean_confidence", "confidence_one_or_zero"]
    )


@dataclass
class RLCR:
    trainer_name: str = "rlcr"
    format_pattern: str = "think_answer_confidence"
    sys_prompt_name: str = "think_answer_confidence"
    optimization_rewards: dict[str, float] = field(
        default_factory=lambda: {"format": 0.5, "accuracy": 0.5, "brier": 0.5}
    )
    monitoring_rewards: list[str] = field(default_factory=lambda: ["mean_confidence", "confidence_one_or_zero"])


@dataclass
class MathRLCR:
    trainer_name: str = "rlcr"
    format_pattern: str = "think_answer_confidence"
    sys_prompt_name: str = "think_answer_confidence"
    optimization_rewards: dict[str, float] = field(
        default_factory=lambda: {"format": 0.5, "accuracy": 0.5, "brier": 0.5}
    )
    monitoring_rewards: list[str] = field(default_factory=lambda: ["mean_confidence", "confidence_one_or_zero"])


@dataclass
class CoCA:
    trainer_name: str = "coca"
    format_pattern: str = "think_answer_confidence"
    sys_prompt_name: str = "think_answer_confidence"
    learning_rate: float = 1e-6
    temperature: float = 1.0
    optimization_rewards: dict[str, float] = field(
        default_factory=lambda: {"format": 0.5, "accuracy": 0.5, "brier": 0.5}
    )
    monitoring_rewards: list[str] = field(default_factory=lambda: ["mean_confidence", "confidence_one_or_zero"])


@dataclass
class MathCoCA:
    trainer_name: str = "coca"
    format_pattern: str = "think_answer_confidence"
    sys_prompt_name: str = "think_answer_confidence"
    learning_rate: float = 1e-6
    temperature: float = 1.0
    optimization_rewards: dict[str, float] = field(
        default_factory=lambda: {"format": 0.5, "accuracy": 0.5, "brier": 0.5}
    )
    monitoring_rewards: list[str] = field(default_factory=lambda: ["mean_confidence", "confidence_one_or_zero"])


@dataclass
class CoCABayesian:
    trainer_name: str = "coca_bayesian"
    format_pattern: str = "think_answer_confidence"
    sys_prompt_name: str = "think_answer_confidence"
    learning_rate: float = 1e-6
    temperature: float = 1.0
    bayesian_alpha_ratio: float = 1.0
    bayesian_prior_path: str = "logs/eval/HotpotRLCR_Train/Qwen25_1_5B_Instruct/Baseline/self_consistency/calibration/isotonic_regression/calibrated_predictions.jsonl"
    bayesian_eval_prior_path: str = "logs/eval/HotpotRLCR_Eval/Qwen25_1_5B_Instruct/Baseline/self_consistency/calibration/isotonic_regression/calibrated_predictions.jsonl"
    bayesian_prior_reduce: str = "mean"
    optimization_rewards: dict[str, float] = field(
        default_factory=lambda: {"format": 0.5, "accuracy": 0.5, "brier": 0.5}
    )
    monitoring_rewards: list[str] = field(default_factory=lambda: ["mean_confidence", "confidence_one_or_zero"])


@dataclass
class RLCRSFT(MathRLCR):
    model_name_or_path: str = "mehuldamani/qwen-base-verifier-sft-v1"
