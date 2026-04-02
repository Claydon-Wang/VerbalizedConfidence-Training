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
class RLCRContrastive:
    trainer_name: str = "rlcr_contrastive"
    format_pattern: str = "think_answer_confidence"
    sys_prompt_name: str = "think_answer_confidence"
    optimization_rewards: dict[str, float] = field(
        default_factory=lambda: {"format": 0.5, "accuracy": 0.5, "brier": 0.5, "separation": 0.1}
    )
    monitoring_rewards: list[str] = field(default_factory=lambda: ["mean_confidence", "confidence_one_or_zero"])
    separation_margin: float = 0.1


@dataclass
class MathRLCRContrastive(RLCRContrastive):
    pass


@dataclass
class BAR:
    trainer_name: str = "bar"
    format_pattern: str = "think_answer_confidence"
    sys_prompt_name: str = "think_answer_confidence"
    bar_alpha: float = 0.5
    optimization_rewards: dict[str, float] = field(
        default_factory=lambda: {"format": 0.5, "accuracy": 0.5, "brier": 0.5}
    )
    monitoring_rewards: list[str] = field(default_factory=lambda: ["mean_confidence", "confidence_one_or_zero"])


@dataclass
class MathBAR(BAR):
    pass


@dataclass
class COCA_difficulty:
    trainer_name: str = "coca_difficulty"
    format_pattern: str = "think_answer_difficulty_confidence"
    sys_prompt_name: str = "think_answer_difficulty_confidence"
    learning_rate: float = 1e-6
    temperature: float = 1.0
    optimization_rewards: dict[str, float] = field(
        default_factory=lambda: {"format": 0.5, "accuracy": 0.5, "difficulty": 0.5, "brier": 0.5}
    )
    monitoring_rewards: list[str] = field(default_factory=lambda: ["mean_confidence", "confidence_one_or_zero"])


@dataclass
class MathCOCA_difficulty(COCA_difficulty):
    pass


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
class BRPC:
    trainer_name: str = "brpc"
    format_pattern: str = "think_answer_confidence"
    sys_prompt_name: str = "think_answer_confidence"
    learning_rate: float = 1e-6
    temperature: float = 1.0
    brpc_probe_loss_weight: float = 0.3
    brpc_probe_hidden_size: int = 512
    brpc_detach_hidden_states: bool = True
    brpc_probe_loss_type: str = "bce"
    optimization_rewards: dict[str, float] = field(
        default_factory=lambda: {"format": 0.5, "accuracy": 0.5, "brier": 0.5}
    )
    monitoring_rewards: list[str] = field(default_factory=lambda: ["mean_confidence", "confidence_one_or_zero"])


@dataclass
class RLCRSFT(MathRLCR):
    model_name_or_path: str = "mehuldamani/qwen-base-verifier-sft-v1"
