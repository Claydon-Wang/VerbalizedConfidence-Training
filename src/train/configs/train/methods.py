from dataclasses import dataclass, field


@dataclass
class RLVR:
    trainer_name: str = "rlvr"
    format_pattern: str = "ta"
    sys_prompt_name: str = "gen"
    optimization_rewards: dict[str, float] = field(default_factory=lambda: {"format": 0.5, "accuracy": 0.5})
    monitoring_rewards: list[str] = field(
        default_factory=lambda: ["brier", "mean_confidence", "confidence_one_or_zero"]
    )


@dataclass
class RLCR:
    trainer_name: str = "rlcr"
    format_pattern: str = "tabc"
    sys_prompt_name: str = "tabc_long"
    optimization_rewards: dict[str, float] = field(
        default_factory=lambda: {"format": 0.5, "accuracy": 0.5, "brier": 0.5}
    )
    monitoring_rewards: list[str] = field(default_factory=lambda: ["mean_confidence", "confidence_one_or_zero"])


@dataclass
class MathRLCR:
    trainer_name: str = "rlcr"
    format_pattern: str = "tabc"
    sys_prompt_name: str = "tabc"
    optimization_rewards: dict[str, float] = field(
        default_factory=lambda: {"format": 0.5, "accuracy": 0.5, "brier": 0.5}
    )
    monitoring_rewards: list[str] = field(default_factory=lambda: ["mean_confidence", "confidence_one_or_zero"])


@dataclass
class RLCRSFT(MathRLCR):
    model_name_or_path: str = "mehuldamani/qwen-base-verifier-sft-v1"
