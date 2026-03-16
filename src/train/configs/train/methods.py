from dataclasses import dataclass, field


@dataclass
class RLVR:
    format_pattern: str = "ta"
    sys_prompt_name: str = "gen"
    reward_weights: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.000001, 0.000001, 0.000001])


@dataclass
class RLCR:
    format_pattern: str = "tabc"
    sys_prompt_name: str = "tabc_long"
    reward_weights: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5, 0.000001, 0.000001])


@dataclass
class MathRLCR:
    format_pattern: str = "tabc"
    sys_prompt_name: str = "tabc"
    reward_weights: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5, 0.00001, 0.00001])


@dataclass
class RLCRSFT(MathRLCR):
    model_name_or_path: str = "mehuldamani/qwen-base-verifier-sft-v1"
