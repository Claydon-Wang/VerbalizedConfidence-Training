from dataclasses import dataclass

from src.train.configs.train.base import TrainConfig


@dataclass
class Hotpot(TrainConfig):
    dataset_name: str = "mehuldamani/hotpot_qa"
    max_prompt_length: int = 3072
    max_completion_length: int = 1536
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 32
    num_train_epochs: float = 1
    learning_rate: float = 1e-6
    warmup_ratio: float = 0.05


@dataclass
class Math(TrainConfig):
    dataset_name: str = "mehuldamani/big-math-digits"
    max_prompt_length: int = 1024
    max_completion_length: int = 4096
    per_device_train_batch_size: int = 6
    per_device_eval_batch_size: int = 32
    num_train_epochs: float = 0.5
    learning_rate: float = 5e-6
    lr_scheduler_type: str = "linear"
    mask_truncated_completions: bool = True
    warmup_ratio: float = 0.20


@dataclass
class GSM8K(Math):
    dataset_name: str = "openai/gsm8k"
    dataset_config: str = "main"
    max_completion_length: int = 1024
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 32
    num_train_epochs: float = 2
    gradient_accumulation_steps: int = 64
    num_generations: int = 32
    warmup_ratio: float = 0.1


@dataclass
class HotpotQAHard(Hotpot):
    dataset_name: str = "AAAexp/0413_rlcr_analysis/hotpot_difficulty_subsets/hotpotQA_hard"


@dataclass
class HotpotQAMedium(Hotpot):
    dataset_name: str = "AAAexp/0413_rlcr_analysis/hotpot_difficulty_subsets/hotpotQA_medium"


@dataclass
class HotpotQAEasy(Hotpot):
    dataset_name: str = "AAAexp/0413_rlcr_analysis/hotpot_difficulty_subsets/hotpotQA_easy"
