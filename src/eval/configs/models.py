from dataclasses import dataclass


@dataclass
class BaseModelConfig:
    model_name_or_path: str
    num_generations: int = 1
    temperature: float = 0.0
    max_tokens: int = 4096
    seed: int = 42
    task_spec: str = "generation"
    correctness_fn: str | None = None


@dataclass
class Qwen25_1_5B(BaseModelConfig):
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B"


@dataclass
class Qwen25_1_5BInstruct(BaseModelConfig):
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass
class Qwen25_7B(BaseModelConfig):
    model_name_or_path: str = "Qwen/Qwen2.5-7B"


@dataclass
class Qwen25_7BInstruct(BaseModelConfig):
    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
