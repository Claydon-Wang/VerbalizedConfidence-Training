from dataclasses import dataclass


@dataclass
class Qwen25_1_5B:
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B"


@dataclass
class Qwen25_1_5BInstruct:
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass
class Qwen25_7B:
    model_name_or_path: str = "Qwen/Qwen2.5-7B"


@dataclass
class Qwen25_7BInstruct:
    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
