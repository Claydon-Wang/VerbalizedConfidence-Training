from dataclasses import dataclass


@dataclass
class Qwen25_1_5B:
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B"


@dataclass
class Qwen25_1_5B_Instruct:
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass
class Qwen25_7B:
    model_name_or_path: str = "Qwen/Qwen2.5-7B"


@dataclass
class Qwen25_7B_Instruct:
    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"


@dataclass
class Qwen3_1_7B:
    model_name_or_path: str = "Qwen/Qwen3-1.7B"


@dataclass
class Qwen3_4B_Instruct_2507:
    model_name_or_path: str = "Qwen/Qwen3-4B-Instruct-2507"


# Backward-compatible aliases for older CLI invocations.
Qwen25_1_5BInstruct = Qwen25_1_5B_Instruct
Qwen25_7BInstruct = Qwen25_7B_Instruct
Qwen3_4BInstruct_2507 = Qwen3_4B_Instruct_2507
