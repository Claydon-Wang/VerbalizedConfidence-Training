from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EvalBaseConfig:
    # dataset
    dataset_config_name: str | None = None
    dataset_cls: str | None = None
    dataset_name: str | None = None
    dataset_config: Optional[str] = None
    split: str = "test"
    sample_size: int | None = None
    fresh: bool = False

    # logging
    logs_root: str = "logs/eval"
    store_name: str | None = None
    log_path: str | None = None
    name: str = "Baseline"
    model_config_name: str | None = None
    policy_name: str | None = None
    checkpoint_name: str | None = None

    # model
    gpu_memory_utilization: float = 0.9
    model_name_or_path: str | None = None
    num_generations: int = 1
    temperature: float = 0
    max_tokens: int = 4096
    seed: Optional[int] = 42

    # evaluation
    check_fn: str | None = None
    check_fn_args: Dict = field(default_factory=dict)
    evaluator_name: str = "confidence"
    pass_k_vals: List = field(default_factory=list)
    ece_bins: int = 10
    save_reliability_diagram: bool = True

    # inference
    task_spec: str = "generation"
    sys_prompt_name: str = "ver"
    inferencer_name: str = "verbalized_confidence"

    # verification
    correctness_fn: str | None = None
    judge_model_name_or_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    judge_gpu_memory_utilization: float = 0.8
    judge_max_tokens: int = 20
