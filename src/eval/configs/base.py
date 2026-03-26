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
    save_predictions_jsonl: bool = False
    predictions_jsonl_name: str = "predictions.jsonl"
    max_question_save_tokens: int | None = 200
    name: str = "Baseline"
    model_config_name: str | None = None
    policy_name: str | None = None
    checkpoint_name: str | None = None

    # model
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int | None = None
    model_name_or_path: str | None = None
    num_generations: int = 1
    self_consistency_num_generations: int = 10
    self_consistency_temperature: float = 0.7
    temperature: float = 0
    max_tokens: int = 4096
    seed: Optional[int] = 42

    # evaluation
    answer_verifier_name: str | None = None
    answer_verifier_args: Dict = field(default_factory=dict)
    evaluator_name: str = "confidence"
    pass_k_vals: List = field(default_factory=list)
    ece_bins: int = 10
    save_reliability_diagram: bool = True

    # inference
    task_spec: str = "generation"
    response_prompt_name: str | None = None
    fine_tuned_dataset: str | None = None
    fine_tuned_algorithm: str | None = None
    inferencer_name: str = "verbalized_confidence"

    # verification
    correctness_fn: str | None = None
    judge_model_name_or_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    judge_gpu_memory_utilization: float = 0.8
    judge_max_model_len: int = 50000
    judge_max_tokens: int = 20
