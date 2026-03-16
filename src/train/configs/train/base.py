from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TrainConfig:
    """Composable config that maps onto the TRL training argument schema."""

    logs_root: str = "logs/train"
    run_name: Optional[str] = None
    output_dir: Optional[str] = None

    model_name_or_path: str = "Qwen/Qwen2.5-1.5B"
    model_revision: str = "main"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    trust_remote_code: bool = True

    dataset_name: str = ""
    dataset_config: Optional[str] = None
    dataset_train_split: str = "train"
    dataset_test_split: str = "test"
    train_subset_size: Optional[int] = None
    eval_subset_size: Optional[int] = None

    bf16: bool = True
    beta: float = 0.0
    eval_strategy: str = "steps"
    eval_steps: int = 50
    eval_on_start: bool = False
    format_pattern: str = "ta"
    gradient_accumulation_steps: int = 64
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: dict[str, Any] = field(default_factory=lambda: {"use_reentrant": False})
    hub_strategy: str = "end"
    learning_rate: float = 1e-6
    log_level: str = "info"
    logging_steps: int = 5
    logging_strategy: str = "steps"
    lr_scheduler_type: str = "constant_with_warmup"
    mask_truncated_completions: bool = False
    max_prompt_length: int = 3072
    max_completion_length: int = 1536
    max_steps: float = -1
    num_generations: int = 32
    num_iterations: int = 1
    num_train_epochs: float = 1
    overwrite_output_dir: bool = True
    per_device_eval_batch_size: int = 32
    per_device_train_batch_size: int = 8
    push_to_hub: bool = False
    report_to: list[str] = field(default_factory=lambda: ["swanlab"])
    reward_funcs: list[str] = field(
        default_factory=lambda: ["format", "accuracy", "brier", "mean_confidence", "confidence_one_or_zero"]
    )
    reward_weights: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5, 0.000001, 0.000001])
    save_strategy: str = "steps"
    save_steps: int = 60
    save_total_limit: int = 1
    scale_rewards: bool = False
    seed: int = 43
    sys_prompt_name: str = "gen"
    task_spec: str = "gen"
    temperature: float = 0.7
    use_vllm: bool = True
    vllm_device: str = "auto"
    vllm_gpu_memory_utilization: float = 0.4
    warmup_ratio: float = 0.05
    wandb_project: str = "RLCR"

    extra_args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.output_dir is None and self.run_name is not None:
            self.output_dir = f"{self.logs_root}/{self.run_name}"

    def to_config_dict(self) -> dict[str, Any]:
        config = {
            "run_name": self.run_name,
            "output_dir": self.output_dir,
            "model_name_or_path": self.model_name_or_path,
            "model_revision": self.model_revision,
            "torch_dtype": self.torch_dtype,
            "attn_implementation": self.attn_implementation,
            "trust_remote_code": self.trust_remote_code,
            "dataset_name": self.dataset_name,
            "dataset_config": self.dataset_config,
            "dataset_train_split": self.dataset_train_split,
            "dataset_test_split": self.dataset_test_split,
            "train_subset_size": self.train_subset_size,
            "eval_subset_size": self.eval_subset_size,
            "bf16": self.bf16,
            "beta": self.beta,
            "eval_strategy": self.eval_strategy,
            "eval_steps": self.eval_steps,
            "eval_on_start": self.eval_on_start,
            "format_pattern": self.format_pattern,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "gradient_checkpointing": self.gradient_checkpointing,
            "gradient_checkpointing_kwargs": self.gradient_checkpointing_kwargs,
            "hub_strategy": self.hub_strategy,
            "learning_rate": self.learning_rate,
            "log_level": self.log_level,
            "logging_steps": self.logging_steps,
            "logging_strategy": self.logging_strategy,
            "lr_scheduler_type": self.lr_scheduler_type,
            "mask_truncated_completions": self.mask_truncated_completions,
            "max_prompt_length": self.max_prompt_length,
            "max_completion_length": self.max_completion_length,
            "max_steps": self.max_steps,
            "num_generations": self.num_generations,
            "num_iterations": self.num_iterations,
            "num_train_epochs": self.num_train_epochs,
            "overwrite_output_dir": self.overwrite_output_dir,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "push_to_hub": self.push_to_hub,
            "report_to": self.report_to,
            "reward_funcs": self.reward_funcs,
            "reward_weights": self.reward_weights,
            "save_strategy": self.save_strategy,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "scale_rewards": self.scale_rewards,
            "seed": self.seed,
            "sys_prompt_name": self.sys_prompt_name,
            "task_spec": self.task_spec,
            "temperature": self.temperature,
            "use_vllm": self.use_vllm,
            "vllm_device": self.vllm_device,
            "vllm_gpu_memory_utilization": self.vllm_gpu_memory_utilization,
            "warmup_ratio": self.warmup_ratio,
            "wandb_project": self.wandb_project,
        }
        config.update(self.extra_args)
        return {key: value for key, value in config.items() if value is not None}
