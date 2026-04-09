from src.train.trainers.grpo_trainer import BaseGRPOTrainer
import torch
from accelerate.utils import gather


class RLCRTrainer(BaseGRPOTrainer):
    trainer_name = "rlcr"

    def validate_reward_specs(self, optimization_rewards, monitoring_rewards):
        super().validate_reward_specs(optimization_rewards, monitoring_rewards)
        if "brier" not in optimization_rewards and "alpha_score" not in optimization_rewards:
            raise ValueError("RLCRTrainer requires 'brier' or 'alpha_score' in optimization_rewards.")

    def compute_rewards(self, generation_outputs, inputs):
        device = self.accelerator.device
        prompts = generation_outputs["prompts"]
        completions = generation_outputs["completions"]
        optimization_rewards_per_func = torch.zeros(len(prompts), len(self.optimization_reward_names), device=device)
        monitoring_rewards_per_func = torch.zeros(len(prompts), len(self.monitoring_reward_names), device=device)

        for i, reward_name in enumerate(self.optimization_reward_names):
            optimization_rewards_per_func[:, i] = self.run_reward_function(
                reward_name,
                prompts,
                completions,
                generation_outputs["completion_ids_list"],
                inputs,
            )

        for i, reward_name in enumerate(self.monitoring_reward_names):
            monitoring_rewards_per_func[:, i] = self.run_reward_function(
                reward_name,
                prompts,
                completions,
                generation_outputs["completion_ids_list"],
                inputs,
            )

        optimization_rewards_per_func = gather(optimization_rewards_per_func)
        if self.monitoring_reward_names:
            monitoring_rewards_per_func = gather(monitoring_rewards_per_func)

        reward_weights = torch.tensor(
            [self.optimization_rewards[name] for name in self.optimization_reward_names], dtype=torch.float32, device=device
        )
        rewards = (optimization_rewards_per_func * reward_weights.unsqueeze(0)).sum(dim=1)
        return {
            "optimization_rewards_per_func": optimization_rewards_per_func,
            "monitoring_rewards_per_func": monitoring_rewards_per_func,
            "rewards": rewards,
        }
