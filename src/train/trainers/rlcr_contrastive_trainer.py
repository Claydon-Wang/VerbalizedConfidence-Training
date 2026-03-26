import torch
from accelerate.utils import gather

from src.train.trainers.rlcr_trainer import RLCRTrainer


class RLCRContrastiveTrainer(RLCRTrainer):
    trainer_name = "rlcr_contrastive"

    def validate_reward_specs(self, optimization_rewards, monitoring_rewards):
        super().validate_reward_specs(optimization_rewards, monitoring_rewards)
        if "separation" not in optimization_rewards:
            raise ValueError("RLCRContrastiveTrainer requires 'separation' in optimization_rewards.")

    def _run_separation_reward(self, prompts, completions, completion_ids_list, inputs):
        if self.args.separation_margin < 0.0:
            raise ValueError("RLCRContrastiveTrainer requires separation_margin >= 0.")
        reward_func = self.reward_functions["separation"]
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
        output_reward_func = reward_func(
            prompts=prompts,
            completions=completions,
            completion_ids=completion_ids_list,
            group_size=self.num_generations,
            separation_margin=self.args.separation_margin,
            format_pattern=self.args.format_pattern,
            **reward_kwargs,
        )
        return [reward if reward is not None else torch.nan for reward in output_reward_func]

    def compute_rewards(self, generation_outputs, inputs):
        device = self.accelerator.device
        prompts = generation_outputs["prompts"]
        completions = generation_outputs["completions"]

        optimization_rewards_per_func = torch.zeros(len(prompts), len(self.optimization_reward_names), device=device)
        monitoring_rewards_per_func = torch.zeros(len(prompts), len(self.monitoring_reward_names), device=device)

        separation_rewards = torch.tensor(
            self._run_separation_reward(
                prompts,
                completions,
                generation_outputs["completion_ids_list"],
                inputs,
            ),
            dtype=torch.float32,
            device=device,
        )

        for i, reward_name in enumerate(self.optimization_reward_names):
            if reward_name == "separation":
                optimization_rewards_per_func[:, i] = separation_rewards
            else:
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
            [self.optimization_rewards[name] for name in self.optimization_reward_names],
            dtype=torch.float32,
            device=device,
        )
        rewards = (optimization_rewards_per_func * reward_weights.unsqueeze(0)).sum(dim=1)

        return {
            "optimization_rewards_per_func": optimization_rewards_per_func,
            "monitoring_rewards_per_func": monitoring_rewards_per_func,
            "rewards": rewards,
        }
