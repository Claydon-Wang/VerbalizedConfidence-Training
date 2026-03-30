import torch
from accelerate.utils import gather

from src.train.rewards.reward_functions import extract_confidence_value
from src.train.trainers.rlcr_trainer import RLCRTrainer


class BARTrainer(RLCRTrainer):
    trainer_name = "bar"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bar_alpha = float(getattr(self.args, "bar_alpha", 0.5))
        if not 0.0 <= self.bar_alpha <= 1.0:
            raise ValueError(f"bar_alpha must be in [0, 1], got {self.bar_alpha}.")

    def validate_reward_specs(self, optimization_rewards, monitoring_rewards):
        super().validate_reward_specs(optimization_rewards, monitoring_rewards)
        required_rewards = {"format", "accuracy", "brier"}
        missing_rewards = sorted(required_rewards - set(optimization_rewards))
        if missing_rewards:
            raise ValueError(
                "BARTrainer requires optimization_rewards to include "
                f"{sorted(required_rewards)}. Missing: {missing_rewards}"
            )

    def _parse_confidence_scores(self, completions):
        completion_contents = [completion[0]["content"] for completion in completions]
        scores = []
        for content in completion_contents:
            confidence = extract_confidence_value(content)
            scores.append(0.0 if confidence is None else confidence)
        return scores

    def compute_rewards(self, generation_outputs, inputs):
        reward_outputs = super().compute_rewards(generation_outputs, inputs)
        device = self.accelerator.device
        completions = generation_outputs["completions"]

        reward_name_to_idx = {name: idx for idx, name in enumerate(self.optimization_reward_names)}
        optimization_rewards_per_func = reward_outputs["optimization_rewards_per_func"]
        format_rewards = optimization_rewards_per_func[:, reward_name_to_idx["format"]]
        answer_rewards = optimization_rewards_per_func[:, reward_name_to_idx["accuracy"]]

        confidence_scores_local = torch.tensor(self._parse_confidence_scores(completions), dtype=torch.float32, device=device)
        confidence_scores = gather(confidence_scores_local)

        group_success_rate = answer_rewards.view(-1, self.num_generations).mean(dim=1)
        group_success_rate = group_success_rate.repeat_interleave(self.num_generations, dim=0)
        coupled_targets = self.bar_alpha * answer_rewards + (1.0 - self.bar_alpha) * group_success_rate

        bar_rewards = answer_rewards - torch.square(confidence_scores - coupled_targets)
        bar_rewards = torch.where(format_rewards > 0, bar_rewards, torch.zeros_like(bar_rewards))
        optimization_rewards_per_func[:, reward_name_to_idx["brier"]] = bar_rewards

        reward_weights = torch.tensor(
            [self.optimization_rewards[name] for name in self.optimization_reward_names], dtype=torch.float32, device=device
        )
        reward_outputs["rewards"] = (optimization_rewards_per_func * reward_weights.unsqueeze(0)).sum(dim=1)
        reward_outputs["confidence_scores"] = confidence_scores
        reward_outputs["group_success_rate"] = group_success_rate
        reward_outputs["coupled_targets"] = coupled_targets
        return reward_outputs
