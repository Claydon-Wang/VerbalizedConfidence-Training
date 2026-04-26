import torch

from src.train.trainers.rlcr_split_trainer import RLCRSplitTrainer


class RLCRSplitRandomTargetTrainer(RLCRSplitTrainer):
    trainer_name = "rlcr_split_random_target"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_target_gap = float(getattr(self.args, "random_target_gap", 0.1))
        if self.random_target_gap < 0.0:
            raise ValueError(f"random_target_gap must be >= 0.0, got {self.random_target_gap}.")
        self._random_target_sample_counters = {"train": 0, "eval": 0}

    def _sample_confidence_targets(
        self,
        answer_rewards: torch.Tensor,
        group_success_rate: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        if mode not in self._random_target_sample_counters:
            self._random_target_sample_counters[mode] = 0

        counter = self._random_target_sample_counters[mode]
        self._random_target_sample_counters[mode] += 1

        lower = torch.clamp(group_success_rate - self.random_target_gap, min=0.0)
        upper = torch.clamp(group_success_rate + self.random_target_gap, max=1.0)

        generator = torch.Generator(device="cpu")
        seed = int(self.args.seed) + 1000003 * counter + (0 if mode == "train" else 500000003)
        generator.manual_seed(seed)
        random_draws = torch.rand(answer_rewards.shape, generator=generator, dtype=torch.float32)
        random_draws = random_draws.to(device=answer_rewards.device, dtype=answer_rewards.dtype)

        correct_targets = group_success_rate + random_draws * (upper - group_success_rate)
        wrong_targets = lower + random_draws * (group_success_rate - lower)
        confidence_targets = torch.where(answer_rewards > 0.5, correct_targets, wrong_targets)
        return confidence_targets.clamp_(0.0, 1.0)

    def compute_rewards(self, generation_outputs, inputs):
        reward_outputs = super().compute_rewards(generation_outputs, inputs)
        mode = "train" if self.model.training else "eval"

        answer_rewards = reward_outputs["optimization_rewards_per_func"][:, 1]
        confidence_scores = reward_outputs["confidence_scores"]
        group_success_rate = reward_outputs["group_success_rate"]
        confidence_targets = self._sample_confidence_targets(answer_rewards, group_success_rate, mode)

        raw_negative_brier = -torch.square(confidence_scores - confidence_targets)
        reward_outputs["optimization_rewards_per_func"][:, 2] = raw_negative_brier
        reward_outputs["confidence_rewards"] = self.optimization_rewards["brier"] * raw_negative_brier
        reward_outputs["confidence_targets"] = confidence_targets
        return reward_outputs
