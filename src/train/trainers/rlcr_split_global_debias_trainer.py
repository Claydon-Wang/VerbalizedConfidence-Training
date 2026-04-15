import torch

from src.train.trainers.rlcr_split_global_trainer import RLCRSplitGlobalTrainer


class RLCRSplitGlobalDebiasTrainer(RLCRSplitGlobalTrainer):
    trainer_name = "rlcr_split_global_debias"

    confidence_ema_beta = 0.99
    confidence_std_floor = 0.05
    confidence_adv_eps = 1e-4

    def compute_rewards(self, generation_outputs, inputs):
        reward_outputs = super().compute_rewards(generation_outputs, inputs)

        answer_rewards = reward_outputs["optimization_rewards_per_func"][:, 1]
        confidence_scores = reward_outputs["confidence_scores"]
        group_success_rate = reward_outputs["group_success_rate"]

        raw_negative_brier = -torch.square(answer_rewards - confidence_scores)
        label_variance = group_success_rate * (1.0 - group_success_rate)
        debiased_confidence_reward = raw_negative_brier + label_variance

        reward_outputs["optimization_rewards_per_func"][:, 2] = debiased_confidence_reward
        reward_outputs["confidence_rewards"] = self.optimization_rewards["brier"] * debiased_confidence_reward

        return reward_outputs
