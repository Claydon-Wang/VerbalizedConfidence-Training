import torch

from src.train.trainers.rlcr_split_trainer import RLCRSplitTrainer


class RLCRSplitVarSquareTrainer(RLCRSplitTrainer):
    trainer_name = "rlcr_split_var_square"

    def compute_rewards(self, generation_outputs, inputs):
        reward_outputs = super().compute_rewards(generation_outputs, inputs)

        confidence_scores = reward_outputs["confidence_scores"]
        group_success_rate = reward_outputs["group_success_rate"]
        variance_square_bonus = self.args.variance_square_lambda * torch.square(confidence_scores - group_success_rate)

        reward_outputs["optimization_rewards_per_func"][:, 2] = (
            reward_outputs["optimization_rewards_per_func"][:, 2] + variance_square_bonus
        )
        reward_outputs["confidence_rewards"] = (
            self.optimization_rewards["brier"] * reward_outputs["optimization_rewards_per_func"][:, 2]
        )

        return reward_outputs
