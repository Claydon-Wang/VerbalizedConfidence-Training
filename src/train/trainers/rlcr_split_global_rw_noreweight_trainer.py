import torch

from src.train.trainers.rlcr_split_global_rw_trainer import RLCRSplitGlobalRWTrainer


class RLCRSplitGlobalRWNoReweightTrainer(RLCRSplitGlobalRWTrainer):
    trainer_name = "rlcr_split_global_rw_noreweight"

    def _compute_grouped_global_rw_confidence_advantages(
        self,
        confidence_rewards: torch.Tensor,
        answer_correctness: torch.Tensor,
        update_ema: bool,
    ):
        device = confidence_rewards.device
        dtype = confidence_rewards.dtype
        answer_correctness = answer_correctness.to(torch.long)

        advantages = torch.zeros_like(confidence_rewards)
        ema_means = {}
        ema_stds = {}
        batch_means = {}
        batch_stds = {}
        batch_props = {}
        balance_weights = {0: 1.0, 1: 1.0}

        total_count = max(int(confidence_rewards.numel()), 1)
        for label in (0, 1):
            mask = answer_correctness == label
            count = int(mask.sum().item())
            batch_props[label] = count / total_count

            if count == 0:
                if not self._confidence_reward_ema_initialized[label]:
                    self._confidence_reward_ema_mean[label] = 0.0
                    self._confidence_reward_ema_sq_mean[label] = self.confidence_std_floor**2
                ema_means[label] = torch.tensor(self._confidence_reward_ema_mean[label], dtype=dtype, device=device)
                ema_stds[label] = torch.tensor(self.confidence_std_floor, dtype=dtype, device=device)
                batch_means[label] = torch.tensor(float("nan"), dtype=dtype, device=device)
                batch_stds[label] = torch.tensor(float("nan"), dtype=dtype, device=device)
                continue

            rewards_group = confidence_rewards[mask]
            batch_mean = rewards_group.mean()
            batch_sq_mean = rewards_group.square().mean()
            batch_std = torch.sqrt(torch.clamp(batch_sq_mean - batch_mean.square(), min=0.0))

            if not self._confidence_reward_ema_initialized[label]:
                self._confidence_reward_ema_mean[label] = float(batch_mean.item())
                self._confidence_reward_ema_sq_mean[label] = float(batch_sq_mean.item())
                self._confidence_reward_ema_initialized[label] = True
            elif update_ema:
                beta = self.confidence_ema_beta
                self._confidence_reward_ema_mean[label] = beta * self._confidence_reward_ema_mean[label] + (
                    1.0 - beta
                ) * float(batch_mean.item())
                self._confidence_reward_ema_sq_mean[label] = beta * self._confidence_reward_ema_sq_mean[label] + (
                    1.0 - beta
                ) * float(batch_sq_mean.item())

            ema_mean = torch.tensor(self._confidence_reward_ema_mean[label], dtype=dtype, device=device)
            ema_sq_mean = torch.tensor(self._confidence_reward_ema_sq_mean[label], dtype=dtype, device=device)
            ema_var = torch.clamp(ema_sq_mean - ema_mean.square(), min=self.confidence_std_floor**2)
            ema_std = torch.sqrt(ema_var)

            group_advantages = (rewards_group - ema_mean) / (ema_std + self.confidence_adv_eps)
            advantages[mask] = group_advantages

            ema_means[label] = ema_mean
            ema_stds[label] = ema_std
            batch_means[label] = batch_mean
            batch_stds[label] = batch_std

        return advantages, ema_means, ema_stds, batch_means, batch_stds, batch_props, balance_weights
