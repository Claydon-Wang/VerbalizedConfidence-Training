import torch
from accelerate.utils import gather_object

from src.train.trainers.rlcr_split_trainer import RLCRSplitTrainer


class RLCRSplitNoStdTrainer(RLCRSplitTrainer):
    trainer_name = "rlcr_split_nostd"

    def _generate_and_score_completions(self, inputs, eval=False):
        mode = "train" if self.model.training else "eval"
        device = self.accelerator.device
        prompt_bundle = self.prepare_prompts(inputs)
        generation_outputs = self.generate_completions(prompt_bundle)
        reward_outputs = self.compute_rewards(generation_outputs, inputs)
        optimization_rewards_per_func = reward_outputs["optimization_rewards_per_func"]
        monitoring_rewards_per_func = reward_outputs["monitoring_rewards_per_func"]

        answer_advantages, answer_group_means, answer_group_stds = self._normalize_group_rewards(
            reward_outputs["answer_rewards"]
        )
        _, confidence_group_means, confidence_group_stds = self._normalize_group_rewards(
            reward_outputs["confidence_rewards"]
        )
        confidence_advantages = reward_outputs["confidence_rewards"] - confidence_group_means

        process_slice = slice(
            self.accelerator.process_index * len(prompt_bundle["prompts"]),
            (self.accelerator.process_index + 1) * len(prompt_bundle["prompts"]),
        )
        answer_advantages = answer_advantages[process_slice]
        confidence_advantages = confidence_advantages[process_slice]

        answer_mask, confidence_mask = self.build_segment_masks(
            generation_outputs["completion_ids"],
            generation_outputs["completion_mask"],
            generation_outputs["completions_text"],
        )

        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather_for_metrics(
                generation_outputs["attention_mask"].sum()
            ).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        agg_completion_mask = self.accelerator.gather_for_metrics(generation_outputs["completion_mask"].sum(1))
        self._metrics[mode]["completions/mean_length"].append(agg_completion_mask.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_mask.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_mask.float().max().item())

        agg_terminated_with_eos = self.accelerator.gather_for_metrics(generation_outputs["is_eos"].any(dim=1))
        term_completion_mask = agg_completion_mask[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_mask) / len(agg_completion_mask)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_mask) == 0:
            term_completion_mask = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_mask.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_mask.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_mask.float().max().item())

        reward_metric_names = self.optimization_reward_names + self.monitoring_reward_names
        reward_metric_values = torch.cat([optimization_rewards_per_func, monitoring_rewards_per_func], dim=1)
        for i, reward_func_name in enumerate(reward_metric_names):
            mean_rewards = torch.nanmean(reward_metric_values[:, i]).item()
            self._metrics[mode][f"reward_values/{reward_func_name}"].append(mean_rewards)

        self._metrics[mode]["reward_total"].append(
            0.5 * (answer_group_means.mean().item() + confidence_group_means.mean().item())
        )
        self._metrics[mode]["reward_std"].append(
            0.5 * (answer_group_stds.mean().item() + confidence_group_stds.mean().item())
        )
        self._metrics[mode]["reward_values/group_success_rate"].append(
            reward_outputs["group_success_rate"].mean().item()
        )
        self._metrics[mode]["reward_values/mean_confidence_prediction"].append(
            reward_outputs["confidence_scores"].mean().item()
        )
        self._metrics[mode]["segments/answer_tokens_mean"].append(answer_mask.sum(1).float().mean().item())
        self._metrics[mode]["segments/confidence_tokens_mean"].append(confidence_mask.sum(1).float().mean().item())
        self._metrics[mode]["reward_values/confidence_advantage_mean_abs"].append(
            confidence_advantages.abs().mean().item()
        )

        num_completions_to_log = self.args.num_completions_to_log
        self._textual_logs["step"].extend([str(self.state.global_step)] * num_completions_to_log)
        self._textual_logs["prompt"].extend(gather_object(prompt_bundle["prompts_text"])[0:num_completions_to_log])
        self._textual_logs["completion"].extend(gather_object(generation_outputs["completions_text"])[0:num_completions_to_log])
        for i, name in enumerate(reward_metric_names):
            self._textual_logs["rewards"][name].extend(reward_metric_values[:, i].tolist()[0:num_completions_to_log])

        return {
            "prompt_ids": generation_outputs["prompt_ids"],
            "prompt_mask": generation_outputs["prompt_mask"],
            "completion_ids": generation_outputs["completion_ids"],
            "completion_mask": generation_outputs["completion_mask"],
            "answer_mask": answer_mask,
            "confidence_mask": confidence_mask,
            "answer_advantages": answer_advantages,
            "confidence_advantages": confidence_advantages,
            "old_per_token_logps": generation_outputs["old_per_token_logps"],
        }
