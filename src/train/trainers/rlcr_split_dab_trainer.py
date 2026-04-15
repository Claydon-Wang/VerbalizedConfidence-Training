import torch
from accelerate.utils import gather

from src.train.trainers.rlcr_split_trainer import RLCRSplitTrainer


class RLCRSplitDABTrainer(RLCRSplitTrainer):
    trainer_name = "rlcr_split_dab"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dab_lambda = float(getattr(self.args, "dab_lambda", 0.5))

    def compute_rewards(self, generation_outputs, inputs):
        device = self.accelerator.device
        prompts = generation_outputs["prompts"]
        completions = generation_outputs["completions"]

        format_rewards_local = self.run_reward_function(
            "format",
            prompts,
            completions,
            generation_outputs["completion_ids_list"],
            inputs,
        )
        format_rewards_local = torch.as_tensor(format_rewards_local, dtype=torch.float32, device=device)
        format_rewards = gather(format_rewards_local)

        answers = [example["answer"] for example in inputs]
        sources = [example["source"] for example in inputs] if "source" in inputs[0] else None
        answer_rewards_local = self._compute_answer_rewards(completions, answers, sources)
        answer_rewards_local = torch.tensor(answer_rewards_local, dtype=torch.float32, device=device)
        answer_rewards = gather(answer_rewards_local)

        confidence_scores_local = self._parse_confidence_scores(completions)
        confidence_scores_local = torch.tensor(confidence_scores_local, dtype=torch.float32, device=device)
        confidence_scores = gather(confidence_scores_local)

        question_accuracy = answer_rewards.view(-1, self.num_generations).mean(dim=1)
        batch_accuracy = question_accuracy.mean()
        difficulty_delta = (question_accuracy - batch_accuracy).repeat_interleave(self.num_generations, dim=0)

        brier_term = -torch.square(confidence_scores - answer_rewards)
        alignment_term = self.dab_lambda * difficulty_delta * (2.0 * confidence_scores - 1.0)
        confidence_rewards = brier_term + alignment_term
        confidence_rewards = torch.where(format_rewards > 0, confidence_rewards, torch.zeros_like(confidence_rewards))

        group_success_rate = question_accuracy.repeat_interleave(self.num_generations, dim=0)
        optimization_rewards_per_func = torch.stack([format_rewards, answer_rewards, confidence_rewards], dim=1)

        reward_weights = self.optimization_rewards
        answer_objective_rewards = (
            reward_weights["format"] * format_rewards
            + reward_weights["accuracy"] * answer_rewards
        )
        confidence_objective_rewards = reward_weights["brier"] * confidence_rewards

        monitoring_rewards_per_func = torch.zeros(len(prompts), len(self.monitoring_reward_names), device=device)
        for i, reward_name in enumerate(self.monitoring_reward_names):
            monitoring_rewards_per_func[:, i] = self.run_reward_function(
                reward_name,
                prompts,
                completions,
                generation_outputs["completion_ids_list"],
                inputs,
            )
        if self.monitoring_reward_names:
            monitoring_rewards_per_func = gather(monitoring_rewards_per_func)

        return {
            "optimization_rewards_per_func": optimization_rewards_per_func,
            "monitoring_rewards_per_func": monitoring_rewards_per_func,
            "answer_rewards": answer_objective_rewards,
            "confidence_rewards": confidence_objective_rewards,
            "confidence_scores": confidence_scores,
            "group_success_rate": group_success_rate,
            "question_accuracy": question_accuracy,
            "batch_accuracy": batch_accuracy,
            "difficulty_delta": difficulty_delta,
        }
