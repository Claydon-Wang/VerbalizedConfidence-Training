import torch
from accelerate.utils import gather

from src.train.trainers.coca_trainer import CoCATrainer


class DCPOTrainer(CoCATrainer):
    trainer_name = "dcpo"

    def validate_reward_specs(self, optimization_rewards, monitoring_rewards):
        super().validate_reward_specs({"format": 1.0, "accuracy": 1.0, "brier": 1.0}, monitoring_rewards)
        optimization_reward_names = set(optimization_rewards)
        required_rewards = {"format", "accuracy", "brier"}
        if optimization_reward_names != required_rewards:
            raise ValueError(
                "DCPOTrainer requires optimization_rewards to be exactly "
                "{'format', 'accuracy', 'brier'}."
            )

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
        answer_rewards_local = torch.as_tensor(answer_rewards_local, dtype=torch.float32, device=device)
        answer_rewards = gather(answer_rewards_local)

        confidence_scores_local = self._parse_confidence_scores(completions)
        confidence_scores_local = torch.as_tensor(confidence_scores_local, dtype=torch.float32, device=device)
        confidence_scores = gather(confidence_scores_local)

        group_success_rate = answer_rewards.view(-1, self.num_generations).mean(dim=1)
        group_success_rate = group_success_rate.repeat_interleave(self.num_generations, dim=0)

        target_confidence = (
            self.args.dcpo_lambda * group_success_rate + (1.0 - self.args.dcpo_lambda) * answer_rewards
        )
        confidence_rewards = -torch.abs(confidence_scores - target_confidence)

        optimization_rewards_per_func = torch.stack([format_rewards, answer_rewards, confidence_rewards], dim=1)

        reward_weights = self.optimization_rewards
        answer_objective_rewards = reward_weights["format"] * format_rewards + reward_weights["accuracy"] * answer_rewards
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
            "target_confidence": target_confidence,
        }
