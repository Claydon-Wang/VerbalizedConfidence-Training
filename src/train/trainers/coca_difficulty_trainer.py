import re

import torch
from accelerate.utils import gather, gather_object

from src.train.trainers.coca_trainer import CoCATrainer
from src.train.trainers.trainer_utils import nanmax, nanmin


class CoCADifficultyTrainer(CoCATrainer):
    trainer_name = "coca_difficulty"

    def validate_reward_specs(self, optimization_rewards, monitoring_rewards):
        super().validate_reward_specs({"format": 1.0, "accuracy": 1.0, "brier": 1.0}, monitoring_rewards)
        optimization_reward_names = set(optimization_rewards)
        required_rewards = {"format", "accuracy", "difficulty", "brier"}
        if optimization_reward_names != required_rewards:
            raise ValueError(
                "CoCADifficultyTrainer requires optimization_rewards to be exactly "
                "{'format', 'accuracy', 'difficulty', 'brier'}."
            )

    def _parse_scalar_scores(self, completions, tag_name):
        pattern = rf"<{tag_name}>(.*?)</{tag_name}>"
        completion_contents = [completion[0]["content"] for completion in completions]
        scores = []
        for content in completion_contents:
            matches = re.findall(pattern, content, re.DOTALL | re.MULTILINE)
            last_value = matches[-1] if matches else ""
            if last_value == "":
                scores.append(0.0)
                continue
            try:
                scores.append(max(0.0, min(float(last_value), 1.0)))
            except Exception:
                scores.append(0.0)
        return scores

    def _tag_char_span(self, completion_text: str, tag_name: str):
        matches = list(re.finditer(rf"<{tag_name}>.*?</{tag_name}>", completion_text, re.DOTALL | re.MULTILINE))
        if not matches:
            return None
        last_match = matches[-1]
        return last_match.start(), last_match.end()

    def build_segment_masks(self, completion_ids: torch.Tensor, completion_mask: torch.Tensor, completions_text: list[str]):
        difficulty_masks = []
        confidence_masks = []
        for ids_row, mask_row, completion_text in zip(completion_ids, completion_mask, completions_text):
            active_len = int(mask_row.sum().item())
            active_ids = ids_row[:active_len]

            row_difficulty_mask = self._build_segment_mask_for_completion(
                active_ids,
                completion_text,
                self._tag_char_span(completion_text, "difficulty"),
            )
            row_confidence_mask = self._build_segment_mask_for_completion(
                active_ids,
                completion_text,
                self._tag_char_span(completion_text, "confidence"),
            )

            padded_difficulty_mask = torch.zeros_like(mask_row)
            padded_confidence_mask = torch.zeros_like(mask_row)
            padded_difficulty_mask[:active_len] = row_difficulty_mask
            padded_confidence_mask[:active_len] = row_confidence_mask
            difficulty_masks.append(padded_difficulty_mask)
            confidence_masks.append(padded_confidence_mask)

        difficulty_mask = torch.stack(difficulty_masks, dim=0)
        confidence_mask = torch.stack(confidence_masks, dim=0)
        answer_mask = (completion_mask - difficulty_mask - confidence_mask).clamp(min=0)
        return answer_mask.int(), difficulty_mask.int(), confidence_mask.int()

    def _compute_mad_group_advantages(self, rewards: torch.Tensor):
        grouped_rewards = rewards.view(-1, self.num_generations)
        mean_grouped_rewards = grouped_rewards.mean(dim=1)
        mad_grouped_rewards = (grouped_rewards - mean_grouped_rewards.unsqueeze(1)).abs().mean(dim=1)

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        mad_grouped_rewards = mad_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (mad_grouped_rewards + 1e-4)
        return advantages, mean_grouped_rewards, mad_grouped_rewards

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

        difficulty_scores_local = torch.tensor(
            self._parse_scalar_scores(completions, "difficulty"),
            dtype=torch.float32,
            device=device,
        )
        difficulty_scores = gather(difficulty_scores_local)

        confidence_scores_local = torch.tensor(
            self._parse_scalar_scores(completions, "confidence"),
            dtype=torch.float32,
            device=device,
        )
        confidence_scores = gather(confidence_scores_local)

        group_success_rate = answer_rewards.view(-1, self.num_generations).mean(dim=1)
        group_success_rate = group_success_rate.repeat_interleave(self.num_generations, dim=0)
        difficulty_targets = 1.0 - group_success_rate
        difficulty_rewards = -torch.square(difficulty_scores - difficulty_targets)
        confidence_rewards = -torch.square(confidence_scores - answer_rewards)

        optimization_rewards_per_func = torch.stack(
            [format_rewards, answer_rewards, difficulty_rewards, confidence_rewards],
            dim=1,
        )

        reward_weights = self.optimization_rewards
        answer_objective_rewards = (
            reward_weights["format"] * format_rewards
            + reward_weights["accuracy"] * answer_rewards
        )
        difficulty_objective_rewards = reward_weights["difficulty"] * difficulty_rewards
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
            "difficulty_rewards": difficulty_objective_rewards,
            "confidence_rewards": confidence_objective_rewards,
            "difficulty_scores": difficulty_scores,
            "difficulty_targets": difficulty_targets,
            "confidence_scores": confidence_scores,
            "group_success_rate": group_success_rate,
        }

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
        difficulty_advantages, difficulty_group_means, difficulty_group_stds = self._normalize_group_rewards(
            reward_outputs["difficulty_rewards"]
        )
        confidence_advantages, confidence_group_means, confidence_group_mads = self._compute_mad_group_advantages(
            reward_outputs["confidence_rewards"]
        )

        process_slice = slice(
            self.accelerator.process_index * len(prompt_bundle["prompts"]),
            (self.accelerator.process_index + 1) * len(prompt_bundle["prompts"]),
        )
        answer_advantages = answer_advantages[process_slice]
        difficulty_advantages = difficulty_advantages[process_slice]
        confidence_advantages = confidence_advantages[process_slice]

        answer_mask, difficulty_mask, confidence_mask = self.build_segment_masks(
            generation_outputs["completion_ids"],
            generation_outputs["completion_mask"],
            generation_outputs["completions_text"],
        )

        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather_for_metrics(
                generation_outputs["attention_mask"].sum()
            ).sum().item()
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
        self._log_reward_metric_summaries(mode, reward_metric_names, reward_metric_values)
        self._metrics[mode]["reward_values/group_success_rate"].append(
            reward_outputs["group_success_rate"].mean().item()
        )
        self._metrics[mode]["reward_values/mean_difficulty_target"].append(
            reward_outputs["difficulty_targets"].mean().item()
        )
        self._metrics[mode]["reward_values/mean_difficulty_prediction"].append(
            reward_outputs["difficulty_scores"].mean().item()
        )
        self._metrics[mode]["reward_values/mean_confidence_prediction"].append(
            reward_outputs["confidence_scores"].mean().item()
        )
        self._metrics[mode]["segments/answer_tokens_mean"].append(answer_mask.sum(1).float().mean().item())
        self._metrics[mode]["segments/difficulty_tokens_mean"].append(difficulty_mask.sum(1).float().mean().item())
        self._metrics[mode]["segments/confidence_tokens_mean"].append(confidence_mask.sum(1).float().mean().item())

        num_completions_to_log = self.args.num_completions_to_log
        self._textual_logs["step"].extend([str(self.state.global_step)] * num_completions_to_log)
        self._textual_logs["prompt"].extend(gather_object(prompt_bundle["prompts_text"])[0:num_completions_to_log])
        self._textual_logs["completion"].extend(gather_object(generation_outputs["completions_text"])[0:num_completions_to_log])
        for i, name in enumerate(reward_metric_names):
            self._textual_logs["rewards"][name].extend(reward_metric_values[:, i].tolist()[0:num_completions_to_log])

        self._record_batch_confidences(mode, generation_outputs, reward_outputs, inputs)

        return {
            "prompt_ids": generation_outputs["prompt_ids"],
            "prompt_mask": generation_outputs["prompt_mask"],
            "completion_ids": generation_outputs["completion_ids"],
            "completion_mask": generation_outputs["completion_mask"],
            "answer_mask": answer_mask,
            "difficulty_mask": difficulty_mask,
            "confidence_mask": confidence_mask,
            "answer_advantages": answer_advantages,
            "difficulty_advantages": difficulty_advantages,
            "confidence_advantages": confidence_advantages,
            "old_per_token_logps": generation_outputs["old_per_token_logps"],
        }

    def update_policy(self, model, inputs):
        mode = "train" if self.model.training else "eval"
        if mode == "train":
            torch.cuda.empty_cache()
            if not self.vllm_sleeping:
                self.llm.sleep()
                self.accelerator.wait_for_everyone()

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        answer_mask = inputs["answer_mask"]
        difficulty_mask = inputs["difficulty_mask"]
        confidence_mask = inputs["confidence_mask"]

        max_dim = completion_mask.sum(1).max()
        completion_mask = completion_mask[:, :max_dim]
        completion_ids = completion_ids[:, :max_dim]
        answer_mask = answer_mask[:, :max_dim]
        difficulty_mask = difficulty_mask[:, :max_dim]
        confidence_mask = confidence_mask[:, :max_dim]

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        if self.beta != 0.0:
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, input_ids, attention_mask, logits_to_keep
                    )
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        old_per_token_logps = (
            per_token_logps.detach() if inputs["old_per_token_logps"] is None else inputs["old_per_token_logps"]
        )
        answer_advantages = inputs["answer_advantages"]
        difficulty_advantages = inputs["difficulty_advantages"]
        confidence_advantages = inputs["confidence_advantages"]

        answer_per_token_loss, answer_coef_1 = self.compute_policy_loss(
            per_token_logps,
            old_per_token_logps,
            answer_advantages,
            answer_mask,
        )
        difficulty_per_token_loss, difficulty_coef_1 = self.compute_policy_loss(
            per_token_logps,
            old_per_token_logps,
            difficulty_advantages,
            difficulty_mask,
        )
        confidence_per_token_loss, confidence_coef_1 = self.compute_policy_loss(
            per_token_logps,
            old_per_token_logps,
            confidence_advantages,
            confidence_mask,
        )
        per_token_loss = (
            answer_per_token_loss * answer_mask
            + difficulty_per_token_loss * difficulty_mask
            + confidence_per_token_loss * confidence_mask
        )

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).nanmean().item())

        answer_low_clipped = (answer_coef_1 < 1 - self.epsilon_low) & (answer_advantages.unsqueeze(1) < 0)
        answer_high_clipped = (answer_coef_1 > 1 + self.epsilon_high) & (answer_advantages.unsqueeze(1) > 0)
        difficulty_low_clipped = (difficulty_coef_1 < 1 - self.epsilon_low) & (difficulty_advantages.unsqueeze(1) < 0)
        difficulty_high_clipped = (difficulty_coef_1 > 1 + self.epsilon_high) & (difficulty_advantages.unsqueeze(1) > 0)
        confidence_low_clipped = (confidence_coef_1 < 1 - self.epsilon_low) & (confidence_advantages.unsqueeze(1) < 0)
        confidence_high_clipped = (confidence_coef_1 > 1 + self.epsilon_high) & (confidence_advantages.unsqueeze(1) > 0)

        is_low_clipped = (
            answer_low_clipped * answer_mask
            + difficulty_low_clipped * difficulty_mask
            + confidence_low_clipped * confidence_mask
        )
        is_high_clipped = (
            answer_high_clipped * answer_mask
            + difficulty_high_clipped * difficulty_mask
            + confidence_high_clipped * confidence_mask
        )
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = is_low_clipped.sum() / completion_mask.sum()
        high_clip = is_high_clipped.sum() / completion_mask.sum()
        clip_ratio = is_region_clipped.sum() / completion_mask.sum()

        gathered_low_clip = self.accelerator.gather_for_metrics(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather_for_metrics(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather_for_metrics(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())

        return loss


class CoCADATrainer(CoCADifficultyTrainer):
    trainer_name = "coca_da"
