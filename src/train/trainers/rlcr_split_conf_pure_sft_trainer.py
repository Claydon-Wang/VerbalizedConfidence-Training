import re

import torch
from accelerate.utils import gather, gather_object
from torch.nn.utils.rnn import pad_sequence
from trl.trainer.utils import selective_log_softmax

from src.train.trainers.coca_trainer import CoCATrainer
from src.train.trainers.grpo_trainer import BaseGRPOTrainer
from src.train.trainers.trainer_utils import nanmax, nanmin, nanstd


class RLCRSplitConfPureSFTTrainer(CoCATrainer):
    trainer_name = "rlcr_split_confpuresft"

    def validate_reward_specs(self, optimization_rewards, monitoring_rewards):
        BaseGRPOTrainer.validate_reward_specs(self, optimization_rewards, monitoring_rewards)
        if set(optimization_rewards) != {"format", "accuracy"}:
            raise ValueError(
                "RLCRSplitConfPureSFTTrainer requires optimization_rewards to be exactly {'format', 'accuracy'}."
            )

    def get_reward_names(self, optimization_rewards, monitoring_rewards):
        return ["format", "accuracy"], list(monitoring_rewards)

    def _normalize_answer_rewards(self, rewards: torch.Tensor):
        grouped_rewards = rewards.view(-1, self.num_generations)
        mean_grouped_rewards = grouped_rewards.mean(dim=1)
        std_grouped_rewards = grouped_rewards.std(dim=1)

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        return advantages, mean_grouped_rewards, std_grouped_rewards

    def _format_confidence_target(self, target: float) -> str:
        return f"{max(0.0, min(float(target), 1.0)):.2f}"

    def _replace_or_append_confidence(self, completion_text: str, target: float) -> str:
        target_text = self._format_confidence_target(target)
        pattern = re.compile(r"(<confidence>)(.*?)(</confidence>)", re.DOTALL | re.MULTILINE)
        matches = list(pattern.finditer(completion_text))
        if matches:
            match = matches[-1]
            return completion_text[: match.start(2)] + target_text + completion_text[match.end(2) :]
        return completion_text.rstrip() + f" <confidence>{target_text}</confidence>"

    def _build_confidence_sft_batch(self, prompt_ids, prompt_mask, completions_text, confidence_targets):
        device = prompt_ids.device
        target_texts = [
            self._replace_or_append_confidence(completion_text, target.item())
            for completion_text, target in zip(completions_text, confidence_targets)
        ]
        target_ids = [
            torch.tensor(
                self.processing_class(text, add_special_tokens=False)["input_ids"][: self.max_completion_length],
                dtype=torch.long,
                device=device,
            )
            for text in target_texts
        ]
        if not target_ids:
            empty = torch.empty((0, 0), dtype=torch.long, device=device)
            return empty, empty.float(), empty.float()

        target_completion_ids = pad_sequence(
            target_ids,
            batch_first=True,
            padding_value=self.processing_class.pad_token_id,
        )
        target_completion_mask = (target_completion_ids != self.processing_class.pad_token_id).int()
        confidence_target_mask = self._build_tag_token_mask(
            target_texts,
            target_completion_ids.size(1),
            "confidence",
            device,
        )
        confidence_target_mask = confidence_target_mask * target_completion_mask.float()
        return target_completion_ids, target_completion_mask, confidence_target_mask

    def _get_sft_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep, mode="train"):
        batch_size = self.args.per_device_train_batch_size
        if mode == "eval":
            batch_size = 1

        all_logps = []
        for i in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[i : i + batch_size]
            attention_mask_batch = attention_mask[i : i + batch_size]
            logits = model(
                input_ids=input_ids_batch,
                attention_mask=attention_mask_batch,
                logits_to_keep=logits_to_keep + 1,
            ).logits
            logits = logits[:, :-1, :]
            input_ids_batch = input_ids_batch[:, -logits_to_keep:]
            logits = logits[:, -logits_to_keep:]
            all_logps.append(selective_log_softmax(logits, input_ids_batch))
        return torch.cat(all_logps, dim=0)

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
        raw_answer_rewards = gather(answer_rewards_local)
        answer_rewards = raw_answer_rewards * format_rewards

        confidence_scores_local = self._parse_confidence_scores(completions)
        confidence_scores_local = torch.as_tensor(confidence_scores_local, dtype=torch.float32, device=device)
        confidence_scores = gather(confidence_scores_local)

        group_success_rate = answer_rewards.view(-1, self.num_generations).mean(dim=1)
        group_success_rate = group_success_rate.repeat_interleave(self.num_generations, dim=0)
        target_lambda = float(self.args.conf_pure_sft_lambda)
        confidence_targets = target_lambda * answer_rewards + (1.0 - target_lambda) * group_success_rate

        optimization_rewards_per_func = torch.stack([format_rewards, answer_rewards], dim=1)
        reward_weights = self.optimization_rewards
        answer_objective_rewards = (
            reward_weights["format"] * format_rewards
            + reward_weights["accuracy"] * answer_rewards
        )

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
            "answer_correctness": answer_rewards,
            "format_rewards": format_rewards,
            "confidence_scores": confidence_scores,
            "confidence_targets": confidence_targets,
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

        answer_advantages, answer_group_means, answer_group_stds = self._normalize_answer_rewards(
            reward_outputs["answer_rewards"]
        )

        process_slice = slice(
            self.accelerator.process_index * len(prompt_bundle["prompts"]),
            (self.accelerator.process_index + 1) * len(prompt_bundle["prompts"]),
        )
        answer_advantages = answer_advantages[process_slice]
        confidence_targets = reward_outputs["confidence_targets"][process_slice]

        answer_mask, confidence_mask = self.build_segment_masks(
            generation_outputs["completion_ids"],
            generation_outputs["completion_mask"],
            generation_outputs["completions_text"],
        )
        answer_mask = answer_mask.float()
        confidence_mask = confidence_mask.float()
        format_rewards = reward_outputs["format_rewards"][process_slice]
        target_completion_ids, target_completion_mask, confidence_target_mask = self._build_confidence_sft_batch(
            generation_outputs["prompt_ids"],
            generation_outputs["prompt_mask"],
            generation_outputs["completions_text"],
            confidence_targets,
        )
        confidence_target_mask = confidence_target_mask * format_rewards.unsqueeze(1)

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
        self._metrics[mode]["reward_values/mean_confidence_prediction"].append(
            reward_outputs["confidence_scores"].mean().item()
        )
        self._metrics[mode]["reward_values/mean_confidence_target"].append(
            reward_outputs["confidence_targets"].mean().item()
        )
        self._log_batch_calibration_metrics(mode, generation_outputs, reward_outputs)
        self._metrics[mode]["reward_total"].append(answer_group_means.mean().item())
        self._metrics[mode]["reward_std"].append(answer_group_stds.mean().item())
        self._metrics[mode]["completions/answer_tokens_mean"].append(answer_mask.sum(1).float().mean().item())
        self._metrics[mode]["completions/confidence_sft_tokens_mean"].append(
            confidence_target_mask.sum(1).float().mean().item()
        )
        self._metrics[mode]["completions/confidence_sft_valid_ratio"].append(
            (format_rewards > 0.0).float().mean().item()
        )

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
            "confidence_mask": confidence_mask,
            "answer_advantages": answer_advantages,
            "target_completion_ids": target_completion_ids,
            "target_completion_mask": target_completion_mask,
            "confidence_target_mask": confidence_target_mask,
            "old_per_token_logps": generation_outputs["old_per_token_logps"],
        }

    def _reduce_masked_loss(self, per_token_loss: torch.Tensor, mask: torch.Tensor):
        if self.loss_type == "grpo":
            per_row_loss = (per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            return per_row_loss.mean()
        if self.loss_type == "bnpo":
            return (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
        if self.loss_type == "dr_grpo":
            return (per_token_loss * mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        raise ValueError(f"Unknown loss type: {self.loss_type}")

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
        confidence_mask = inputs["confidence_mask"]

        max_dim = completion_mask.sum(1).max()
        completion_mask = completion_mask[:, :max_dim]
        completion_ids = completion_ids[:, :max_dim]
        answer_mask = answer_mask[:, :max_dim]
        confidence_mask = confidence_mask[:, :max_dim]

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps, per_token_entropies = self._get_per_token_logps(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            return_entropies=True,
        )
        self._log_token_entropy_metrics(
            mode,
            per_token_entropies,
            completion_mask,
            answer_mask,
            confidence_mask,
        )

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
            per_token_logps.detach() if inputs["old_per_token_logps"] is None else inputs["old_per_token_logps"][:, :max_dim]
        )
        answer_advantages = inputs["answer_advantages"]
        answer_per_token_loss, answer_coef_1 = self.compute_policy_loss(
            per_token_logps,
            old_per_token_logps,
            answer_advantages,
            answer_mask,
        )
        if self.beta != 0.0:
            answer_per_token_loss = answer_per_token_loss + self.beta * per_token_kl
        answer_loss = self._reduce_masked_loss(answer_per_token_loss, answer_mask)

        target_completion_ids = inputs["target_completion_ids"]
        target_completion_mask = inputs["target_completion_mask"]
        confidence_target_mask = inputs["confidence_target_mask"]
        target_input_ids = torch.cat([prompt_ids, target_completion_ids], dim=1)
        target_attention_mask = torch.cat([prompt_mask, target_completion_mask], dim=1)
        target_logps = self._get_sft_per_token_logps(
            model,
            target_input_ids,
            target_attention_mask,
            target_completion_ids.size(1),
            mode=mode,
        )
        confidence_sft_loss = self._reduce_masked_loss(-target_logps, confidence_target_mask)
        loss = answer_loss + float(self.args.conf_pure_sft_alpha) * confidence_sft_loss

        self._metrics[mode]["loss/answer_grpo"].append(
            self.accelerator.gather_for_metrics(answer_loss.detach()).nanmean().item()
        )
        self._metrics[mode]["loss/confidence_sft"].append(
            self.accelerator.gather_for_metrics(confidence_sft_loss.detach()).nanmean().item()
        )

        if self.beta != 0.0:
            mean_kl = (per_token_kl * answer_mask).sum() / answer_mask.sum().clamp(min=1.0)
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).nanmean().item())

        answer_low_clipped = (answer_coef_1 < 1 - self.epsilon_low) & (answer_advantages.unsqueeze(1) < 0)
        answer_high_clipped = (answer_coef_1 > 1 + self.epsilon_high) & (answer_advantages.unsqueeze(1) > 0)
        answer_mask_bool = answer_mask.bool()
        is_low_clipped = answer_low_clipped & answer_mask_bool
        is_high_clipped = answer_high_clipped & answer_mask_bool
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = is_low_clipped.sum() / answer_mask.sum().clamp(min=1.0)
        high_clip = is_high_clipped.sum() / answer_mask.sum().clamp(min=1.0)
        clip_ratio = is_region_clipped.sum() / answer_mask.sum().clamp(min=1.0)

        gathered_low_clip = self.accelerator.gather_for_metrics(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather_for_metrics(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather_for_metrics(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())

        return loss
