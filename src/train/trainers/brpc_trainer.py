import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import gather, gather_object
from contextlib import nullcontext

from math_verify import parse, verify

from src.train.rewards.reward_functions import exact_match_score
from src.train.trainers.grpo_trainer import BaseGRPOTrainer
from src.train.trainers.trainer_utils import nanmax, nanmin


class BayesianResidualProbe(nn.Module):
    def __init__(self, hidden_size: int, probe_hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, probe_hidden_size),
            nn.ReLU(),
            nn.Linear(probe_hidden_size, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states).squeeze(-1)


class BRPCTrainer(BaseGRPOTrainer):
    trainer_name = "brpc"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.brpc_probe_loss_weight = float(getattr(self.args, "brpc_probe_loss_weight", 1.0))
        self.brpc_probe_hidden_size = int(getattr(self.args, "brpc_probe_hidden_size", 512))
        self.brpc_eps = float(getattr(self.args, "brpc_eps", 1e-6))
        self.brpc_detach_hidden_states = bool(getattr(self.args, "brpc_detach_hidden_states", True))
        self.brpc_probe_loss_type = str(getattr(self.args, "brpc_probe_loss_type", "bce")).lower()

        hidden_size = self._resolve_model_hidden_size()
        probe = BayesianResidualProbe(hidden_size=hidden_size, probe_hidden_size=self.brpc_probe_hidden_size)
        probe.to(self.accelerator.device)
        self.model.brpc_probe = probe

    @staticmethod
    def _get_probe_module(model):
        probe = getattr(model, "brpc_probe", None)
        if probe is not None:
            return probe
        probe = getattr(getattr(model, "module", None), "brpc_probe", None)
        if probe is not None:
            return probe
        raise AttributeError("BRPC probe module not found on the current model.")

    def _resolve_model_hidden_size(self) -> int:
        config = self.model.config
        for attr in ("hidden_size", "d_model", "n_embd"):
            value = getattr(config, attr, None)
            if value is not None:
                return int(value)
        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            for attr in ("hidden_size", "d_model", "n_embd"):
                value = getattr(text_config, attr, None)
                if value is not None:
                    return int(value)
        raise ValueError("Unable to infer hidden size for BRPC probe.")

    def validate_reward_specs(self, optimization_rewards, monitoring_rewards):
        super().validate_reward_specs(optimization_rewards, monitoring_rewards)
        optimization_reward_names = set(optimization_rewards)
        required_rewards = {"format", "accuracy", "brier"}
        if optimization_reward_names != required_rewards:
            raise ValueError(
                "BRPCTrainer requires optimization_rewards to be exactly {'format', 'accuracy', 'brier'}."
            )

    def _normalize_group_rewards(self, rewards: torch.Tensor):
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards

        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        return advantages, mean_grouped_rewards, std_grouped_rewards

    def _parse_confidence_scores(self, completions):
        confidence_pattern = r"<confidence>(.*?)</confidence>"
        completion_contents = [completion[0]["content"] for completion in completions]
        scores = []
        for content in completion_contents:
            confidence_matches = re.findall(confidence_pattern, content, re.DOTALL | re.MULTILINE)
            last_confidence = confidence_matches[-1] if confidence_matches else ""
            if last_confidence == "":
                scores.append(0.0)
                continue
            try:
                scores.append(max(0.0, min(float(last_confidence), 1.0)))
            except Exception:
                scores.append(0.0)
        return scores

    def _compute_answer_rewards(self, completions, answers, sources=None):
        answer_pattern = r"<answer>(.*?)</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = []

        if sources is None:
            sources = [None] * len(answers)

        for content, gold_answer, source in zip(completion_contents, answers, sources):
            answer_matches = re.findall(answer_pattern, content, re.DOTALL | re.MULTILINE)
            last_answer = answer_matches[-1].strip() if answer_matches else ""
            if last_answer == "":
                matches.append(0.0)
                continue

            try:
                if source == "hotpot":
                    label = exact_match_score(last_answer, gold_answer)
                else:
                    label = verify(gold_answer, parse(last_answer))
                matches.append(float(label))
            except Exception:
                try:
                    matches.append(float(exact_match_score(last_answer, gold_answer)))
                except Exception:
                    matches.append(0.0)

        return matches

    def _confidence_char_span(self, completion_text: str):
        matches = list(re.finditer(r"<confidence>.*?</confidence>", completion_text, re.DOTALL | re.MULTILINE))
        if not matches:
            return None
        last_match = matches[-1]
        return last_match.start(), last_match.end()

    def _feature_char_span(self, completion_text: str):
        confidence_span = self._confidence_char_span(completion_text)
        if confidence_span is None:
            return 0, len(completion_text)
        return 0, confidence_span[0]

    def _build_segment_mask_for_completion(self, completion_ids_row, completion_text: str, span):
        active_len = completion_ids_row.size(0)
        segment_mask = torch.zeros(active_len, dtype=torch.int, device=completion_ids_row.device)
        if span is None:
            return segment_mask

        span_start, span_end = span
        active_ids = completion_ids_row.tolist()
        tokenizer = self.processing_class

        try:
            encoded = tokenizer(
                completion_text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            encoded_ids = encoded["input_ids"]
            offsets = encoded["offset_mapping"]
            non_special_positions = []
            non_special_ids = []
            special_ids = set(tokenizer.all_special_ids)
            for idx, token_id in enumerate(active_ids):
                if token_id in special_ids:
                    continue
                non_special_positions.append(idx)
                non_special_ids.append(token_id)

            if encoded_ids == non_special_ids:
                for token_position, (token_start, token_end) in zip(non_special_positions, offsets):
                    if token_end > span_start and token_start < span_end:
                        segment_mask[token_position] = 1
                return segment_mask
        except Exception:
            pass

        prefix_start = 0
        for idx in range(active_len):
            token_span_end = len(
                tokenizer.decode(
                    completion_ids_row[: idx + 1],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )
            if token_span_end > span_start and prefix_start < span_end:
                segment_mask[idx] = 1
            prefix_start = token_span_end

        return segment_mask

    def build_segment_masks(self, completion_ids: torch.Tensor, completion_mask: torch.Tensor, completions_text: list[str]):
        answer_masks = []
        confidence_masks = []
        feature_masks = []
        for ids_row, mask_row, completion_text in zip(completion_ids, completion_mask, completions_text):
            active_len = int(mask_row.sum().item())
            active_ids = ids_row[:active_len]

            row_confidence_mask = self._build_segment_mask_for_completion(
                active_ids, completion_text, self._confidence_char_span(completion_text)
            )
            row_feature_mask = self._build_segment_mask_for_completion(
                active_ids, completion_text, self._feature_char_span(completion_text)
            )

            padded_confidence_mask = torch.zeros_like(mask_row)
            padded_confidence_mask[:active_len] = row_confidence_mask
            confidence_masks.append(padded_confidence_mask)

            padded_feature_mask = torch.zeros_like(mask_row)
            padded_feature_mask[:active_len] = row_feature_mask
            feature_masks.append(padded_feature_mask)

            answer_masks.append((mask_row - padded_confidence_mask).clamp(min=0))

        answer_mask = torch.stack(answer_masks, dim=0).int()
        confidence_mask = torch.stack(confidence_masks, dim=0).int()
        feature_mask = torch.stack(feature_masks, dim=0).int()
        return answer_mask, confidence_mask, feature_mask

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
        confidence_rewards = -torch.square(confidence_scores - group_success_rate)

        optimization_rewards_per_func = torch.stack([format_rewards, answer_rewards, confidence_rewards], dim=1)

        reward_weights = self.optimization_rewards
        answer_objective_rewards = (
            reward_weights["format"] * format_rewards + reward_weights["accuracy"] * answer_rewards
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
        confidence_advantages, confidence_group_means, confidence_group_stds = self._normalize_group_rewards(
            reward_outputs["confidence_rewards"]
        )

        process_slice = slice(
            self.accelerator.process_index * len(prompt_bundle["prompts"]),
            (self.accelerator.process_index + 1) * len(prompt_bundle["prompts"]),
        )
        answer_advantages = answer_advantages[process_slice]
        confidence_advantages = confidence_advantages[process_slice]
        correctness_targets = optimization_rewards_per_func[:, 1][process_slice]
        confidence_scores = reward_outputs["confidence_scores"][process_slice]
        group_success_rate = reward_outputs["group_success_rate"][process_slice]

        answer_mask, confidence_mask, feature_mask = self.build_segment_masks(
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
        self._metrics[mode]["reward_values/mean_confidence_prediction"].append(
            reward_outputs["confidence_scores"].mean().item()
        )
        self._metrics[mode]["segments/answer_tokens_mean"].append(answer_mask.sum(1).float().mean().item())
        self._metrics[mode]["segments/confidence_tokens_mean"].append(confidence_mask.sum(1).float().mean().item())
        self._metrics[mode]["segments/feature_tokens_mean"].append(feature_mask.sum(1).float().mean().item())

        num_completions_to_log = self.args.num_completions_to_log
        self._textual_logs["step"].extend([str(self.state.global_step)] * num_completions_to_log)
        self._textual_logs["prompt"].extend(gather_object(prompt_bundle["prompts_text"])[0:num_completions_to_log])
        self._textual_logs["completion"].extend(
            gather_object(generation_outputs["completions_text"])[0:num_completions_to_log]
        )
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
            "feature_mask": feature_mask,
            "answer_advantages": answer_advantages,
            "confidence_advantages": confidence_advantages,
            "correctness_targets": correctness_targets,
            "confidence_scores": confidence_scores,
            "group_success_rate": group_success_rate,
            "old_per_token_logps": generation_outputs["old_per_token_logps"],
        }

    def _forward_with_hidden_states(self, model, input_ids, attention_mask, logits_to_keep, mode="train"):
        batch_size = self.args.per_device_train_batch_size
        if mode == "eval":
            batch_size = 1

        all_logps = []
        all_hidden_states = []
        for i in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[i : i + batch_size]
            attention_mask_batch = attention_mask[i : i + batch_size]

            outputs = model(
                input_ids=input_ids_batch,
                attention_mask=attention_mask_batch,
                logits_to_keep=logits_to_keep + 1,
                output_hidden_states=True,
            )
            logits = outputs.logits[:, :-1, :]
            input_ids_target = input_ids_batch[:, -logits_to_keep:]
            logits = logits[:, -logits_to_keep:] / self.temperature
            logps = F.log_softmax(logits, dim=-1).gather(dim=-1, index=input_ids_target.unsqueeze(-1)).squeeze(-1)
            completion_hidden_states = outputs.hidden_states[-1][:, -logits_to_keep:, :]

            all_logps.append(logps)
            all_hidden_states.append(completion_hidden_states)

        return torch.cat(all_logps, dim=0), torch.cat(all_hidden_states, dim=0)

    def _move_model_to_vllm(self):
        gather_if_zero3 = nullcontext
        for name, param in self.model.named_parameters():
            if name.startswith("brpc_probe."):
                continue
            with gather_if_zero3([param]):
                if self.vllm_mode == "server" and self.accelerator.is_main_process:
                    self.vllm_client.update_named_param(name, param.data)
                elif self.vllm_mode == "colocate":
                    llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                    llm_model.load_weights([(name, param.data)])

        if self.vllm_mode == "colocate":
            self.llm.reset_prefix_cache()

    def _compute_probe_outputs(self, model, completion_hidden_states, feature_mask, confidence_scores, correctness_targets):
        pooled_hidden = (
            completion_hidden_states.detach() if self.brpc_detach_hidden_states else completion_hidden_states
        )
        feature_mask = feature_mask.to(dtype=pooled_hidden.dtype)
        pooled_hidden = (pooled_hidden * feature_mask.unsqueeze(-1)).sum(dim=1)
        pooled_hidden = pooled_hidden / feature_mask.sum(dim=1, keepdim=True).clamp(min=1.0)

        gathered_hidden = gather(pooled_hidden)
        gathered_confidence_scores = gather(confidence_scores.float())
        gathered_correctness_targets = gather(correctness_targets.float())

        delta_raw = self._get_probe_module(model)(gathered_hidden)
        grouped_delta_raw = delta_raw.view(-1, self.num_generations)
        group_delta_mean = grouped_delta_raw.mean(dim=1, keepdim=True).repeat(1, self.num_generations).reshape(-1)
        delta = delta_raw - group_delta_mean

        gathered_confidence_scores = gathered_confidence_scores.clamp(min=self.brpc_eps, max=1.0 - self.brpc_eps)
        prior_logits = torch.logit(gathered_confidence_scores)
        combined_logits = prior_logits.detach() + delta

        if self.brpc_probe_loss_type == "mse":
            calibrated_confidence = torch.sigmoid(combined_logits)
            probe_loss = F.mse_loss(calibrated_confidence, gathered_correctness_targets)
        elif self.brpc_probe_loss_type == "bce":
            calibrated_confidence = torch.sigmoid(combined_logits)
            probe_loss = F.binary_cross_entropy_with_logits(combined_logits, gathered_correctness_targets)
        else:
            raise ValueError(
                f"Unsupported brpc_probe_loss_type='{self.brpc_probe_loss_type}'. Supported: bce, mse."
            )

        residual_mse = torch.square(calibrated_confidence - gathered_correctness_targets)
        return {
            "probe_loss": probe_loss,
            "delta": delta,
            "calibrated_confidence": calibrated_confidence,
            "residual_mse": residual_mse,
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
        answer_mask, confidence_mask = inputs["answer_mask"], inputs["confidence_mask"]
        feature_mask = inputs["feature_mask"]

        max_dim = completion_mask.sum(1).max()
        completion_mask = completion_mask[:, :max_dim]
        completion_ids = completion_ids[:, :max_dim]
        answer_mask = answer_mask[:, :max_dim]
        confidence_mask = confidence_mask[:, :max_dim]
        feature_mask = feature_mask[:, :max_dim]

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps, completion_hidden_states = self._forward_with_hidden_states(
            model, input_ids, attention_mask, logits_to_keep, mode=mode
        )

        if self.beta != 0.0:
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, input_ids, attention_mask, logits_to_keep, mode=mode
                    )
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        old_per_token_logps = (
            per_token_logps.detach() if inputs["old_per_token_logps"] is None else inputs["old_per_token_logps"]
        )
        answer_advantages = inputs["answer_advantages"]
        confidence_advantages = inputs["confidence_advantages"]

        answer_per_token_loss, answer_coef_1 = self.compute_policy_loss(
            per_token_logps,
            old_per_token_logps,
            answer_advantages,
            answer_mask,
        )
        confidence_per_token_loss, confidence_coef_1 = self.compute_policy_loss(
            per_token_logps,
            old_per_token_logps,
            confidence_advantages,
            confidence_mask,
        )
        per_token_loss = answer_per_token_loss * answer_mask + confidence_per_token_loss * confidence_mask

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            policy_loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            policy_loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            policy_loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        probe_outputs = self._compute_probe_outputs(
            model=model,
            completion_hidden_states=completion_hidden_states,
            feature_mask=feature_mask,
            confidence_scores=inputs["confidence_scores"],
            correctness_targets=inputs["correctness_targets"],
        )
        loss = policy_loss + self.brpc_probe_loss_weight * probe_outputs["probe_loss"]

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).nanmean().item())

        answer_low_clipped = (answer_coef_1 < 1 - self.epsilon_low) & (answer_advantages.unsqueeze(1) < 0)
        answer_high_clipped = (answer_coef_1 > 1 + self.epsilon_high) & (answer_advantages.unsqueeze(1) > 0)
        confidence_low_clipped = (confidence_coef_1 < 1 - self.epsilon_low) & (confidence_advantages.unsqueeze(1) < 0)
        confidence_high_clipped = (confidence_coef_1 > 1 + self.epsilon_high) & (confidence_advantages.unsqueeze(1) > 0)

        is_low_clipped = answer_low_clipped * answer_mask + confidence_low_clipped * confidence_mask
        is_high_clipped = answer_high_clipped * answer_mask + confidence_high_clipped * confidence_mask
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

        gathered_probe_loss = self.accelerator.gather_for_metrics(probe_outputs["probe_loss"].detach())
        self._metrics[mode]["brpc/probe_loss"].append(gathered_probe_loss.nanmean().item())
        gathered_delta = self.accelerator.gather_for_metrics(probe_outputs["delta"].detach())
        self._metrics[mode]["brpc/delta_mean"].append(gathered_delta.float().mean().item())
        self._metrics[mode]["brpc/delta_abs_mean"].append(gathered_delta.float().abs().mean().item())
        gathered_calibrated = self.accelerator.gather_for_metrics(probe_outputs["calibrated_confidence"].detach())
        self._metrics[mode]["brpc/calibrated_confidence_mean"].append(gathered_calibrated.float().mean().item())
        gathered_residual_mse = self.accelerator.gather_for_metrics(probe_outputs["residual_mse"].detach())
        self._metrics[mode]["brpc/residual_mse_mean"].append(gathered_residual_mse.float().mean().item())
        gathered_group_success = self.accelerator.gather_for_metrics(inputs["group_success_rate"])
        self._metrics[mode]["brpc/group_success_rate_mean"].append(gathered_group_success.float().mean().item())

        return loss
