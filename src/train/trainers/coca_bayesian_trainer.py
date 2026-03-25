import json
import os

import torch
from accelerate.utils import gather, gather_object

from src.train.trainers.coca_trainer import CoCATrainer


TRUNCATION_SUFFIX = "[Truncated for storage]"


class CoCABayesianTrainer(CoCATrainer):
    trainer_name = "coca_bayesian"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bayesian_alpha_ratio = float(getattr(self.args, "bayesian_alpha_ratio", 1.0))
        self.bayesian_prior_path = getattr(self.args, "bayesian_prior_path", None)
        self.bayesian_eval_prior_path = getattr(self.args, "bayesian_eval_prior_path", None) or self.bayesian_prior_path
        self.bayesian_prior_field = getattr(self.args, "bayesian_prior_field", "calibrated_confidences")
        self.bayesian_prior_reduce = getattr(self.args, "bayesian_prior_reduce", "mean")
        self.bayesian_require_question_match = bool(getattr(self.args, "bayesian_require_question_match", True))
        self.train_prior_state = self._load_prior_state(self.bayesian_prior_path)
        self.eval_prior_state = (
            self.train_prior_state
            if self.bayesian_eval_prior_path == self.bayesian_prior_path
            else self._load_prior_state(self.bayesian_eval_prior_path)
        )
        self._validate_prior_alignment(self.train_dataset, self.train_prior_state, "train")
        if self.eval_dataset is not None:
            self._validate_prior_alignment(self.eval_dataset, self.eval_prior_state, "eval")

    def _reduce_prior_values(self, values):
        if not values:
            raise ValueError("Prior confidence list is empty.")
        if self.bayesian_prior_reduce == "mean":
            return float(sum(values) / len(values))
        if self.bayesian_prior_reduce == "first":
            return float(values[0])
        raise ValueError(
            f"Unsupported bayesian_prior_reduce='{self.bayesian_prior_reduce}'. Supported: mean, first."
        )

    @staticmethod
    def _normalize_question_for_match(question):
        if question is None:
            return None
        question = str(question)
        if TRUNCATION_SUFFIX in question:
            question = question.split(TRUNCATION_SUFFIX, 1)[0].rstrip()
        return question

    def _questions_match(self, train_question, prior_question):
        if prior_question is None:
            return True
        train_normalized = self._normalize_question_for_match(train_question)
        prior_normalized = self._normalize_question_for_match(prior_question)
        if train_normalized == prior_normalized:
            return True
        return bool(prior_normalized and train_normalized and train_normalized.startswith(prior_normalized))

    @staticmethod
    def _get_example_question(example):
        if example is None:
            return None
        if "question" in example and example.get("question") is not None:
            return example.get("question")
        if "problem" in example and example.get("problem") is not None:
            return example.get("problem")
        return None

    def _active_prior_state(self):
        return self.train_prior_state if self.model.training else self.eval_prior_state

    def _load_prior_state(self, prior_path):
        if not prior_path:
            raise ValueError("CoCABayesianTrainer requires args.bayesian_prior_path to be set.")
        if not os.path.isfile(prior_path):
            raise FileNotFoundError(f"Bayesian prior file not found: {prior_path}")

        prior_state = {}
        with open(prior_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if "id" not in row:
                    raise ValueError(f"Missing 'id' in prior file at line {line_no}.")
                example_id = row["id"]
                if example_id in prior_state:
                    raise ValueError(f"Duplicate id {example_id!r} in prior file: {prior_path}")
                prior_values = row.get(self.bayesian_prior_field)
                if prior_values is None:
                    raise ValueError(
                        f"Missing prior field '{self.bayesian_prior_field}' for id={example_id!r} in {prior_path}"
                    )
                if not isinstance(prior_values, list):
                    prior_values = [prior_values]
                prior_state[example_id] = {
                    "prior": self._reduce_prior_values([float(value) for value in prior_values]),
                    "question": row.get("question"),
                }
        return prior_state

    def _validate_prior_alignment(self, dataset, prior_state, split_name):
        seen_ids = set()
        missing_ids = []
        mismatched_questions = []
        for example in dataset:
            example_id = example.get("id")
            if example_id is None:
                raise ValueError("Training examples must contain an 'id' field for CoCABayesianTrainer.")
            if example_id in seen_ids:
                raise ValueError(f"Duplicate training example id detected: {example_id!r}")
            seen_ids.add(example_id)
            prior_entry = prior_state.get(example_id)
            if prior_entry is None:
                missing_ids.append(example_id)
                continue
            if self.bayesian_require_question_match and not self._questions_match(
                self._get_example_question(example), prior_entry.get("question")
            ):
                mismatched_questions.append(example_id)

        if missing_ids:
            preview = ", ".join(map(str, missing_ids[:5]))
            raise ValueError(
                f"Bayesian prior file is missing {len(missing_ids)} {split_name} ids. First missing ids: {preview}"
            )
        if mismatched_questions:
            preview = ", ".join(map(str, mismatched_questions[:5]))
            raise ValueError(
                f"Bayesian prior question mismatch for {len(mismatched_questions)} {split_name} ids. First mismatched ids: {preview}"
            )

    def _lookup_prior_values(self, inputs):
        prior_state = self._active_prior_state()
        priors = []
        for example in inputs:
            example_id = example.get("id")
            if example_id not in prior_state:
                raise KeyError(f"Missing Bayesian prior for example id={example_id!r}")
            prior_entry = prior_state[example_id]
            if self.bayesian_require_question_match and not self._questions_match(
                self._get_example_question(example), prior_entry.get("question")
            ):
                raise ValueError(f"Question mismatch for example id={example_id!r} during prior lookup.")
            priors.append(prior_entry["prior"])
        return priors

    def _update_prior_state(self, inputs, posterior_targets_local):
        grouped_ids = []
        grouped_questions = []
        grouped_posteriors = []
        for start in range(0, len(inputs), self.num_generations):
            grouped_inputs = inputs[start : start + self.num_generations]
            if not grouped_inputs:
                continue
            grouped_ids.append(grouped_inputs[0]["id"])
            grouped_questions.append(self._get_example_question(grouped_inputs[0]))
            grouped_posteriors.append(float(posterior_targets_local[start].item()))

        gathered_ids = gather_object(grouped_ids)
        gathered_questions = gather_object(grouped_questions)
        gathered_posteriors = gather_object(grouped_posteriors)

        for example_id, question, posterior in zip(gathered_ids, gathered_questions, gathered_posteriors):
            if example_id not in self.train_prior_state:
                raise KeyError(f"Cannot update Bayesian prior for unknown id={example_id!r}")
            if self.bayesian_require_question_match and not self._questions_match(
                question, self.train_prior_state[example_id].get("question")
            ):
                raise ValueError(f"Question mismatch while updating Bayesian prior for id={example_id!r}")
            self.train_prior_state[example_id]["prior"] = float(posterior)

    def compute_rewards(self, generation_outputs, inputs):
        reward_outputs = super().compute_rewards(generation_outputs, inputs)
        device = self.accelerator.device

        prior_scores_local = torch.tensor(self._lookup_prior_values(inputs), dtype=torch.float32, device=device)
        prior_scores = gather(prior_scores_local)

        answer_rewards = reward_outputs["optimization_rewards_per_func"][:, 1]
        correct_counts = answer_rewards.view(-1, self.num_generations).sum(dim=1)
        prior_per_group = prior_scores.view(-1, self.num_generations)[:, 0]
        prior_strength = self.bayesian_alpha_ratio * self.num_generations
        posterior_per_group = (prior_strength * prior_per_group + correct_counts) / (
            prior_strength + self.num_generations
        )
        posterior_targets = posterior_per_group.repeat_interleave(self.num_generations, dim=0)

        confidence_scores = reward_outputs["confidence_scores"]
        confidence_rewards = -torch.square(confidence_scores - posterior_targets)
        reward_outputs["optimization_rewards_per_func"][:, 2] = confidence_rewards
        reward_outputs["confidence_rewards"] = self.optimization_rewards["brier"] * confidence_rewards
        reward_outputs["prior_confidence"] = prior_scores
        reward_outputs["posterior_target"] = posterior_targets

        if self.model.training:
            process_slice = slice(
                self.accelerator.process_index * len(inputs),
                (self.accelerator.process_index + 1) * len(inputs),
            )
            self._update_prior_state(inputs, posterior_targets[process_slice])

        return reward_outputs
