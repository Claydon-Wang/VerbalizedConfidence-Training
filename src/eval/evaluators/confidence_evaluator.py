import os

import numpy as np

from src.eval.evaluators.base_evaluator import BaseEvaluator
from src.eval.evaluators.metrics import compute_pass_n, get_auroc, get_brier, get_ece, plot_reliability_diagram
from src.eval.verifiers import llm_verifier, rule_verifier


class ConfidenceEvaluator(BaseEvaluator):
    def run_answer_verifier(self, dataset_eval):
        if self.config.answer_verifier_name is None:
            return None
        if self.config.answer_verifier_name == "rule_verifier":
            return rule_verifier(dataset_eval, self.config, **self.config.answer_verifier_args)
        if self.config.answer_verifier_name == "llm_verifier":
            return llm_verifier(dataset_eval, self.config, **self.config.answer_verifier_args)
        raise ValueError(f"Unknown answer_verifier_name: {self.config.answer_verifier_name}")

    def verify_results(self, dataset_eval):
        label_dict = self.run_answer_verifier(dataset_eval)
        if label_dict is not None:
            dataset_eval = self.merge_output_columns(dataset_eval, label_dict)
        if "generation_len" not in dataset_eval.column_names:
            generation_len = [[len(generation_text) for generation_text in generations] for generations in dataset_eval["generations"]]
            dataset_eval = self.merge_output_columns(dataset_eval, {"generation_len": generation_len})
        return dataset_eval

    def summarize_results(self, dataset_eval):
        if "is_correct" not in dataset_eval.column_names:
            return {}

        is_correct = dataset_eval["is_correct"]
        generation_len = dataset_eval["generation_len"]
        confidences = dataset_eval["confidences"]
        is_conf_legal = dataset_eval["is_conf_legal"]

        metrics = {}
        n = self.config.num_generations
        if n not in self.config.pass_k_vals:
            self.config.pass_k_vals.append(n)
        if 1 not in self.config.pass_k_vals:
            self.config.pass_k_vals.append(1)
        for k in self.config.pass_k_vals:
            if k <= n:
                metrics[f"pass@{k}"] = compute_pass_n(is_correct, k)

        correctness_array = np.array(is_correct).flatten()
        confidence_array = np.array(confidences).flatten()
        metrics["brier_score"] = get_brier(correctness_array, confidence_array)
        metrics["ece"] = get_ece(correctness_array, confidence_array, n_bins=self.config.ece_bins)
        metrics["auroc"] = get_auroc(correctness_array, confidence_array)
        metrics["accuracy"] = metrics["pass@1"]
        metrics["generation_length"] = np.mean(np.array(generation_len))
        metrics["confidence_avg"] = np.mean(np.array(confidences))
        metrics["confidence_legal_ratio"] = np.mean(np.array(is_conf_legal))
        return metrics

    def save_results(self, metrics, dataset_eval=None):
        super().save_results(metrics, dataset_eval=dataset_eval)
        if (
            not self.config.save_reliability_diagram
            or self.config.log_path is None
            or dataset_eval is None
            or "is_correct" not in dataset_eval.column_names
        ):
            return

        correctness_array = np.array(dataset_eval["is_correct"]).flatten()
        confidence_array = np.array(dataset_eval["confidences"]).flatten()
        save_path = os.path.join(self.config.log_path, "reliability_diagram.png")
        plot_reliability_diagram(
            correctness_array,
            confidence_array,
            n_bins=self.config.ece_bins,
            title=self.config.name,
            save_path=save_path,
        ).close()
