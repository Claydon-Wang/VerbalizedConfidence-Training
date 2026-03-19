import os

import numpy as np

from src.eval.evaluators.base_evaluator import BaseEvaluator
from src.eval.evaluators.metrics import compute_pass_n, get_auroc, get_brier, get_ece, plot_reliability_diagram
from src.eval.verifiers import confidence_verifier, llm_confidence_verifier


class ConfidenceEvaluator(BaseEvaluator):
    def run_check_fn(self, dataset_eval):
        if self.config.check_fn is None:
            return None
        if self.config.check_fn == "confidence_verifier":
            return confidence_verifier(dataset_eval, self.config, **self.config.check_fn_args)
        if self.config.check_fn == "llm_confidence_verifier":
            return llm_confidence_verifier(dataset_eval, self.config, **self.config.check_fn_args)
        raise ValueError(f"Unknown check_fn: {self.config.check_fn}")

    def verify_results(self, dataset_eval):
        label_dict = self.run_check_fn(dataset_eval)
        if label_dict is not None:
            dataset_eval = self.merge_output_columns(dataset_eval, label_dict)
        return dataset_eval

    def summarize_results(self, dataset_eval):
        if "is_correct" not in dataset_eval.column_names:
            return {}

        is_correct = dataset_eval["is_correct"]
        generation_len = dataset_eval["generation_len"]
        confidence = dataset_eval["confidence"]
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
        confidence_array = np.array(confidence).flatten()
        metrics["brier_score"] = get_brier(correctness_array, confidence_array)
        metrics["ece"] = get_ece(correctness_array, confidence_array, n_bins=self.config.ece_bins)
        metrics["auroc"] = get_auroc(correctness_array, confidence_array)
        metrics["accuracy"] = metrics["pass@1"]
        metrics["generation_length"] = np.mean(np.array(generation_len))
        metrics["confidence_avg"] = np.mean(np.array(confidence))
        metrics["confidence_legal_ratio"] = np.mean(np.array(is_conf_legal))
        return metrics

    def record_results(self, metrics, dataset_eval=None):
        super().record_results(metrics)
        if (
            not self.config.save_reliability_diagram
            or self.config.log_path is None
            or dataset_eval is None
            or "is_correct" not in dataset_eval.column_names
        ):
            return

        correctness_array = np.array(dataset_eval["is_correct"]).flatten()
        confidence_array = np.array(dataset_eval["confidence"]).flatten()
        save_path = os.path.join(self.config.log_path, "reliability_diagram.png")
        plot_reliability_diagram(
            correctness_array,
            confidence_array,
            n_bins=self.config.ece_bins,
            title=self.config.name,
            save_path=save_path,
        ).close()
