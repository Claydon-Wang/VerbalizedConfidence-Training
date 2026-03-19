import csv
import os


class BaseEvaluator:
    EVAL_CSV_COLUMNS = [
        "dataset_config_name",
        "dataset_name",
        "model_config_name",
        "model_name_or_path",
        "policy_name",
        "checkpoint_name",
        "pass@1",
        "auroc",
        "brier_score",
        "ece",
        "confidence_avg",
        "confidence_legal_ratio",
        "accuracy",
        "generation_length",
        "log_path",
    ]

    def __init__(self, config):
        self.config = config

    def run(self, dataset_eval):
        dataset_eval = self.verify_results(dataset_eval)
        metrics = self.summarize_results(dataset_eval)
        self.record_results(metrics, dataset_eval=dataset_eval)
        return dataset_eval, metrics

    def merge_output_columns(self, dataset, output_columns):
        for key, value in output_columns.items():
            if key in dataset.column_names:
                dataset = dataset.remove_columns([key])
            dataset = dataset.add_column(key, value)
        return dataset

    def verify_results(self, dataset_eval):
        return dataset_eval

    def summarize_results(self, dataset_eval):
        return {}

    @staticmethod
    def format_metric_value(key, value):
        percent_metric_keys = {
            "brier_score",
            "ece",
            "auroc",
            "accuracy",
            "confidence_avg",
            "confidence_legal_ratio",
        }
        if key.startswith("pass@"):
            return f"{value * 100:.2f}"
        if key in percent_metric_keys:
            return f"{value * 100:.2f}"
        if key == "generation_length":
            return str(int(round(value)))
        return str(value)

    def record_results(self, metrics, dataset_eval=None):
        if self.config.log_path is not None:
            os.makedirs(self.config.log_path, exist_ok=True)
            log_file = os.path.join(self.config.log_path, "log.txt")
            with open(log_file, "a") as f:
                f.write("\n")
                f.write(f"[Evaluation] {self.config.name}\n")
                for key, value in metrics.items():
                    f.write(f"{key}: {self.format_metric_value(key, value)}\n")

        self.record_eval_csv(metrics)

    def record_eval_csv(self, metrics):
        os.makedirs(self.config.logs_root, exist_ok=True)
        csv_path = os.path.join(self.config.logs_root, "eval.csv")
        row = {
            "dataset_config_name": self.config.dataset_config_name,
            "dataset_name": self.config.dataset_name,
            "model_config_name": self.config.model_config_name,
            "model_name_or_path": self.config.model_name_or_path,
            "policy_name": self.config.policy_name,
            "checkpoint_name": self.config.checkpoint_name,
            "pass@1": self.format_metric_value("pass@1", metrics.get("pass@1")) if "pass@1" in metrics else "",
            "auroc": self.format_metric_value("auroc", metrics.get("auroc")) if "auroc" in metrics else "",
            "brier_score": self.format_metric_value("brier_score", metrics.get("brier_score"))
            if "brier_score" in metrics
            else "",
            "ece": self.format_metric_value("ece", metrics.get("ece")) if "ece" in metrics else "",
            "confidence_avg": self.format_metric_value("confidence_avg", metrics.get("confidence_avg"))
            if "confidence_avg" in metrics
            else "",
            "confidence_legal_ratio": self.format_metric_value(
                "confidence_legal_ratio", metrics.get("confidence_legal_ratio")
            )
            if "confidence_legal_ratio" in metrics
            else "",
            "accuracy": self.format_metric_value("accuracy", metrics.get("accuracy")) if "accuracy" in metrics else "",
            "generation_length": self.format_metric_value("generation_length", metrics.get("generation_length"))
            if "generation_length" in metrics
            else "",
            "log_path": self.config.log_path,
        }
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.EVAL_CSV_COLUMNS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
