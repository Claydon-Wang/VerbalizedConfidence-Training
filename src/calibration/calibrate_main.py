import argparse
import csv
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

from src.calibration.calibrators import binary_log_loss, build_calibrator
from src.calibration.io_utils import (
    attach_calibrated_confidences,
    extract_confidence_labels,
    load_predictions_jsonl,
    write_json,
    write_jsonl,
)
from src.eval.evaluators.metrics import get_auroc, get_brier, get_ece, plot_reliability_diagram


CALIBRATION_CSV_COLUMNS = [
    "method",
    "fit_path",
    "eval_path",
    "dataset_split",
    "count",
    "metric",
    "before",
    "after",
    "output_dir",
]


def summarize_metrics(labels, probabilities, ece_bins):
    return {
        "count": int(len(probabilities)),
        "accuracy": float(labels.mean()),
        "confidence_avg": float(probabilities.mean()),
        "brier_score": float(get_brier(labels, probabilities)),
        "ece": float(get_ece(labels, probabilities, n_bins=ece_bins)),
        "auroc": float(get_auroc(labels, probabilities)),
        "nll": float(binary_log_loss(labels, probabilities)),
    }


def default_output_dir(eval_path, method):
    eval_dir = os.path.dirname(eval_path)
    return os.path.join(eval_dir, "calibration", method)


def default_calibration_csv_path(eval_path, output_root=None):
    if output_root is not None:
        return os.path.join(output_root, "calibration.csv")
    return os.path.join(os.path.dirname(eval_path), "calibration", "calibration.csv")


def append_calibration_csv(csv_path, summary, output_dir):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    metric_keys = ["accuracy", "confidence_avg", "auroc", "brier_score", "ece", "nll"]
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CALIBRATION_CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        for dataset_split in ("fit", "eval"):
            before_metrics = summary[f"{dataset_split}_before"]
            after_metrics = summary[f"{dataset_split}_after"]
            for metric_key in metric_keys:
                writer.writerow(
                    {
                        "method": summary["method"],
                        "fit_path": summary["fit_path"],
                        "eval_path": summary["eval_path"],
                        "dataset_split": dataset_split,
                        "count": before_metrics["count"],
                        "metric": metric_key,
                        "before": before_metrics[metric_key],
                        "after": after_metrics[metric_key],
                        "output_dir": output_dir,
                    }
                )


def run_one_method(method, fit_path, eval_path, output_dir, ece_bins):
    fit_rows = load_predictions_jsonl(fit_path)
    eval_rows = load_predictions_jsonl(eval_path)
    fit_confidences, fit_labels = extract_confidence_labels(fit_rows)
    eval_confidences, eval_labels = extract_confidence_labels(eval_rows)

    calibrator = build_calibrator(method)
    calibrator.fit(fit_confidences, fit_labels)
    calibrated_fit_confidences = calibrator.transform(fit_confidences)
    calibrated_eval_confidences = calibrator.transform(eval_confidences)
    calibrated_eval_rows = attach_calibrated_confidences(eval_rows, calibrated_eval_confidences)

    os.makedirs(output_dir, exist_ok=True)
    write_jsonl(os.path.join(output_dir, "calibrated_predictions.jsonl"), calibrated_eval_rows)
    write_json(
        os.path.join(output_dir, "calibrator.json"),
        {
            "fit_path": fit_path,
            "eval_path": eval_path,
            "method": method,
            "calibrator": calibrator.to_metadata(),
        },
    )

    summary = {
        "fit_path": fit_path,
        "eval_path": eval_path,
        "method": method,
        "fit_before": summarize_metrics(fit_labels, fit_confidences, ece_bins),
        "fit_after": summarize_metrics(fit_labels, calibrated_fit_confidences, ece_bins),
        "eval_before": summarize_metrics(eval_labels, eval_confidences, ece_bins),
        "eval_after": summarize_metrics(eval_labels, calibrated_eval_confidences, ece_bins),
    }
    write_json(os.path.join(output_dir, "metrics.json"), summary)

    plot_reliability_diagram(
        fit_labels,
        fit_confidences,
        n_bins=ece_bins,
        title=f"{method} fit before",
        save_path=os.path.join(output_dir, "fit_reliability_before.png"),
    ).close()
    plot_reliability_diagram(
        fit_labels,
        calibrated_fit_confidences,
        n_bins=ece_bins,
        title=f"{method} fit after",
        save_path=os.path.join(output_dir, "fit_reliability_after.png"),
    ).close()
    plot_reliability_diagram(
        eval_labels,
        eval_confidences,
        n_bins=ece_bins,
        title=f"{method} eval before",
        save_path=os.path.join(output_dir, "eval_reliability_before.png"),
    ).close()
    plot_reliability_diagram(
        eval_labels,
        calibrated_eval_confidences,
        n_bins=ece_bins,
        title=f"{method} eval after",
        save_path=os.path.join(output_dir, "eval_reliability_after.png"),
    ).close()
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Fit offline confidence calibration models on predictions.jsonl files.")
    parser.add_argument("--fit_path", required=True, help="Path to predictions.jsonl used to fit the calibrator.")
    parser.add_argument(
        "--eval_path",
        default=None,
        help="Path to predictions.jsonl used for evaluation and calibrated export. Defaults to fit_path.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["isotonic_regression"],
        help=(
            "Calibration methods to run. Supported: identity, isotonic_regression, "
            "temperature_scaling, platt_scaling, beta_calibration, histogram_binning."
        ),
    )
    parser.add_argument("--output_root", default=None, help="Optional root directory to save calibration artifacts.")
    parser.add_argument("--ece_bins", type=int, default=10, help="Number of bins for ECE and reliability diagrams.")
    return parser.parse_args()


def main():
    args = parse_args()
    eval_path = args.eval_path or args.fit_path
    summaries = []
    calibration_csv_path = default_calibration_csv_path(eval_path, args.output_root)
    for method in args.methods:
        output_dir = (
            os.path.join(args.output_root, method)
            if args.output_root is not None
            else default_output_dir(eval_path, method)
        )
        summary = run_one_method(method, args.fit_path, eval_path, output_dir, args.ece_bins)
        summaries.append(summary)
        append_calibration_csv(calibration_csv_path, summary, output_dir)
        print(f"[{method}] fit_path={summary['fit_path']}")
        print(f"[{method}] eval_path={summary['eval_path']}")
        print(f"[{method}] fit ECE before={summary['fit_before']['ece']:.4f}, after={summary['fit_after']['ece']:.4f}")
        print(
            f"[{method}] fit Brier before={summary['fit_before']['brier_score']:.4f}, "
            f"after={summary['fit_after']['brier_score']:.4f}"
        )
        print(
            f"[{method}] eval ECE before={summary['eval_before']['ece']:.4f}, "
            f"after={summary['eval_after']['ece']:.4f}"
        )
        print(
            f"[{method}] eval Brier before={summary['eval_before']['brier_score']:.4f}, "
            f"after={summary['eval_after']['brier_score']:.4f}"
        )
        print(f"[{method}] output_dir={output_dir}")

    if len(summaries) > 1:
        comparison_root = args.output_root or os.path.join(os.path.dirname(eval_path), "calibration")
        write_json(os.path.join(comparison_root, "comparison.json"), {"runs": summaries})


if __name__ == "__main__":
    main()
