import csv
import logging
import os
import sys

import datasets
import transformers


logger = logging.getLogger(__name__)


def logger_setup(script_args, training_args, model_args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")


def normalize_report_to(report_to):
    if report_to is None:
        return []
    if isinstance(report_to, str):
        return [] if report_to == "none" else [report_to]
    return list(report_to)


def configure_tracking(training_args, tracking_root):
    os.makedirs(tracking_root, exist_ok=True)

    wandb_root = os.path.join(tracking_root, "wandb")
    swanlab_root = os.path.join(tracking_root, "swanlab")

    os.makedirs(wandb_root, exist_ok=True)
    os.makedirs(swanlab_root, exist_ok=True)

    os.environ["WANDB_DIR"] = wandb_root
    os.environ.setdefault("WANDB_CACHE_DIR", os.path.join(wandb_root, "cache"))
    os.environ["SWANLAB_LOG_DIR"] = swanlab_root
    os.environ["SWANLAB_WORKDIR"] = swanlab_root

    report_to = normalize_report_to(training_args.report_to)

    if "swanlab" in report_to:
        try:
            import swanlab
        except ImportError as exc:
            raise ImportError("`report_to=swanlab` requires the `swanlab` package to be installed.") from exc

        logger.info(
            "Using SwanLab through its wandb compatibility bridge because transformers==4.48.3 does not provide "
            "the native SwanLab integration."
        )
        report_to = [target for target in report_to if target != "swanlab"]
        if "wandb" not in report_to:
            report_to.append("wandb")

        swanlab.sync_wandb(wandb_run=False)
        logger.info("Enabled SwanLab wandb sync bridge.")

    training_args.report_to = report_to


def extract_latest_metrics(log_history, metric_names):
    latest_metrics = {metric_name: "" for metric_name in metric_names}
    remaining = set(metric_names)
    for record in reversed(log_history):
        for metric_name in list(remaining):
            if metric_name in record:
                latest_metrics[metric_name] = record[metric_name]
                remaining.remove(metric_name)
        if not remaining:
            break
    return latest_metrics


def append_train_summary_csv(trainer, script_args, training_args, model_args):
    logs_root = os.path.dirname(os.path.dirname(os.path.dirname(training_args.output_dir)))
    csv_path = os.path.join(logs_root, "train.csv")
    metric_names = [
        "train/reward_total",
        "train/reward_values/format",
        "train/reward_values/accuracy",
        "train/reward_values/brier",
        "eval/reward_total",
        "eval/reward_values/format",
        "eval/reward_values/accuracy",
        "eval/reward_values/brier",
    ]
    metrics = extract_latest_metrics(trainer.state.log_history, metric_names)
    row = {
        "dataset": script_args.dataset_name,
        "model": model_args.model_name_or_path,
        "method": script_args.trainer_name,
        **metrics,
    }
    fieldnames = list(row.keys())

    os.makedirs(logs_root, exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
