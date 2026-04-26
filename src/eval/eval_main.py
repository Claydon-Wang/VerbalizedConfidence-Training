import gc

from src.eval.configs.config_utils import build_eval_config
from src.eval.datasets import build_dataset
from src.eval.evaluators import build_evaluator
from src.eval.inferencers import build_inferencer
from src.eval.logger import setup_eval_logger
from src.eval.models import build_model


def main(config):
    setup_eval_logger(config.log_path)
    dataset = build_dataset(config)
    model = build_model(config)
    inferencer = build_inferencer(config, model)
    evaluator = build_evaluator(config)
    dataset_eval = inferencer.run(dataset)
    model.close()
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    evaluator.run(dataset_eval)


def cli_main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Eval dataset config class name")
    parser.add_argument("--model", type=str, help="Base model config class name")
    parser.add_argument("--fine_tuned_algorithm", type=str, default=None, help="Optional fine-tuned algorithm class name")
    parser.add_argument(
        "--fine_tuned_dataset",
        type=str,
        default=None,
        help="Optional fine-tuned dataset class name.",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path or HF repo id")
    parser.add_argument("--inferencer", type=str, default=None, help="Eval inferencer name")
    parser.add_argument(
        "--confidence_mode",
        type=str,
        default=None,
        choices=["raw_p"],
        help="How to populate evaluation confidences: raw verbalized probability.",
    )
    args = parser.parse_args()

    if not args.dataset or not args.model:
        raise ValueError(
            "Evaluation requires --dataset <Class> --model <Class> "
            "[--fine_tuned_algorithm <Class>] [--fine_tuned_dataset <Class>] [--checkpoint <path>]"
        )
    config = build_eval_config(
        args.dataset,
        args.model,
        args.fine_tuned_algorithm,
        args.fine_tuned_dataset,
        args.checkpoint,
        args.inferencer,
        args.confidence_mode,
    )
    main(config)


if __name__ == "__main__":
    cli_main()
