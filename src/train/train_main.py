from src.train.configs.config_schema import GRPOScriptArguments, GRPOConfig, ModelConfig
from src.train.configs.config_utils import build_train_config, split_config_dict
from trl import get_peft_config
from transformers import set_seed
import argparse
import logging
from datasets import load_dataset
import sys
import os
from transformers.trainer_utils import get_last_checkpoint
from src.common.dataset_processing import process_dataset
from src.train.logger import append_train_summary_csv, configure_tracking, logger_setup
from src.train.trainers.trainer_registry import build_trainer
import torch


logger = logging.getLogger(__name__)
TRACKING_ROOT = os.path.abspath("temp/exp_tracking")

def model_init(model_args, training_args):
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    return model_kwargs

def load_config(argv):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--dataset")
    pre_parser.add_argument("--method")
    pre_parser.add_argument("--model")
    known_args, _ = pre_parser.parse_known_args(argv[1:])

    if not all([known_args.dataset, known_args.method, known_args.model]):
        raise ValueError("Training requires --dataset <Class> --method <Class> --model <Class>")

    return split_config_dict(build_train_config(known_args.dataset, known_args.method, known_args.model))

def main(script_args, training_args, model_args):
    set_seed(training_args.seed)
    logger_setup(script_args, training_args, model_args) 
    configure_tracking(training_args, TRACKING_ROOT)

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    dataset = process_dataset(dataset, script_args)  

    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    model_init_kwargs = model_init(model_args, training_args)
    training_args.model_init_kwargs = model_init_kwargs

    if training_args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project

    train_dataset = dataset[script_args.dataset_train_split]
    eval_dataset = dataset[script_args.dataset_test_split]
    if script_args.train_subset_size is not None:
        train_dataset = train_dataset.select(range(script_args.train_subset_size))
    if script_args.eval_subset_size is not None:
        eval_dataset = eval_dataset.select(range(script_args.eval_subset_size))
        
    #############################
    # Initialize the GRPO trainer
    #############################
    trainer = build_trainer(
        script_args.trainer_name,
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.eval_strategy != "no" else None,
    )

     ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = script_args.train_subset_size
    # trainer.log_metrics("train", metrics)
    # trainer.save_metrics("train", metrics)
    try:
        trainer.save_state()
    except:
        print("Failed to save state, please debug")
        pass

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["rl-verify"],
    }
    if trainer.accelerator.is_main_process:
        append_train_summary_csv(trainer, script_args, training_args, model_args)
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)



if __name__ == "__main__":
    script_args, training_args, model_args = load_config(sys.argv)
    main(script_args, training_args, model_args)
