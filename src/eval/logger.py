import logging
import os
import sys


def setup_eval_logger(log_path: str | None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_path is not None:
        os.makedirs(log_path, exist_ok=True)
        handlers.append(logging.FileHandler(os.path.join(log_path, "log.txt"), mode="w"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
