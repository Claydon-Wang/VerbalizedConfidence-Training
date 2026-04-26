import os
import re

from datasets import load_dataset, load_from_disk

from src.common.dataset_processing import process_dataset
from src.train.datasets.base_dataset import BaseDataset


FORMATTED_CACHE_DIR = "temp/data/gsm8k_train_formatted"


class GSM8K(BaseDataset):
    def __init__(self, config):
        self.args = config

        if os.path.isdir(FORMATTED_CACHE_DIR) and os.path.exists(os.path.join(FORMATTED_CACHE_DIR, "dataset_dict.json")):
            self.dataset = load_from_disk(FORMATTED_CACHE_DIR)
            return

        dataset = self._load_raw_dataset()
        dataset = self.reformat(dataset)
        dataset = process_dataset(dataset, self.args)
        os.makedirs(os.path.dirname(FORMATTED_CACHE_DIR), exist_ok=True)
        dataset.save_to_disk(FORMATTED_CACHE_DIR)
        self.dataset = dataset

    def _load_raw_dataset(self):
        if self.args.dataset_config is not None:
            return load_dataset(self.args.dataset_name, self.args.dataset_config)
        return load_dataset(self.args.dataset_name)

    def reformat(self, dataset):
        def extract_final_answer(raw_answer):
            if raw_answer is None:
                return ""
            text = str(raw_answer).strip()
            match = re.search(r"####\s*(.*)$", text, re.DOTALL)
            if match:
                return match.group(1).strip()
            return text

        def mapping(example, idx):
            question = example["question"]
            solution = example["answer"]
            return {
                "id": idx + 1,
                "problem": question,
                "question": question,
                "answer": extract_final_answer(solution),
                "solution": solution,
            }

        for split in dataset:
            dataset[split] = dataset[split].map(mapping, with_indices=True)
        return dataset
