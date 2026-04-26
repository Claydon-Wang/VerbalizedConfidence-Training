from abc import ABC
import os

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset, load_from_disk

from src.common.dataset_processing import process_dataset


class BaseDataset(ABC):
    def __init__(self, config):
        self.args = config
        dataset = self.load_dataset()
        dataset = self.reformat(dataset)
        self.dataset = process_dataset(dataset, self.args)

    def load_dataset(self):
        dataset_path = self.args.dataset_name
        if os.path.isdir(dataset_path) and os.path.exists(os.path.join(dataset_path, "dataset_dict.json")):
            return load_from_disk(dataset_path)

        load_args = [dataset_path]
        if self.args.dataset_config is not None:
            load_args.append(self.args.dataset_config)
        dataset = load_dataset(*load_args)

        if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
            return dataset
        if isinstance(dataset, (Dataset, IterableDataset)):
            return DatasetDict({self.args.dataset_train_split: dataset})
        raise TypeError(f"Unsupported dataset type: {type(dataset)}")

    def reformat(self, dataset):
        return dataset

    def finalize_split(
        self,
        dataset,
        question_key: str | None = None,
        problem_key: str | None = None,
        answer_key: str = "answer",
        id_key: str | None = None,
        source_value: str | None = None,
    ):
        def mapping(example, idx):
            if problem_key is not None and example.get(problem_key) is not None:
                problem = example[problem_key]
            elif question_key is not None and example.get(question_key) is not None:
                problem = example[question_key]
            else:
                raise KeyError(f"Could not find question field in example keys: {list(example.keys())}")

            output = {"problem": problem, "question": problem, "answer": example[answer_key]}
            if id_key is not None and id_key in example:
                output["id"] = example[id_key]
            elif "id" not in example:
                output["id"] = idx + 1
            if source_value is not None and ("source" not in example or example.get("source") is None):
                output["source"] = source_value
            return output

        return dataset.map(mapping, with_indices=True)

