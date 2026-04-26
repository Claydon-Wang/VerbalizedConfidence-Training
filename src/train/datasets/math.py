from src.train.datasets.base_dataset import BaseDataset


class Math(BaseDataset):
    def reformat(self, dataset):
        for split in dataset:
            dataset[split] = self.finalize_split(dataset[split], problem_key="problem", answer_key="answer")
        return dataset

