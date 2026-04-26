from src.train.datasets.base_dataset import BaseDataset


class Hotpot(BaseDataset):
    def reformat(self, dataset):
        for split in dataset:
            question_key = "question" if "question" in dataset[split].column_names else None
            problem_key = "problem" if "problem" in dataset[split].column_names else None
            dataset[split] = self.finalize_split(
                dataset[split],
                question_key=question_key,
                problem_key=problem_key,
                answer_key="answer",
                id_key="id" if "id" in dataset[split].column_names else None,
                source_value="hotpot",
            )
        return dataset

