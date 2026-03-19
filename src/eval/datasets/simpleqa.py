from src.eval.datasets.base_dataset import BaseDataset


class SimpleQA(BaseDataset):
    def reformat(self, dataset):
        question_key = "question" if "question" in dataset.column_names else "problem"
        answer_key = "answer" if "answer" in dataset.column_names else "best_answer"
        id_key = "id" if "id" in dataset.column_names else None
        return self.finalize_dataset(dataset, question_key=question_key, answer_key=answer_key, id_key=id_key)
