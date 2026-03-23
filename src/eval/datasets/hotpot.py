from src.eval.datasets.base_dataset import BaseDataset


class HotpotRLCR_Eval(BaseDataset):
    def reformat(self, dataset):
        return self.finalize_dataset(dataset, question_key="problem", answer_key="answer", id_key="id")


class HotpotRLCR_Train(BaseDataset):
    def reformat(self, dataset):
        return self.finalize_dataset(dataset, question_key="problem", answer_key="answer", id_key="id")


class HotpotVanilla(BaseDataset):
    def reformat(self, dataset):
        return self.finalize_dataset(dataset, question_key="problem", answer_key="answer", id_key="id")
