from src.eval.datasets.base_dataset import BaseDataset


class BigMathDigits(BaseDataset):
    def reformat(self, dataset):
        return self.finalize_dataset(dataset, question_key="problem", answer_key="answer")


class BigMath_Eval(BigMathDigits):
    pass


class BigMath_Train(BigMathDigits):
    pass
