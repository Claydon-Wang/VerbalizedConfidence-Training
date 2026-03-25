from src.eval.datasets.base_dataset import BaseDataset


class BoolQ(BaseDataset):
    def reformat(self, dataset):
        def mapping(example, idx):
            passage = str(example["passage"]).strip()
            question = str(example["question"]).strip()
            answer = "yes" if bool(example["answer"]) else "no"

            prompt = f"Passage:\n{passage}\n\nQuestion:\n{question}\n\nAnswer with yes or no."
            return {
                "id": example.get("idx", idx + 1),
                "question": prompt,
                "answer": answer,
            }

        dataset = dataset.map(mapping, with_indices=True)
        keep_columns = [column for column in ["id", "question", "answer"] if column in dataset.column_names]
        return dataset.remove_columns([column for column in dataset.column_names if column not in keep_columns])
