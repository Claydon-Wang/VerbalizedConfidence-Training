from src.eval.datasets.base_dataset import BaseDataset


class SQuAD(BaseDataset):
    def reformat(self, dataset):
        def mapping(example, idx):
            context = str(example["context"]).strip()
            question = str(example["question"]).strip()
            answers = example.get("answers", {})
            answer_texts = answers.get("text", []) if isinstance(answers, dict) else []
            answer = str(answer_texts[0]).strip() if answer_texts else ""

            prompt = f"Context:\n{context}\n\nQuestion:\n{question}"
            return {
                "id": example.get("id", idx + 1),
                "question": prompt,
                "answer": answer,
            }

        dataset = dataset.map(mapping, with_indices=True)
        keep_columns = [column for column in ["id", "question", "answer"] if column in dataset.column_names]
        return dataset.remove_columns([column for column in dataset.column_names if column not in keep_columns])
