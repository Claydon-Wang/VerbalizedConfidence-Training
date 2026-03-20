from src.eval.datasets.base_dataset import BaseDataset


CHOICE_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


class MMLUPro(BaseDataset):
    def reformat(self, dataset):
        def mapping(example, idx):
            question = str(example["question"]).strip()
            options = example["options"]
            formatted_options = []
            for label, option in zip(CHOICE_LABELS, options):
                option_text = str(option).strip()
                if option_text and option_text != "N/A":
                    formatted_options.append(f"({label}) {option_text}")

            prompt = question
            if formatted_options:
                prompt += "\n\nChoices:\n" + "\n".join(formatted_options)
                prompt += "\n\nAnswer with the option letter only."

            return {
                "id": idx + 1,
                "question": prompt,
                "answer": str(example["answer"]).strip(),
            }

        dataset = dataset.map(mapping, with_indices=True)
        keep_columns = [column for column in ["id", "question", "answer"] if column in dataset.column_names]
        return dataset.remove_columns([column for column in dataset.column_names if column not in keep_columns])
