from src.eval.datasets.base_dataset import BaseDataset


CHOICE_LABELS = ["A", "B", "C", "D", "E", "F"]


class ScienceQA(BaseDataset):
    @staticmethod
    def _has_image(image_value):
        if image_value is None:
            return False
        if isinstance(image_value, str):
            return bool(image_value.strip())
        if isinstance(image_value, (list, tuple, dict, bytes, bytearray)):
            return len(image_value) > 0
        return True

    def reformat(self, dataset):
        if "image" in dataset.column_names:
            dataset = dataset.filter(lambda example: not self._has_image(example.get("image")))

        def mapping(example, idx):
            question = str(example["question"]).strip()
            choices = example["choices"]
            answer_idx = int(example["answer"])

            formatted_choices = []
            for label, choice in zip(CHOICE_LABELS, choices):
                choice_text = str(choice).strip()
                formatted_choices.append(f"({label}) {choice_text}")

            prompt = question
            if formatted_choices:
                prompt += "\n\nChoices:\n" + "\n".join(formatted_choices)
                prompt += "\n\nAnswer with the option letter only."

            answer = CHOICE_LABELS[answer_idx]
            return {
                "id": idx + 1,
                "question": prompt,
                "answer": answer,
            }

        dataset = dataset.map(mapping, with_indices=True)
        keep_columns = [column for column in ["id", "question", "answer"] if column in dataset.column_names]
        return dataset.remove_columns([column for column in dataset.column_names if column not in keep_columns])
