import logging
import re

from src.common.system_prompts import get_sys_prompt


class BaseInferencer:
    def __init__(self, config, model):
        self.config = config
        self.model = model

    def run(self, dataset):
        dataset_eval = dataset.dataset
        sys_messages = self.build_sys_messages(dataset_eval)
        model_inputs, _ = self.model.build_generation_inputs(sys_messages)
        logging.info("Running eval for %s on %d samples", self.config.name, len(dataset_eval))
        generations = self.generate_outputs(model_inputs)
        generations = self.fill_missing_answers(model_inputs, generations)
        generations = self.estimate_confidence(model_inputs, generations)

        output_columns = self.extract_output_columns(generations)
        dataset_eval = self.merge_output_columns(dataset_eval, output_columns)
        return dataset_eval

    def generate_outputs(self, texts):
        return self.model.generate(texts, progress_desc="Generate responses")

    @staticmethod
    def confidence_extractor(confidence_text):
        if confidence_text == "":
            return 0, 0.0
        try:
            confidence = float(confidence_text)
            if 0 <= confidence <= 1:
                return 1, confidence
            if 1 < confidence <= 100:
                return 1, confidence / 100
            return 0, 0.0
        except Exception:
            first_number = re.search(r"-?\d+(?:\.\d+)?", confidence_text)
            if first_number:
                first_number = float(first_number.group())
                if 0 <= first_number <= 1:
                    return 1, first_number
                if 1 < first_number <= 100:
                    return 1, first_number / 100
            return 0, 0.0

    def fill_missing_answers(self, texts, outputs):
        inst = "Thinking time ended \n\n. My final answer is "
        missing_answer_indices = []
        prompts = []
        for output_idx, (text, output) in enumerate(zip(texts, outputs)):
            for sample_idx in range(self.config.num_generations):
                ans_pattern = r"<answer>(.*?)</answer>"
                ans_matches = re.findall(ans_pattern, output.outputs[sample_idx].text, re.DOTALL | re.MULTILINE)
                last_answer = ans_matches[-1] if ans_matches else ""
                if last_answer == "":
                    missing_answer_indices.append((output_idx, sample_idx))
                    prompts.append(text + output.outputs[sample_idx].text + inst)

        ans_calls_needed = 0
        if prompts:
            ans_outputs = self.model.generate(
                prompts,
                n=1,
                temperature=0,
                max_tokens=50,
                logprobs=None,
                progress_desc="Filling missing <answer>",
            )
            for (output_idx, sample_idx), ans_output in zip(missing_answer_indices, ans_outputs):
                answer_text = ans_output.outputs[0].text
                outputs[output_idx].outputs[sample_idx].text = (
                    outputs[output_idx].outputs[sample_idx].text + "<answer> " + answer_text + " </answer>"
                )
                ans_calls_needed += 1

        total_outputs = self.config.num_generations * len(outputs)
        logging.info(
            "Config %s: filled missing <answer> tags for %d/%d outputs",
            self.config.name,
            ans_calls_needed,
            total_outputs,
        )
        return outputs

    def estimate_confidence(self, texts, outputs):
        return outputs

    def extract_output_columns(self, outputs):
        output_columns = {
            "generations": [],
            "answers": [],
            "confidences": [],
            "analyses": [],
        }
        field_patterns = {
            "answer": r"<answer>(.*?)</answer>",
            "confidence": r"<confidence>(.*?)</confidence>",
            "analysis": r"<analysis>(.*?)</analysis>",
        }

        for output in outputs:
            row_generations = []
            row_answers = []
            row_confidences = []
            row_analyses = []
            for generation in output.outputs:
                generation_text = generation.text
                row_generations.append(generation_text)

                answer_matches = re.findall(field_patterns["answer"], generation_text, re.DOTALL | re.MULTILINE)
                confidence_matches = re.findall(field_patterns["confidence"], generation_text, re.DOTALL | re.MULTILINE)
                analysis_matches = re.findall(field_patterns["analysis"], generation_text, re.DOTALL | re.MULTILINE)
                row_answers.append(answer_matches[-1].strip() if answer_matches else "")
                row_confidences.append(confidence_matches[-1].strip() if confidence_matches else "")
                row_analyses.append(analysis_matches[-1].strip() if analysis_matches else "")

            output_columns["generations"].append(row_generations)
            output_columns["answers"].append(row_answers)
            output_columns["confidences"].append(row_confidences)
            output_columns["analyses"].append(row_analyses)

        return output_columns

    def build_sys_messages(self, dataset_eval):
        sys_prompt = get_sys_prompt(self.config.sys_prompt_name)
        sys_messages = []
        for example in dataset_eval:
            sys_messages.append(
                [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"\n\nPROBLEM: {example['question']}\n\n"},
                ]
            )
        return sys_messages

    def merge_output_columns(self, dataset, output_columns):
        for key, value in output_columns.items():
            if key in dataset.column_names:
                dataset = dataset.remove_columns([key])
            dataset = dataset.add_column(key, value)
        return dataset
