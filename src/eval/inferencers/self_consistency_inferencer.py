import logging
import re

from tqdm.auto import tqdm

from src.eval.inferencers.base_inferencer import BaseInferencer
from src.eval.verifiers import (
    answers_equivalent,
    build_llm_judge,
    close_llm_judge,
    llm_answers_equivalent_batch,
)


class SelfConsistencyInferencer(BaseInferencer):
    def requires_model_for_confidence_estimation(self) -> bool:
        return False

    def are_answers_equivalent(self, question: str, answer_a: str, answer_b: str, tokenizer=None, llm=None) -> bool:
        answer_verifier_name = getattr(self.config, "answer_verifier_name", None)
        if answer_verifier_name == "llm_verifier":
            return bool(
                llm_answers_equivalent_batch(
                    [question],
                    [(answer_a, answer_b)],
                    self.config,
                    tokenizer=tokenizer,
                    llm=llm,
                )[0]
            )
        return answers_equivalent(answer_a, answer_b)

    def build_equivalence_groups(self, question: str, answers: list[str], tokenizer=None, llm=None) -> list[list[int]]:
        groups = []
        for answer_idx, answer in enumerate(answers):
            if not answer:
                continue

            matched_group_idx = None
            if getattr(self.config, "answer_verifier_name", None) == "llm_verifier":
                candidate_groups = [group for group in groups if group]
                if candidate_groups:
                    comparisons = [(answer, answers[group[0]]) for group in candidate_groups]
                    questions = [question] * len(comparisons)
                    matches = llm_answers_equivalent_batch(
                        questions,
                        comparisons,
                        self.config,
                        tokenizer=tokenizer,
                        llm=llm,
                    )
                    for group, is_match in zip(candidate_groups, matches):
                        if is_match:
                            matched_group_idx = groups.index(group)
                            break
            else:
                for group_idx, group in enumerate(groups):
                    if self.are_answers_equivalent(question, answer, answers[group[0]], tokenizer=tokenizer, llm=llm):
                        matched_group_idx = group_idx
                        break

            if matched_group_idx is None:
                groups.append([answer_idx])
            else:
                groups[matched_group_idx].append(answer_idx)
        return groups

    def estimate_confidence(self, texts, outputs):
        invalid_count = 0
        answer_verifier_name = getattr(self.config, "answer_verifier_name", None)
        tokenizer = llm = None
        if answer_verifier_name == "llm_verifier":
            tokenizer, llm = build_llm_judge(self.config)

        try:
            iterator = zip(texts, outputs)
            for text, output in tqdm(iterator, total=len(outputs), desc="Self-consistency grouping"):
                answers = []
                for generation in output.outputs:
                    ans_matches = re.findall(r"<answer>(.*?)</answer>", generation.text, re.DOTALL | re.MULTILINE)
                    answer = ans_matches[-1].strip() if ans_matches else ""
                    answers.append(answer)

                if isinstance(text, list) and text:
                    question = text[-1]["content"]
                else:
                    question = str(text)
                groups = self.build_equivalence_groups(question, answers, tokenizer=tokenizer, llm=llm)
                group_sizes = {}
                for group in groups:
                    for answer_idx in group:
                        group_sizes[answer_idx] = len(group)

                total_votes = len(answers)
                for answer_idx, generation in enumerate(output.outputs):
                    answer = answers[answer_idx]
                    if not answer or total_votes == 0:
                        confidence = 0.0
                        invalid_count += 1
                    else:
                        confidence = group_sizes.get(answer_idx, 0) / total_votes
                    generation.text = generation.text + f"<confidence> {confidence} </confidence>"
        finally:
            if llm is not None:
                close_llm_judge(llm)

        total_outputs = self.config.num_generations * len(outputs)
        logging.info(
            "Config %s: estimated self-consistency confidence for %d/%d outputs; %d had empty answers",
            self.config.name,
            total_outputs - invalid_count,
            total_outputs,
            invalid_count,
        )
        return outputs
