from math_verify import parse, verify

from src.eval.inferencers.base_inferencer import BaseInferencer
from src.eval.verifiers.utils import exact_match_score


def gen_correctness_reward(completions, answer, **kwargs):
    matches = []
    for completion, expected_answer in zip(completions, answer):
        answer_text = completion[0]["content"]
        attempt = parse(answer_text)
        label = verify(expected_answer, attempt)
        if label == 0:
            label = exact_match_score(answer_text, expected_answer)
        matches.append(float(label))
    return matches


def confidence_verifier(
    local_dataset,
    config,
    format_fn="confidence_format",
    format_pattern="think_answer_analysis_confidence",
    **kwargs,
):
    label_dict = {"is_correct": []}
    is_correct = []
    generation_len = []
    confidence = []
    is_conf_legal = []
    correctness_fn = gen_correctness_reward

    for i in range(len(local_dataset)):
        correctness_list, generation_len_list, confidence_list, conf_legal_list = [], [], [], []
        answers = local_dataset[i]["answers"]
        generations = local_dataset[i]["generations"]
        confidences = local_dataset[i]["confidences"]
        gold_answer = local_dataset[i]["answer"]

        for answer_text, generation_text, confidence_text in zip(answers, generations, confidences):
            args = {"completions": [[{"role": "assistant", "content": answer_text}]], "answer": [gold_answer]}
            actual_correctness = correctness_fn(**args)[0]
            conf_format, conf_level = BaseInferencer.confidence_extractor(confidence_text)
            conf_legal_list.append(conf_format)
            generation_len_list.append(len(generation_text))
            confidence_list.append(conf_level)
            correctness_list.append(1 if actual_correctness == 1 else 0)

        is_correct.append(correctness_list)
        generation_len.append(generation_len_list)
        confidence.append(confidence_list)
        is_conf_legal.append(conf_legal_list)

    label_dict["is_correct"] = is_correct
    label_dict["generation_len"] = generation_len
    label_dict["confidence"] = confidence
    label_dict["is_conf_legal"] = is_conf_legal
    return label_dict
