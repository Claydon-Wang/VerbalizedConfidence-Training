from math_verify import parse, verify

from src.eval.verifiers.utils import exact_match_score


def answers_equivalent(answer_a: str, answer_b: str) -> bool:
    if not answer_a or not answer_b:
        return False

    try:
        label = verify(answer_a, parse(answer_b))
        if label == 1:
            return True
    except Exception:
        pass

    try:
        label = verify(answer_b, parse(answer_a))
        if label == 1:
            return True
    except Exception:
        pass

    return exact_match_score(answer_a, answer_b)


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


def rule_verifier(
    local_dataset,
    config,
    format_fn="confidence_format",
    format_pattern="think_answer_analysis_confidence",
    **kwargs,
):
    label_dict = {"is_correct": []}
    is_correct = []
    correctness_fn = gen_correctness_reward

    for i in range(len(local_dataset)):
        correctness_list = []
        predictions = local_dataset[i]["predictions"]
        gold_answer = local_dataset[i]["answer"]

        for answer_text in predictions:
            args = {"completions": [[{"role": "assistant", "content": answer_text}]], "answer": [gold_answer]}
            actual_correctness = correctness_fn(**args)[0]
            correctness_list.append(1 if actual_correctness == 1 else 0)

        is_correct.append(correctness_list)

    label_dict["is_correct"] = is_correct
    return label_dict
