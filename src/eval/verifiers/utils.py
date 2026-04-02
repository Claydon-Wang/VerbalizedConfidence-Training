import re
import string


def normalize_answer(text):
    if text is None:
        text = ""
    elif not isinstance(text, str):
        text = str(text)

    def remove_articles(value):
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value):
        return " ".join(value.split())

    def remove_punc(value):
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    def lower(value):
        return value.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)
