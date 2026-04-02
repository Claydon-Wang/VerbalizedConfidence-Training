import re
from math_verify import verify,parse
import string
from typing import Optional

def normalize_answer(s):
    if s is None:
        s = ""
    elif not isinstance(s, str):
        s = str(s)

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def _extract_last_tag_value(content, tag_name):
    pattern = rf"<{tag_name}>(.*?)</{tag_name}>"
    matches = re.findall(pattern, content, re.DOTALL | re.MULTILINE)
    return matches[-1] if matches else ""


def extract_confidence_value(content) -> Optional[float]:
    raw_confidence = _extract_last_tag_value(content, "confidence")
    if raw_confidence == "":
        return None
    try:
        confidence = float(raw_confidence)
    except Exception:
        return None
    return max(0.0, min(confidence, 1.0))


def extract_difficulty_value(content) -> Optional[float]:
    raw_difficulty = _extract_last_tag_value(content, "difficulty")
    if raw_difficulty == "":
        return None
    try:
        difficulty = float(raw_difficulty)
    except Exception:
        return None
    return max(0.0, min(difficulty, 1.0))


def format_reward(format_pattern,completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    if format_pattern == "think_analysis_answer_confidence":
        pattern = r".*?</think>\s*<analysis>.*?</analysis>\s*<answer>.*?</answer>\s*<confidence>.*?</confidence>\s*\Z"
    elif format_pattern == "think_answer":
        pattern = r".*?</think>\s*<answer>.*?</answer>\s*\Z"
    elif format_pattern == "think_answer_confidence":
        pattern = r".*?</think>\s*<answer>.*?</answer>\s*<confidence>.*?</confidence>\s*\Z" 
    elif format_pattern == "think_answer_difficulty_confidence":
        pattern = r".*?</think>\s*<answer>.*?</answer>\s*<difficulty>.*?</difficulty>\s*<confidence>.*?</confidence>\s*\Z"
    elif format_pattern == "think_answer_analysis_confidence":
        pattern = r".*?</think>\s*<answer>.*?</answer>\s*<analysis>.*?</analysis>\s*<confidence>.*?</confidence>\s*\Z"
    else:
        raise ValueError(f"Invalid format pattern: {format_pattern}")

    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    matches = [1.0 if match else 0.0 for match in matches]

    # If the format matches, check scalar tags are legal probabilities.
    scalar_extractors = []
    if "difficulty" in format_pattern:
        scalar_extractors.append(extract_difficulty_value)
    if "confidence" in format_pattern:
        scalar_extractors.append(extract_confidence_value)

    for i,match in enumerate(matches):
        if match:
            content = completion_contents[i]
            for extractor in scalar_extractors:
                if extractor(content) is None:
                    matches[i] = 0.0
                    break
    return matches

def accuracy_reward(format_pattern,completions,answer,source=None,**kwargs):
    """Reward function that extracts the last occurrence of text inside the answer tags and then checks if a label is present there"""
    ans_pattern = r"<answer>(.*?)</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    eval_contents = [e for e in answer] 
    matches = []
    format_rewards = format_reward(format_pattern,completions) 
    
    for content,e,fr in zip(completion_contents,eval_contents,format_rewards):
        if fr == 0:
            matches.append(0) 
        else:
            ans_matches = re.findall(ans_pattern, content, re.DOTALL | re.MULTILINE)  # Get all <answer>...</answer> occurrences
            last_answer = ans_matches[-1] if ans_matches else ""  # Get the last answer, if exists
            #if source exists in key and is equal to hotpot, then use the exact match score
            if source is not None and source[0] == 'hotpot':
                label = exact_match_score(last_answer,e)
            else:
                attempt = parse(last_answer)
                label = verify(e,attempt)
            matches.append(float(label))
    return matches

def brier_reward(format_pattern,completions,answer,source=None, **kwargs):
    """Reward function that checks if the completion is correct."""
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = []
    correctness_rewards = accuracy_reward(format_pattern,completions,answer,source) 
    format_rewards = format_reward(format_pattern,completions) 
    for content,cr,fr in zip(completion_contents,correctness_rewards,format_rewards):
        if fr == 0:
            matches.append(0) 
        else:
            conf = extract_confidence_value(content)
            if conf is None:
                matches.append(0)
            else:
                reward = 1 - (cr - conf)**2
                matches.append(reward)
    return matches


def difficulty_reward(format_pattern, completions, answer, source=None, **kwargs):
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = []
    format_rewards = format_reward(format_pattern, completions)
    for content, fr in zip(completion_contents, format_rewards):
        if fr == 0:
            matches.append(0.0)
        else:
            difficulty = extract_difficulty_value(content)
            matches.append(0.0 if difficulty is None else difficulty)
    return matches

def mean_confidence_reward(completions,answer, **kwargs):
    """Reward function that extracts the last occurrence of text inside the answer tags and then checks if a label is present there"""
    completion_contents = [completion[0]["content"] for completion in completions]
    eval_contents = [e for e in answer] 
    matches = []

    for content,e in zip(completion_contents,eval_contents):
        confidence = extract_confidence_value(content)
        if confidence is None:
            matches.append(0.0)
        else:
            matches.append(confidence)
    return matches

def confidence_one_or_zero(completions,answer, **kwargs):
    """Reward function that extracts the last occurrence of text inside the answer tags and then checks if a label is present there"""
    completion_contents = [completion[0]["content"] for completion in completions]
    eval_contents = [e for e in answer] 
    matches = []

    for content,e in zip(completion_contents,eval_contents):
        confidence = extract_confidence_value(content)
        if confidence is None:
            matches.append(0.0)
        else:
            if abs(confidence - 1) < 0.01 or abs(confidence - 0) < 0.01:
                matches.append(1.0)
            else:
                matches.append(0.0)
    return matches


def separation_reward(completions, answer, group_size=None, separation_margin=0.1, source=None, format_pattern=None, **kwargs):
    completion_contents = [completion[0]["content"] for completion in completions]
    if group_size is None or group_size <= 0:
        return [0.0 for _ in completion_contents]

    format_rewards = format_reward(format_pattern, completions) if format_pattern is not None else [1.0] * len(completion_contents)
    correctness_rewards = accuracy_reward(
        format_pattern=format_pattern,
        completions=completions,
        answer=answer,
        source=source,
    )
    confidences = [
        extract_confidence_value(content) if format_ok == 1.0 else None
        for content, format_ok in zip(completion_contents, format_rewards)
    ]
    rewards = [0.0 for _ in completion_contents]

    for group_start in range(0, len(completion_contents), group_size):
        group_end = min(group_start + group_size, len(completion_contents))
        group_correctness = correctness_rewards[group_start:group_end]
        group_confidences = confidences[group_start:group_end]

        pos_confidences = [conf for conf, label in zip(group_confidences, group_correctness) if label == 1.0 and conf is not None]
        neg_confidences = [conf for conf, label in zip(group_confidences, group_correctness) if label == 0.0 and conf is not None]
        pos_mean = sum(pos_confidences) / len(pos_confidences) if pos_confidences else None
        neg_mean = sum(neg_confidences) / len(neg_confidences) if neg_confidences else None

        for offset, (label, conf) in enumerate(zip(group_correctness, group_confidences)):
            if conf is None:
                rewards[group_start + offset] = 0.0
                continue
            if label == 1.0 and neg_mean is not None:
                rewards[group_start + offset] = -max(0.0, separation_margin - (conf - neg_mean))
            elif label == 0.0 and pos_mean is not None:
                rewards[group_start + offset] = -max(0.0, separation_margin - (pos_mean - conf))
            else:
                rewards[group_start + offset] = 0.0

    return rewards
if __name__ == '__main__':
    s = "    h   ello whatever </think> <answer> The number of non-empty subsets 31 </answer> <confidence> 0.9 </confidence>   \n \n  "
 
    pattern = r".*?</think>\s*<answer>.*?</answer>\s*<confidence>.*?</confidence>\s*\Z" 
    match = re.match(pattern, s, re.DOTALL | re.MULTILINE)
    print(match)
    print(match[0])
