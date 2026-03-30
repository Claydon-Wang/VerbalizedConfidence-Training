from functools import partial

from src.train.rewards.reward_functions import (
    accuracy_reward,
    brier_reward,
    confidence_one_or_zero,
    difficulty_reward,
    format_reward,
    mean_confidence_reward,
    separation_reward,
)


def build_reward_function(reward_name, format_pattern):
    if reward_name == "format":
        return partial(format_reward, format_pattern=format_pattern)
    if reward_name == "accuracy":
        return partial(accuracy_reward, format_pattern=format_pattern)
    if reward_name == "brier":
        return partial(brier_reward, format_pattern=format_pattern)
    if reward_name == "difficulty":
        return partial(difficulty_reward, format_pattern=format_pattern)
    if reward_name == "mean_confidence":
        return mean_confidence_reward
    if reward_name == "confidence_one_or_zero":
        return confidence_one_or_zero
    if reward_name == "separation":
        return separation_reward
    raise ValueError(f"Unknown reward name: {reward_name}")


def build_reward_functions(reward_names, format_pattern):
    return [build_reward_function(reward_name, format_pattern) for reward_name in reward_names]
