from src.eval.evaluators.base_evaluator import BaseEvaluator
from src.eval.evaluators.confidence_evaluator import ConfidenceEvaluator
from src.eval.evaluators.generation_evaluator import GenerationEvaluator


def build_evaluator(config):
    if config.evaluator_name == "confidence":
        return ConfidenceEvaluator(config)
    if config.evaluator_name == "generation":
        return GenerationEvaluator(config)
    if config.evaluator_name == "base":
        return BaseEvaluator(config)
    raise ValueError(f"Unknown evaluator_name: {config.evaluator_name}")
