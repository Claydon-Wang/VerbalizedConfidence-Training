from src.eval.inferencers.base_inferencer import BaseInferencer
from src.eval.inferencers.answer_prob_inferencer import AnswerProbInferencer
from src.eval.inferencers.p_true_inferencer import PTrueInferencer
from src.eval.inferencers.self_consistency_inferencer import SelfConsistencyInferencer
from src.eval.inferencers.verbalized_confidence_inferencer import VerbalizedConfidenceInferencer


def build_inferencer(config, model):
    inferencer_name = getattr(config, "inferencer_name", None)
    if inferencer_name == "verbalized_confidence":
        return VerbalizedConfidenceInferencer(config, model)
    if inferencer_name == "answer_prob":
        return AnswerProbInferencer(config, model)
    if inferencer_name == "p_true":
        return PTrueInferencer(config, model)
    if inferencer_name == "self_consistency":
        return SelfConsistencyInferencer(config, model)
    if inferencer_name == "base":
        return BaseInferencer(config, model)
    raise ValueError(f"Unknown inferencer_name: {inferencer_name}")
