from src.eval.models.base_model import BaseModel
from src.eval.models.brpc_model import BRPCModel


def build_model(config):
    if getattr(config, "inferencer_name", None) == "brpc" or getattr(config, "fine_tuned_algorithm", None) == "brpc":
        return BRPCModel(config)
    return BaseModel(config)
