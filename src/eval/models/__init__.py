from src.eval.models.base_model import BaseModel


def build_model(config):
    return BaseModel(config)
