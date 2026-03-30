from src.train.trainers.bar_trainer import BARTrainer
from src.train.trainers.coca_bayesian_trainer import CoCABayesianTrainer
from src.train.trainers.coca_difficulty_trainer import CoCADifficultyTrainer
from src.train.trainers.coca_trainer import CoCATrainer
from src.train.trainers.rlcr_contrastive_trainer import RLCRContrastiveTrainer
from src.train.trainers.rlcr_trainer import RLCRTrainer
from src.train.trainers.rlvr_trainer import RLVRTrainer


TRAINER_REGISTRY = {
    "bar": BARTrainer,
    "coca_bayesian": CoCABayesianTrainer,
    "coca_difficulty": CoCADifficultyTrainer,
    "coca": CoCATrainer,
    "rlcr_contrastive": RLCRContrastiveTrainer,
    "rlcr": RLCRTrainer,
    "rlvr": RLVRTrainer,
}


def build_trainer(trainer_name: str, **trainer_kwargs):
    try:
        trainer_cls = TRAINER_REGISTRY[trainer_name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown trainer '{trainer_name}'. Available trainers: {', '.join(sorted(TRAINER_REGISTRY))}") from exc
    return trainer_cls(**trainer_kwargs)
