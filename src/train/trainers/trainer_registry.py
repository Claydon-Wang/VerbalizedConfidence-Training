from src.train.trainers.coca_trainer import CoCATrainer
from src.train.trainers.dcpo_trainer import DCPOTrainer
from src.train.trainers.rlcr_split_trainer import RLCRSplitTrainer
from src.train.trainers.rlcr_trainer import RLCRTrainer
from src.train.trainers.rlvr_trainer import RLVRTrainer


TRAINER_REGISTRY = {
    "coca": CoCATrainer,
    "dcpo": DCPOTrainer,
    "rlcr_split": RLCRSplitTrainer,
    "rlcr": RLCRTrainer,
    "rlvr": RLVRTrainer,
}


def build_trainer(trainer_name: str, **trainer_kwargs):
    try:
        trainer_cls = TRAINER_REGISTRY[trainer_name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown trainer '{trainer_name}'. Available trainers: {', '.join(sorted(TRAINER_REGISTRY))}") from exc
    return trainer_cls(**trainer_kwargs)
