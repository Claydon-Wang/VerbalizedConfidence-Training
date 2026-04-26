from src.train.trainers.bar_trainer import BARTrainer
from src.train.trainers.brpc_trainer import BRPCTrainer
from src.train.trainers.coca_bayesian_trainer import CoCABayesianTrainer
from src.train.trainers.coca_difficulty_trainer import CoCADATrainer, CoCADifficultyTrainer
from src.train.trainers.coca_trainer import CoCATrainer
from src.train.trainers.rlcr_split_cal_trainer import RLCRSplitCalTrainer
from src.train.trainers.rlcr_split_batch_trainer import RLCRSplitBatchTrainer
from src.train.trainers.rlcr_split_dab_trainer import RLCRSplitDABTrainer
from src.train.trainers.rlcr_split_global_debias_trainer import RLCRSplitGlobalDebiasTrainer
from src.train.trainers.rlcr_split_global_debias_nostd_trainer import RLCRSplitGlobalDebiasNoStdTrainer
from src.train.trainers.rlcr_split_global_trainer import RLCRSplitGlobalTrainer
from src.train.trainers.rlcr_split_global_rw_trainer import RLCRSplitGlobalRWTrainer
from src.train.trainers.rlcr_split_global_rw_noreweight_trainer import RLCRSplitGlobalRWNoReweightTrainer
from src.train.trainers.rlcr_split_random_target_trainer import RLCRSplitRandomTargetTrainer
from src.train.trainers.rlcr_split_trainer import RLCRSplitTrainer
from src.train.trainers.rlcr_split_da_trainer import RLCRSplitDATrainer
from src.train.trainers.rlcr_split_nostd_trainer import RLCRSplitNoStdTrainer
from src.train.trainers.rlcr_split_var_square_trainer import RLCRSplitVarSquareTrainer
from src.train.trainers.rlcr_contrastive_trainer import RLCRContrastiveTrainer
from src.train.trainers.rlcr_trainer import RLCRTrainer
from src.train.trainers.rlvr_trainer import RLVRTrainer


TRAINER_REGISTRY = {
    "bar": BARTrainer,
    "brpc": BRPCTrainer,
    "coca_bayesian": CoCABayesianTrainer,
    "coca_da": CoCADATrainer,
    "coca_difficulty": CoCADifficultyTrainer,
    "coca": CoCATrainer,
    "rlcr_contrastive": RLCRContrastiveTrainer,
    "rlcr_split_cal": RLCRSplitCalTrainer,
    "rlcr_split_batch": RLCRSplitBatchTrainer,
    "rlcr_split_dab": RLCRSplitDABTrainer,
    "rlcr_split_global_debias": RLCRSplitGlobalDebiasTrainer,
    "rlcr_split_global_debias_nostd": RLCRSplitGlobalDebiasNoStdTrainer,
    "rlcr_split_global": RLCRSplitGlobalTrainer,
    "rlcr_split_global_rw": RLCRSplitGlobalRWTrainer,
    "rlcr_split_global_rw_noreweight": RLCRSplitGlobalRWNoReweightTrainer,
    "rlcr_split_random_target": RLCRSplitRandomTargetTrainer,
    "rlcr_split_da": RLCRSplitDATrainer,
    "rlcr_split_nostd": RLCRSplitNoStdTrainer,
    "rlcr_split_var_square": RLCRSplitVarSquareTrainer,
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
