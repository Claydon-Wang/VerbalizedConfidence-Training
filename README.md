# RL Confidence Training & Evaluation
This is a training and evaluation codebase for reinforcement-learning style optimization of language models with answer and confidence outputs.

The repository is organized around two executable pipelines:

- `src/train`: training, reward computation, tracking, and checkpointing
- `src/eval`: offline evaluation, confidence scoring, and metric aggregation

## Quick Start

For concrete command examples, refer directly to:

- [train.sh](./scripts/train.sh): training entrypoint
- [eval.sh](./scripts/eval.sh): evaluation entrypoint


Typical training launch:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch \
  --num_processes 4 \
  --config_file src/train/configs/launch/deepspeed.yaml \
  -m src.train.train_main \
  --dataset Hotpot \
  --method RLCR \
  --model Qwen25_1_5B_Instruct
```

Typical evaluation launch:

```bash
# Baseline
CUDA_VISIBLE_DEVICES=0 \
python -m src.eval.eval_main \
  --dataset Hotpot \
  --model Qwen25_1_5BInstruct \
  --inferencer verbalized_confidence

# RLVR
CUDA_VISIBLE_DEVICES=0 \
python -m src.eval.eval_main \
  --dataset Hotpot \
  --model Qwen25_1_5BInstruct \
  --inferencer verbalized_confidence \
  --policy HotpotRLVR \
  --checkpoint {checkpoint_dir_here}

```

## Training Modules

The training pipeline is driven by [train_main.py](./src/train/train_main.py).

Main modules in `src/train`:

- [configs/train/](./src/train/configs/train/): dataset, model, and method presets used by training.
- [trainers/](./src/train/trainers/): trainer implementations and trainer registry.
- [rewards/](./src/train/rewards/): reward construction and reward functions.

Implemented training methods currently include:

- `RLVR`: standard GRPO-style optimization over `think + answer`
- `RLCR`: answer, uncertainty analysis, and confidence training
- `CoCA`: segmented optimization with separate answer and confidence advantages

## Evaluation Modules

The evaluation pipeline is driven by [eval_main.py](./src/eval/eval_main.py).

Main modules in `src/eval`:

- [configs/](./src/eval/configs/): evaluation config composition for datasets, models, and policies.
- [datasets/](./src/eval/datasets/): dataset loaders and dataset-specific formatting.
- [models/](./src/eval/models/): model wrappers used during evaluation.
- [inferencers/](./src/eval/inferencers/): inference pipelines for answer and confidence generation.
- [evaluators/](./src/eval/evaluators/): metric computation and evaluation summaries.
- [verifiers/](./src/eval/verifiers/): answer verification logic, including exact-match and LLM-based verification.

## Runtime Outputs
- Training checkpoints and train summaries are stored under `logs/train/`.
- Evaluation logs, metrics, and reliability diagrams are stored under `logs/eval/`.
- Local tracking artifacts are stored under `temp/exp_tracking/`.