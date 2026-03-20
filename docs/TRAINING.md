# Training

## Entry Point

Training is launched through [`src/train/train_main.py`](../src/train/train_main.py).

The launcher expects three preset class names:

- `--dataset`: dataset preset from [`src/train/configs/train/datasets.py`](../src/train/configs/train/datasets.py)
- `--method`: method preset from [`src/train/configs/train/methods.py`](../src/train/configs/train/methods.py)
- `--model`: model preset from [`src/train/configs/train/models.py`](../src/train/configs/train/models.py)

For ready-to-run command examples, refer to [`scripts/train.sh`](../scripts/train.sh).

The config loader composes those three presets, derives a timestamped `run_name`, and writes outputs under:

```text
logs/train/<dataset>/<method>/<model>/<timestamp>
```

The timestamp format is `YYYY-MMDD-HHMM`.

## Main Components

- [`src/train/configs/train/`](../src/train/configs/train/): composable presets for datasets, methods, and models
- [`src/train/trainers/`](../src/train/trainers/): trainer implementations and registry
- [`src/train/rewards/`](../src/train/rewards/): reward functions used during rollout scoring
- [`src/common/system_prompts.py`](../src/common/system_prompts.py): system prompt templates used to build training conversations

## Implemented Methods

### RLVR

- Trainer: `rlvr`
- Default format: `think_answer`
- Objective: standard GRPO-style optimization using a single scalar reward over the completion

### RLCR

- Trainer: `rlcr`
- Default format: `think_answer_analysis_confidence`
- Objective: optimize answer quality together with confidence-related rewards

### CoCA

- Trainer: `coca`
- Default format: `think_answer_confidence`
- Objective: compute separate answer and confidence rewards, normalize them separately within each rollout group, and apply them to different token spans

Current CoCA implementation details:

- Answer correctness is extracted only from `<answer>...</answer>`
- Confidence is extracted only from `<confidence>...</confidence>`
- `answer` and `confidence` rewards are decoupled
- The answer-side advantage is applied to non-confidence completion tokens
- The confidence-side advantage is applied only to confidence tokens

## Common Presets

### Datasets

- `Hotpot`
- `Math`

### Models

- `Qwen25_1_5B`
- `Qwen25_1_5BInstruct`
- `Qwen25_7B`
- `Qwen25_7BInstruct`

### Methods

- `RLVR`
- `RLCR`
- `MathRLCR`
- `CoCA`
- `MathCoCA`
- `RLCRSFT`

## Launch Examples

### Hotpot + CoCA + Qwen2.5-1.5B Instruct

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch \
  --num_processes 4 \
  --config_file src/train/configs/launch/deepspeed.yaml \
  -m src.train.train_main \
  --dataset Hotpot \
  --method CoCA \
  --model Qwen25_1_5BInstruct
```

### Hotpot + RLVR + Qwen2.5-1.5B

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch \
  --num_processes 4 \
  --config_file src/train/configs/launch/deepspeed.yaml \
  -m src.train.train_main \
  --dataset Hotpot \
  --method RLVR \
  --model Qwen25_1_5B
```

### Math + MathCoCA + Qwen2.5-1.5B

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
accelerate launch \
  --num_processes 6 \
  --config_file src/train/configs/launch/deepspeed.yaml \
  -m src.train.train_main \
  --dataset Math \
  --method MathCoCA \
  --model Qwen25_1_5BInstruct
```

## Tracking and Outputs

Training writes to:

- `logs/train/...`: checkpoints, trainer state, model config, and run artifacts
- `temp/exp_tracking/`: local tracking backend artifacts

Tracking is configured through [`src/train/logger.py`](../src/train/logger.py) and the `report_to` field in training config presets. The repository is typically used with SwanLab through the `wandb` compatibility bridge described in [INSTALL.md](./INSTALL.md).

## Notes

- The repository defaults to `use_vllm=True` in training config base presets.
- In colocated vLLM mode, increasing `vllm_gpu_memory_utilization` may improve generation throughput, but it also reduces memory headroom for training.
- The shell examples in [`scripts/train.sh`](../scripts/train.sh) are a useful starting point, but the script is not a canonical source of truth. The preset classes under `src/train/configs/train/` are.
