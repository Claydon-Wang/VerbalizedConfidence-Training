# Installation

## Environment Setup

```bash
conda create -n rl python=3.10
conda activate rl
```

## Install Project Dependencies

```bash
pip install -r requirements.txt -i https://mirrors.cloud.tencent.com/pypi/simple/
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## TRL Installation

```bash
cd ../
git clone https://github.com/huggingface/trl.git
cd trl
git checkout 69ad852e5654a77f1695eb4c608906fe0c7e8624
pip install -e .
```

## Experiment Tracking

SwanLab:

```bash
swanlab login
```

Then enable it in your training config:

```yaml
report_to:
  - swanlab
```

This codebase still contains direct `wandb` API calls, so SwanLab is expected to be used through its compatibility bridge rather than by replacing imports.

## Accelerate and DeepSpeed

Make sure `accelerate` and `deepspeed` are configured for your available hardware. A launcher config for multi-GPU training is provided at [deepspeed.yaml](/mnt/sharedata/ssd_large/users/wsy/project/rl/RLCR/src/train/configs/launch/deepspeed.yaml).
