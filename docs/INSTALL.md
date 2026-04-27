# Installation

## Environment Setup

```bash
conda create -n rl python=3.10
conda activate rl
```

## Install Project Dependencies

```bash
pip install -r requirements_new.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# flash attention
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3+cu128torch2.9-cp310-cp310-linux_x86_64.whl


# local
cd /mnt/sharedata/ssd_large/users/wsy/software
pip install flash_attn-2.8.3+cu128torch2.9-cp310-cp310-linux_x86_64.whl

# pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir -i https://mirrors.cloud.tencent.com/pypi/simple/

```

`requirements_new.txt` is the recommended environment for training Qwen3-family models such as `Qwen/Qwen3-1.7B`.

If you want to reproduce the older environment used by earlier experiments, install from `requirements.txt` instead.

## Optional TRL Development Install

```bash
cd ../
git clone https://github.com/huggingface/trl.git
cd trl
git checkout 69ad852e5654a77f1695eb4c608906fe0c7e8624
pip install -e .
```

This step is optional. The default installation path already installs `trl` from `requirements_new.txt`. Only use the editable install above if you explicitly want to debug or modify TRL itself.

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
