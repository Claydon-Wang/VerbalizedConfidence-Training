#!/bin/bash
#SBATCH --job-name=rl_trainig # 任务名
#SBATCH --partition=A100 # 使用什么gpu L40 或者A100
#SBATCH --nodes=1 # 使用多少节点，单节点训练默认是1
#SBATCH --gres=gpu:8 #使用多少张gpu训练
#SBATCH --cpus-per-task=64 # 每个任务使用多少cpu，单节点训练默认是32
#SBATCH --mem=0 # 内存，0表示不限制 256G
#SBATCH --time=infinite # 任务最长运行时间，格式为 days-hours:minutes:seconds，infinite表示不限制
#SBATCH --output=logs/%x-%j.log # 标准输出日志文件，%x表示任务名，%j表示任务id


# ===== 环境 =====
module load python/miniconda3/26.3.2
module load cuda/12.9

eval "$(conda shell.bash hook)"
conda activate rl
mkdir -p logs

# ===== HF =====
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/common/LLMs/
export HF_DATASETS_CACHE=/data/common/datasets/


# ===== debug =====
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi


# ===== 启动 =====
accelerate launch --num_processes 8 \
  --config_file src/train/configs/launch/deepspeed.yaml \
  -m src.train.train_main \
  --dataset Hotpot \
  --method RLCR \
  --model Qwen25_1_5B_Instruct

  accelerate launch --num_processes 8 \
  --config_file src/train/configs/launch/deepspeed.yaml \
  -m src.train.train_main \
  --dataset Hotpot \
  --method CoCA \
  --model Qwen25_1_5B_Instruct

# ===== 启动 =====
# python -m accelerate.commands.launch --num_processes 8 \
#   --config_file src/train/configs/launch/deepspeed.yaml \
#   -m src.train.train_main \
#   --dataset Hotpot \
#   --method RLCR_split_ConfPureSFT \
#   --model Qwen3_1_7B
