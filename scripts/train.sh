# Hugging Face mirrors/caches (optional, adjust as needed)
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/sharedata/ssd_large/common/LLMs/
export HF_DATASETS_CACHE=/mnt/sharedata/ssd_large/common/datasets/

## HOTPOT (4 GPU config) 
# RLVR
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 --config_file src/train/configs/launch/deepspeed.yaml -m src.train.runner --dataset Hotpot --method RLVR --model Qwen25_1_5B
# RLCR
#CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 --config_file src/train/configs/launch/deepspeed.yaml -m src.train.runner --dataset Hotpot --method RLCR --model Qwen25_7B


## MATH (6 GPU config) 
# RLVR
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 accelerate launch --num_processes 6 --config_file src/train/configs/launch/deepspeed.yaml -m src.train.runner --dataset Math --method RLVR --model Qwen25_7B
# RLCR
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 accelerate launch --num_processes 6 --config_file src/train/configs/launch/deepspeed.yaml -m src.train.runner --dataset Math --method MathRLCR --model Qwen25_7B
# SFT+RLCR
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 accelerate launch --num_processes 6 --config_file src/train/configs/launch/deepspeed.yaml -m src.train.runner --dataset Math --method RLCRSFT --model Qwen25_7B

## The generation batch size = num_processes * per_device_train_batch_size * gradient_accumulation_steps
## If more gpus are used, training can be sped up by reducing the gradient accumulation steps and increasing num_processes
## For 7B model, generally a minimum of 4 gpus is needed for training  
