# Hugging Face mirrors/caches (optional, adjust as needed)
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/sharedata/ssd_large/common/LLMs/
export HF_DATASETS_CACHE=/mnt/sharedata/ssd_large/common/datasets/

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 --config_file deepspeed.yaml -m src.train.runner --config src/train/configs/Qwen-7B/hotpot/RLCR.yaml
