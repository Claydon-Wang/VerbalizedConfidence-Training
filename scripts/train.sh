# Hugging Face mirrors/caches (optional, adjust as needed)
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/sharedata/ssd_large/common/LLMs/
export HF_DATASETS_CACHE=/mnt/sharedata/ssd_large/common/datasets/

# Uncomment one command at a time.

## HOTPOT
# RLVR
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# accelerate launch --num_processes 4 \
#   --config_file src/train/configs/launch/deepspeed.yaml \
#   -m src.train.train_main \
#   --dataset Hotpot \
#   --method RLVR \
#   --model Qwen25_1_5B_Instruct

# RLCR
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# accelerate launch --num_processes 8 \
#   --config_file src/train/configs/launch/deepspeed.yaml \
#   -m src.train.train_main \
#   --dataset Hotpot \
#   --method RLCR \
#   --model Qwen25_1_5B_Instruct

# CoCA
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# accelerate launch --num_processes 8 \
#   --config_file src/train/configs/launch/deepspeed.yaml \
#   -m src.train.train_main \
#   --dataset Hotpot \
#   --method CoCA \
#   --model Qwen25_1_5B_Instruct


## MATH
# RLVR
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
# accelerate launch --num_processes 6 \
#   --config_file src/train/configs/launch/deepspeed.yaml \
#   -m src.train.train_main \
#   --dataset Math \
#   --method RLVR \
#   --model Qwen25_1_5B_Instruct

# RLCR
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
# accelerate launch --num_processes 6 \
#   --config_file src/train/configs/launch/deepspeed.yaml \
#   -m src.train.train_main \
#   --dataset Math \
#   --method MathRLCR \
#   --model Qwen25_1_5B_Instruct

# CoCA
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
# accelerate launch --num_processes 6 \
#   --config_file src/train/configs/launch/deepspeed.yaml \
#   -m src.train.train_main \
#   --dataset Math \
#   --method MathCoCA \
#   --model Qwen25_1_5B_Instruct

# Generation batch size = num_processes * per_device_train_batch_size * gradient_accumulation_steps
