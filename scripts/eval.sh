export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/sharedata/ssd_large/common/LLMs/
export HF_DATASETS_CACHE=/mnt/sharedata/ssd_large/common/datasets/

GPU=7

DATASETS=(
  "CommonsenseQA"
  "GPQA"
  "GSM8K"
  "Hotpot"
  "HotpotVanilla"
  "Math500"
  "SimpleQA"
  "Trivia"
)

for dataset in "${DATASETS[@]}"; do
  CUDA_VISIBLE_DEVICES=$GPU python -m src.eval.eval_main --dataset "$dataset" --model Qwen25_1_5B
  CUDA_VISIBLE_DEVICES=$GPU python -m src.eval.eval_main --dataset "$dataset" --model Qwen25_1_5B --policy HotpotRLVR --checkpoint /mnt/sharedata/ssd_large/users/wsy/project/rl/RLCR/temp/train/RLVR-hotpot
  CUDA_VISIBLE_DEVICES=$GPU python -m src.eval.eval_main --dataset "$dataset" --model Qwen25_1_5B --policy HotpotRLCR --checkpoint /mnt/sharedata/ssd_large/users/wsy/project/rl/RLCR/temp/train/RLCR-base-hotpot
done

# #HOTPOT models
# CUDA_VISIBLE_DEVICES=0 python -m src.eval.eval_main --dataset CommonsenseQA --model Qwen25_7B
# CUDA_VISIBLE_DEVICES=0 python -m src.eval.eval_main --dataset CommonsenseQA --model Qwen25_7B --policy HotpotRLVR --checkpoint mehuldamani/hotpot-v2-correctness-7b
# CUDA_VISIBLE_DEVICES=0 python -m src.eval.eval_main --dataset CommonsenseQA --model Qwen25_7B --policy HotpotRLCR --checkpoint mehuldamani/hotpot-v2-brier-7b-no-split
# CUDA_VISIBLE_DEVICES=7 python -m src.eval.eval_main --dataset Hotpot --model Qwen25_7B
# CUDA_VISIBLE_DEVICES=0 python -m src.eval.eval_main --dataset HotpotVanilla --model Qwen25_7B --policy HotpotRLVR --checkpoint mehuldamani/hotpot-v2-correctness-7b
# CUDA_VISIBLE_DEVICES=0 python -m src.eval.eval_main --dataset Trivia --model Qwen25_7B --policy HotpotRLCR --checkpoint mehuldamani/hotpot-v2-brier-7b-no-split

# #MATH models
# CUDA_VISIBLE_DEVICES=0 python -m src.eval.eval_main --dataset BigMathDigits --model Qwen25_7B
# CUDA_VISIBLE_DEVICES=0 python -m src.eval.eval_main --dataset GSM8K --model Qwen25_7B --policy MathRLVR --checkpoint mehuldamani/big-math-digits-v2-correctness
# CUDA_VISIBLE_DEVICES=0 python -m src.eval.eval_main --dataset Math500 --model Qwen25_7B --policy MathRLCR --checkpoint mehuldamani/big-math-digits-v2-brier-base-tabc
# CUDA_VISIBLE_DEVICES=0 python -m src.eval.eval_main --dataset GPQA --model Qwen25_7B --policy MathRLCRSFT --checkpoint mehuldamani/big-math-digits-v2-brier

# #TRIVIA models
# CUDA_VISIBLE_DEVICES=0 python -m src.eval.eval_main --dataset Trivia --model Qwen25_1_5B
# CUDA_VISIBLE_DEVICES=0 python -m src.eval.eval_main --dataset Trivia --model Qwen25_1_5BInstruct
