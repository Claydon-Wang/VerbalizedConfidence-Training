export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/sharedata/ssd_large/common/LLMs/
export HF_DATASETS_CACHE=/mnt/sharedata/ssd_large/common/datasets/

GPU=7
INFERENCER=verbalized_confidence

DATASETS=(
  # Factual
  "Hotpot"
  # "HotpotVanilla"
  # "SimpleQA"
  # "TriviaQA"
  # "CommonsenseQA"
  # "GPQA"
  # "MMLUPro"

  # Math
  # "GSM8K"
  # "Math500"

  # Code
)

for dataset in "${DATASETS[@]}"; do
  CUDA_VISIBLE_DEVICES=$GPU python -m src.eval.eval_main --dataset "$dataset" --model Qwen25_1_5B_Instruct --inferencer "$INFERENCER"

  # CUDA_VISIBLE_DEVICES=$GPU python -m src.eval.eval_main --dataset "$dataset" --model Qwen25_1_5B_Instruct --inferencer "$INFERENCER" --policy HotpotRLVR --checkpoint /mnt/sharedata/ssd_large/users/wsy/project/rl/RLCR/temp/train/RLVR-hotpot
  # CUDA_VISIBLE_DEVICES=$GPU python -m src.eval.eval_main --dataset "$dataset" --model Qwen25_1_5B_Instruct --inferencer "$INFERENCER" --policy HotpotRLCR --checkpoint /mnt/sharedata/ssd_large/users/wsy/project/rl/RLCR/temp/train/RLCR-base-hotpot
  # CUDA_VISIBLE_DEVICES=$GPU python -m src.eval.eval_main --dataset "$dataset" --model Qwen25_1_5B_Instruct --inferencer "$INFERENCER" --policy HotpotCoCA --checkpoint /path/to/hotpot-coca-checkpoint

  # CUDA_VISIBLE_DEVICES=$GPU python -m src.eval.eval_main --dataset "$dataset" --model Qwen25_1_5B_Instruct --inferencer "$INFERENCER" --policy MathRLVR --checkpoint /path/to/math-rlvr-checkpoint
  # CUDA_VISIBLE_DEVICES=$GPU python -m src.eval.eval_main --dataset "$dataset" --model Qwen25_1_5B_Instruct --inferencer "$INFERENCER" --policy MathRLCR --checkpoint /path/to/math-rlcr-checkpoint
  # CUDA_VISIBLE_DEVICES=$GPU python -m src.eval.eval_main --dataset "$dataset" --model Qwen25_1_5B_Instruct --inferencer "$INFERENCER" --policy MathCoCA --checkpoint /path/to/math-coca-checkpoint
done
