export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/sharedata/ssd_large/common/LLMs/
export HF_DATASETS_CACHE=/mnt/sharedata/ssd_large/common/datasets/

GPU=7
INFERENCER=verbalized_confidence

DATASETS=(
  "Hotpot"
  "HotpotVanilla"
  "SimpleQA"
  "TriviaQA"
  "CommonsenseQA"
  "GPQA"
  "MMLUPro"
  "GSM8K"
  "Math500"
)

# different methods
for dataset in "${DATASETS[@]}"; do
  CUDA_VISIBLE_DEVICES=$GPU python -m src.eval.eval_main --dataset "$dataset" --model Qwen25_1_5B_Instruct --inferencer "$INFERENCER"
  CUDA_VISIBLE_DEVICES=$GPU python -m src.eval.eval_main --dataset "$dataset" --model Qwen25_1_5B_Instruct --inferencer "$INFERENCER" --policy HotpotRLVR --checkpoint logs/train/hotpot/rlvr/qwen25_1_5b_instruct/2026-0322-0454
  CUDA_VISIBLE_DEVICES=$GPU python -m src.eval.eval_main --dataset "$dataset" --model Qwen25_1_5B_Instruct --inferencer "$INFERENCER" --policy HotpotRLCR --checkpoint logs/train/hotpot/rlcr/qwen25_1_5b_instruct/2026-0322-0450
  CUDA_VISIBLE_DEVICES=$GPU python -m src.eval.eval_main --dataset "$dataset" --model Qwen25_1_5B_Instruct --inferencer "$INFERENCER" --policy HotpotCoCA --checkpoint logs/train/hotpot/coca/qwen25_1_5b_instruct/2026-0322-1101

  # CUDA_VISIBLE_DEVICES=$GPU python -m src.eval.eval_main --dataset "$dataset" --model Qwen25_1_5B_Instruct --inferencer "$INFERENCER" --policy MathRLVR --checkpoint /path/to/math-rlvr-checkpoint
  # CUDA_VISIBLE_DEVICES=$GPU python -m src.eval.eval_main --dataset "$dataset" --model Qwen25_1_5B_Instruct --inferencer "$INFERENCER" --policy MathRLCR --checkpoint /path/to/math-rlcr-checkpoint
  # CUDA_VISIBLE_DEVICES=$GPU python -m src.eval.eval_main --dataset "$dataset" --model Qwen25_1_5B_Instruct --inferencer "$INFERENCER" --policy MathCoCA --checkpoint /path/to/math-coca-checkpoint
done
