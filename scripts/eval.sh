export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/sharedata/ssd_large/common/LLMs/
export HF_DATASETS_CACHE=/mnt/sharedata/ssd_large/common/datasets/

GPU=7
INFERENCER=verbalized_confidence

DATASETS=(
  # "HotpotRLCR_Train"
  # "BigMath_Train"
  "HotpotRLCR_Eval"
  "HotpotVanilla"
  "BigMath_Eval"
  "SimpleQA"
  "TriviaQA"
  "CommonsenseQA"
  "GPQA"
  "MMLUPro"
  "GSM8K"
  "Math500"
  "ScienceQA"
  "SQuAD"
  "BoolQ"
  "AIME2024"
)

# different methods
for dataset in "${DATASETS[@]}"; do
  CUDA_VISIBLE_DEVICES=$GPU python -m src.eval.eval_main --dataset "$dataset" --model Qwen25_1_5B_Instruct --inferencer "$INFERENCER"
  CUDA_VISIBLE_DEVICES=$GPU python -m src.eval.eval_main --dataset "$dataset" --model Qwen25_1_5B_Instruct --inferencer "$INFERENCER" --fine_tuned_algorithm RLVR --fine_tuned_dataset Hotpot --checkpoint logs/train/hotpot/rlvr/qwen25_1_5b_instruct/2026-0322-0454
  CUDA_VISIBLE_DEVICES=$GPU python -m src.eval.eval_main --dataset "$dataset" --model Qwen25_1_5B_Instruct --inferencer "$INFERENCER" --fine_tuned_algorithm RLCR --fine_tuned_dataset Hotpot --checkpoint logs/train/hotpot/rlcr/qwen25_1_5b_instruct/2026-0322-0450
  CUDA_VISIBLE_DEVICES=$GPU python -m src.eval.eval_main --dataset "$dataset" --model Qwen25_1_5B_Instruct --inferencer "$INFERENCER" --fine_tuned_algorithm CoCA --fine_tuned_dataset Hotpot --checkpoint logs/train/hotpot/coca/qwen25_1_5b_instruct/2026-0322-1101

  # CUDA_VISIBLE_DEVICES=$GPU python -m src.eval.eval_main --dataset "$dataset" --model Qwen25_1_5B_Instruct --inferencer "$INFERENCER" --fine_tuned_algorithm RLVR --fine_tuned_dataset BigMath --checkpoint /path/to/big-math-rlvr-checkpoint
  # CUDA_VISIBLE_DEVICES=$GPU python -m src.eval.eval_main --dataset "$dataset" --model Qwen25_1_5B_Instruct --inferencer "$INFERENCER" --fine_tuned_algorithm RLCR --fine_tuned_dataset BigMath --checkpoint /path/to/big-math-rlcr-checkpoint
  # CUDA_VISIBLE_DEVICES=$GPU python -m src.eval.eval_main --dataset "$dataset" --model Qwen25_1_5B_Instruct --inferencer "$INFERENCER" --fine_tuned_algorithm CoCA --fine_tuned_dataset BigMath --checkpoint /path/to/big-math-coca-checkpoint
done
