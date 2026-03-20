export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/sharedata/ssd_large/common/LLMs/
export HF_DATASETS_CACHE=/mnt/sharedata/ssd_large/common/datasets/

GPU=6

DATASETS=(
  # "CommonsenseQA"
  # "GPQA"
  # "GSM8K"
  # "Hotpot"
  # "HotpotVanilla"
  # "Math500"
  # "SimpleQA"
  "TriviaQA"
)

for dataset in "${DATASETS[@]}"; do
#   CUDA_VISIBLE_DEVICES=$GPU python -m src.eval.eval_main --dataset "$dataset" --model Qwen25_1_5B --inferencer verbalized_confidence --policy HotpotRLVR --checkpoint /mnt/sharedata/ssd_large/users/wsy/project/rl/RLCR/temp/train/RLVR-hotpot
  CUDA_VISIBLE_DEVICES=$GPU python -m src.eval.eval_main --dataset "$dataset" --model Qwen25_1_5B --inferencer answer_sequence_likelihood --policy HotpotRLVR --checkpoint /mnt/sharedata/ssd_large/users/wsy/project/rl/RLCR/temp/train/RLVR-hotpot
  # CUDA_VISIBLE_DEVICES=$GPU python -m src.eval.eval_main --dataset "$dataset" --model Qwen25_1_5B --inferencer self_consistency --policy HotpotRLVR --checkpoint /mnt/sharedata/ssd_large/users/wsy/project/rl/RLCR/temp/train/RLVR-hotpot
  # CUDA_VISIBLE_DEVICES=$GPU python -m src.eval.eval_main --dataset "$dataset" --model Qwen25_1_5B --inferencer p_true --policy HotpotRLVR --checkpoint /mnt/sharedata/ssd_large/users/wsy/project/rl/RLCR/temp/train/RLVR-hotpot
  # CUDA_VISIBLE_DEVICES=$GPU python -m src.eval.eval_main --dataset "$dataset" --model Qwen25_1_5B --inferencer verbalized_confidence --policy HotpotRLVR --checkpoint /mnt/sharedata/ssd_large/users/wsy/project/rl/RLCR/temp/train/RLVR-hotpot
  # CUDA_VISIBLE_DEVICES=$GPU python -m src.eval.eval_main --dataset "$dataset" --model Qwen25_1_5B --inferencer verbalized_confidence --policy HotpotRLCR --checkpoint /mnt/sharedata/ssd_large/users/wsy/project/rl/RLCR/temp/train/RLCR-base-hotpot
done
