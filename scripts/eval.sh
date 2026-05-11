#!/bin/bash
#SBATCH --job-name=rl_eval # 任务名
#SBATCH --partition=L40 # 使用什么gpu L40 或者A100
#SBATCH --nodes=1 # 使用多少节点，单节点评测默认是1
#SBATCH --gres=gpu:1 # 使用多少张gpu评测
#SBATCH --cpus-per-task=12 # 每个任务使用多少cpu
#SBATCH --mem=128G # 内存，0表示不限制
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


# ===== 配置 =====
DATASETS=(
# "HotpotRLCR_Train"
# "BigMath_Train"
"HotpotRLCR_Eval"
"HotpotVanilla"
"BigMath_Eval"
"Math500"
"TriviaQA"
"CommonsenseQA"
"ScienceQA"

)

# verbalized_confidence answer_sequence_likelihood p_true self_consistency
inferencer=(verbalized_confidence)


# ===== 启动 =====
for dataset in "${DATASETS[@]}"; do
  for infer in "${inferencer[@]}"; do

    # python -m src.eval.eval_main --dataset "$dataset" --model Qwen25_1_5B_Instruct --inferencer "$infer"
    # python -m src.eval.eval_main --dataset "$dataset" --model Qwen25_1_5B_Instruct --inferencer "$infer" --fine_tuned_algorithm RLVR --fine_tuned_dataset Hotpot --checkpoint logs/train/hotpot/rlvr/qwen25_1_5b_instruct/2026-0322-0454
    # python -m src.eval.eval_main --dataset "$dataset" --model Qwen25_1_5B_Instruct --inferencer "$infer" --fine_tuned_algorithm RLCRSplit --fine_tuned_dataset Hotpot --checkpoint logs/train/hotpot/rlcr_split/qwen25_1_5b_instruct/2026-0423-0749
    # python -m src.eval.eval_main --dataset "$dataset" --model Qwen25_1_5B_Instruct --inferencer "$infer" --fine_tuned_algorithm RLCR --fine_tuned_dataset Hotpot --checkpoint logs/train/hotpot/rlcr/qwen25_1_5b_instruct/2026-0322-0450
    # python -m src.eval.eval_main --dataset "$dataset" --model Qwen25_1_5B_Instruct --inferencer "$infer" --fine_tuned_algorithm CoCA --fine_tuned_dataset Hotpot --checkpoint logs/train/hotpot/coca/qwen25_1_5b_instruct/2026-0322-1101

    # python -m src.eval.eval_main --dataset "$dataset" --model Qwen25_1_5B_Instruct --inferencer "$infer" --fine_tuned_algorithm RLCRSplitConfPureSFT --fine_tuned_dataset Hotpot --checkpoint logs/train/hotpot/rlcr_split_confpuresft/qwen25_1_5b_instruct/2026-0507-1300

    # python -m src.eval.eval_main --dataset "$dataset" --model Qwen3_1_7B --inferencer "$infer"
    # python -m src.eval.eval_main --dataset "$dataset" --model Qwen3_1_7B --inferencer "$infer" --fine_tuned_algorithm RLCR --fine_tuned_dataset Hotpot --checkpoint logs/train/hotpot/rlcr/qwen3_1_7b/2026-0427-1026
    # python -m src.eval.eval_main --dataset "$dataset" --model Qwen3_1_7B --inferencer "$infer" --fine_tuned_algorithm RLCRSplit --fine_tuned_dataset Hotpot --checkpoint logs/train/hotpot/rlcr_split/qwen3_1_7b/2026-0507-1055
    # python -m src.eval.eval_main --dataset "$dataset" --model Qwen3_1_7B --inferencer "$infer" --fine_tuned_algorithm CoCA --fine_tuned_dataset Hotpot --checkpoint logs/train/hotpot/coca/qwen3_1_7b/2026-0427-1929

  done
done
