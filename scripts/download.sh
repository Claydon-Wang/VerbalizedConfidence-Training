export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/sharedata/ssd_large/common/LLMs/
export HF_DATASETS_CACHE=/mnt/sharedata/ssd_large/common/datasets/

python - << 'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-1.7B"

AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True
)

AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)
EOF