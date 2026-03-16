# Beyond Binary Rewards: Training LMs to Reason about Their Uncertainty

This repository contains the official code for the paper:

> **Beyond Binary Rewards: Training LMs to Reason about Their Uncertainty**  
> Mehul Damani, Isha Puri, Stewart Slocum, Idan Shenfeld, Yoon Kim, Jacob Andreas  
> *[arXiv:2507.16806](https://arxiv.org/abs/2507.16806)*

This repository builds on top of [TRL](https://github.com/huggingface/trl) and [Open-R1](https://github.com/huggingface/open-r1). We thank the authors and maintainers of these projects.

---

## 🛠 Installation

### Environment Setup


```bash
git clone https://github.com/damanimehul/RLCR.git
cd RLCR 
conda env create -f environment.yml
conda activate rl
```
If this build fails, an alternative is to create a conda env from scratch with the correct torch, vllm, transformers, math-verify, datasets, accelerate and deepspeed versions. 

### TRL Installation 

```bash
cd ../
git clone https://github.com/huggingface/trl.git
cd trl/
git checkout 69ad852e5654a77f1695eb4c608906fe0c7e8624
pip install -e .
```

### Login to wandb 

```bash
wandb login
```

### Or log to SwanLab instead of wandb

This repo contains a few direct `wandb` API calls such as `wandb.log(...)` and
`wandb.Table(...)`, so the supported SwanLab path is the compatibility bridge
rather than replacing imports.

```bash
swanlab login
```

Then enable the bridge in your training config:

```yaml
report_to:
  - swanlab
```

This keeps the existing `wandb` logging code working while routing the run to SwanLab.

### Deepspeed & Accelerate Setup

Ensure `deepspeed` and `accelerate` are properly configured for the available hardware ([guide](https://huggingface.co/docs/accelerate/en/index)). 
We provide a launcher config at `src/train/configs/launch/deepspeed.yaml` for 4 GPUs. 

### HuggingFace Models and Data
All models and datasets are available at this [RLCR HuggingFace Collection.](https://huggingface.co/collections/mehuldamani/rlcr-68912f9731b0bce30e4cc8c0)

---

## 🚀 Training

To run RLCR on hotpot:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 --config_file src/train/configs/launch/deepspeed.yaml -m src.train.runner --dataset Hotpot --method RLCR --model Qwen25_7B
```

### 📝 Notes

- Wandb logs from reproduced runs available [here](https://wandb.ai/mehuldamani/RLCR?nw=nwusermehuldamani). Intermediate generations are also logged and are useful for debugging. 
- Additional training scripts are available in `train_runs.sh`.
- Train config classes live in `src/train/configs/train/`, split into dataset, method, and model registries.
- Launch configs for accelerate/deepspeed live in `src/train/configs/launch/`.
- **Compute details**:
  - We ran HotpotQA experiments on **4×A100 GPUs**
  - Math experiments were run on **6×A100 GPUs**
  - The **generation batch size** is computed as:
    ```
    generation_batch_size = num_processes × per_device_train_batch_size × gradient_accumulation_steps
    ```
    It should be kept **constant or increased** if more compute is available. Lowering it may lead to instability during training.
- **Limitations**:
  - This field is evolving rapidly. We believe that both the **base RL implementation** and **hyperparameter settings** can be further improved. Doing so may reduce some training instabilities we encountered and enhance model calibration and reasoning quality.
  - Learning well-calibrated policies requires exploration over a range of verbalized confidences. If training problems have similar difficulty, the policy may collapse to outputting a narrow band—or even a single—confidence value, hindering calibration. If this behavior is encountered, incentivizing more exploration in verbalized confidence scores through higher temperature/modifications to system prompt can be effective.

We welcome suggestions and contributions!

---

## 📊 Evaluation

To run inference with our trained RLCR model on a single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.eval.inference_example
```

### 📚 Available Models

| Name              | Training Dataset                                         | Model Path                                             | System Prompt   |
|:------------------|:------------------------------|:--------------------------------------------------------------|:-------------|
| RLCR-hotpot       | HotpotQA-Modified                          | mehuldamani/hotpot-v2-brier-7b-no-split |   TABC_Long     |
| RLVR-hotpot       | HotpotQA-Modified (RLVR)        | mehuldamani/hotpot-v2-correctness-7b    |    GEN   |
| Classifier-hotpot | HotpotQA-Modified (Classifier)  | mehuldamani/orm-hotpot-v2-final-correctness  |    Gen             |
| RLCR-math         | Big-Math-Digits                            | mehuldamani/big-math-digits-v2-brier-base-tabc |       TABC          |
| SFT-RLCR-math     | Big-Math-Digits (SFT Warmup)                    | mehuldamani/big-math-digits-v2-brier |       TABC          |
| RLVR-math         | Big-Math-Digits (RLVR)          | mehuldamani/big-math-digits-v2-correctness    |      Gen           |
| Classifier-math   | Big-Math-Digits (Classifier) | mehuldamani/orm-big-math-digits-v2-correctness  |      Gen           |

### 🧪 Sample Evaluation Run

Run evaluation on a dataset using a config:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.eval.evaluation --config src/eval/configs/Hotpot-models/trivia.json
```

For a full eval suite on a single GPU (We already provide the outputs/results from this):

```bash
bash scripts/eval.sh
```

### 📝 Notes

- Evaluation outputs and metrics are both stored under `logs/eval/`.
- Training checkpoints and trainer logs are stored under `logs/train/`.
- Local experiment tracking files for `wandb` and `swanlab` are stored under `temp/exp_tracking/`.
- To evaluate on new datasets/models, add them to config files inside `src/eval/configs/`.
- To add or update training runs, edit Python config classes inside `src/train/configs/train/`.
- Default evaluation uses `temperature = 0` and `max_tokens = 4096`.
- Currently supported evaluation functions:
  - **Exact Match** (Used for hotpotqa)
  - **Math Verify** (Used for all math datasets)
  - **LLM-as-a-Judge** (Used for trivia, simpleqa, commonsenseqa, gpqa)

## Project Structure

```text
src/
  common/    shared prompts and dataset processing
  train/     training code
    configs/ train configs and launch configs
  eval/      evaluation code, eval configs, inference example
logs/
  train/     checkpoints and train-side outputs
  eval/      evaluation datasets and metrics
scripts/     runnable shell entrypoints
```

---

## 📄 Citation

If you find this work useful, please cite:

```bibtex
@article{damani2025beyond,
  title={Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty},
  author={Damani, Mehul and Puri, Isha and Slocum, Stewart and Shenfeld, Idan and Choshen, Leshem and Kim, Yoon and Andreas, Jacob},
  journal={arXiv preprint arXiv:2507.16806},
  year={2025}
}
```
