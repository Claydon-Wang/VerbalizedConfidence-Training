# Evaluation

## Entry Point

Evaluation is launched through [`src/eval/eval_main.py`](../src/eval/eval_main.py).

The evaluation pipeline combines:

- dataset presets from [`src/eval/configs/datasets.py`](../src/eval/configs/datasets.py)
- model presets from [`src/eval/configs/models.py`](../src/eval/configs/models.py)
- policy presets from [`src/eval/configs/policies.py`](../src/eval/configs/policies.py)
- inferencer presets from [`src/eval/inferencers/`](../src/eval/inferencers/)

For ready-to-run command examples, refer to [`scripts/eval.sh`](../scripts/eval.sh).

## Main Components

- [`src/eval/datasets/`](../src/eval/datasets/): dataset loaders and formatting
- [`src/eval/models/`](../src/eval/models/): model wrappers and checkpoint loading
- [`src/eval/inferencers/`](../src/eval/inferencers/): generation procedures for different confidence estimation styles
- [`src/eval/verifiers/`](../src/eval/verifiers/): correctness verification
- [`src/eval/evaluators/`](../src/eval/evaluators/): metrics, calibration scores, and summary tables

## Inferencers

Implemented inferencers include:

- `base`
- `verbalized_confidence`
- `answer_sequence_likelihood`
- `self_consistency`
- `p_true`

The inferencer also determines which system prompt is used. Prompt resolution logic lives in [`src/eval/inferencers/base_inferencer.py`](../src/eval/inferencers/base_inferencer.py).

## Policies

Policies capture which fine-tuned dataset and algorithm a checkpoint corresponds to. Current built-in examples include:

- `HotpotRLVR`
- `HotpotRLCR`
- `MathRLVR`
- `MathRLCR`

If you add new training algorithms and want named evaluation presets for them, update [`src/eval/configs/policies.py`](../src/eval/configs/policies.py) and the prompt-resolution logic in [`src/eval/inferencers/base_inferencer.py`](../src/eval/inferencers/base_inferencer.py).

## Verification

Correctness checking uses dataset-appropriate logic:

- math-style answers primarily use `math_verify`
- some QA tasks fall back to normalized exact match
- optional LLM-based verification is implemented separately

Verification code lives in:

- [`src/eval/verifiers/accuracy_verifier.py`](../src/eval/verifiers/accuracy_verifier.py)
- [`src/eval/verifiers/llm_accuracy_verifier.py`](../src/eval/verifiers/llm_accuracy_verifier.py)

## Metrics

Evaluation reports include:

- `pass@k`
- `accuracy`
- `brier_score`
- `ece`
- `auroc`

Reliability diagrams are produced by the confidence evaluator when confidence and correctness columns are available.

## Launch Examples

### Base model evaluation

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m src.eval.eval_main \
  --dataset MMLUPro \
  --model Qwen25_7BInstruct \
  --inferencer verbalized_confidence
```

### Evaluate a fine-tuned RLVR checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m src.eval.eval_main \
  --dataset Hotpot \
  --model Qwen25_1_5B \
  --inferencer verbalized_confidence \
  --policy HotpotRLVR \
  --checkpoint /path/to/checkpoint
```

### Evaluate a fine-tuned RLCR checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m src.eval.eval_main \
  --dataset Hotpot \
  --model Qwen25_1_5B \
  --inferencer verbalized_confidence \
  --policy HotpotRLCR \
  --checkpoint /path/to/checkpoint
```

## Outputs

Evaluation artifacts are written under `logs/eval/`, including:

- raw predictions
- metric summaries
- calibration outputs
- reliability diagrams

The shell examples in [`scripts/eval.sh`](../scripts/eval.sh) show common invocation patterns for batch evaluation across multiple datasets.
