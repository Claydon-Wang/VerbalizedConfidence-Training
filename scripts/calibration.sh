export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/sharedata/ssd_large/common/LLMs/
export HF_DATASETS_CACHE=/mnt/sharedata/ssd_large/common/datasets/

# isotonic_regression temperature_scaling platt_scaling beta_calibration histogram_binning

# trained on training set
# python -m src.calibration.calibrate_main \
#   --fit_path logs/eval/HotpotRLCR_Train/Qwen25_7B_Instruct/Baseline/self_consistency/predictions.jsonl \
#   --methods histogram_binning


# Fit on train and report calibration on eval.
python -m src.calibration.calibrate_main \
  --fit_path logs/eval/HotpotRLCR_Train/Qwen25_7B_Instruct/Baseline/self_consistency/predictions.jsonl \
  --eval_path logs/eval/HotpotRLCR_Eval/Qwen25_7B_Instruct/Baseline/self_consistency/predictions.jsonl \
  --methods isotonic_regression temperature_scaling platt_scaling beta_calibration histogram_binning
