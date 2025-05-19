#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# End-to-end pipeline for the fine-tuned BLIP-2 (LoRA) model
#
# Assumes you have:
#  - configs/train_params.yml
#  - configs/lora_config.yml
#  - Preprocessed data in data/processed/
# ---------------------------------------------------------------------------

set -euo pipefail

# -------------------------- Config files & paths ----------------------------
TRAIN_CFG="configs/train_params.yml"
LORA_CFG="configs/lora_config.yml"

# You should ensure `output_dir` in train_params.yml matches this
MODEL_DIR=$(grep "^output_dir:" $TRAIN_CFG | awk '{print $2}')
VAL_CSV="data/processed/val_meta.csv"
IMAGE_ROOT="data/processed/images"
PRED_OUT="outputs/blip2_predictions.jsonl"
METRIC_OUT="outputs/blip2_metrics.jsonl"

mkdir -p "$MODEL_DIR" "$(dirname $PRED_OUT)" "$(dirname $METRIC_OUT)"

# -------------------------- 1. Train BLIP-2 with LoRA -----------------------
echo ">>> [1/3] Training BLIP-2 (LoRA) ..."
python src/train_blip2.py \
  --config    "$TRAIN_CFG" \
  --lora_cfg  "$LORA_CFG"

# -------------------------- 2. Generate on Validation Set ------------------
echo ">>> [2/3] Generating summaries with fine-tuned BLIP-2 ..."
python src/generate_blip2.py \
  --model_path   "$MODEL_DIR" \
  --test_csv     "$VAL_CSV" \
  --image_root   "$IMAGE_ROOT" \
  --output_file  "$PRED_OUT" \
  --batch_size   8

# -------------------------- 3. Evaluate Generations ------------------------
echo ">>> [3/3] Evaluating BLIP-2 summaries ..."
python src/evaluate_blip2.py \
  --pred_file  "$PRED_OUT" \
  --ref_file   "$VAL_CSV" \
  --out_file   "$METRIC_OUT"

echo "✅ Fine-tuned pipeline complete. Predictions: $PRED_OUT | Metrics: $METRIC_OUT"
