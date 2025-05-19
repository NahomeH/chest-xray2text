#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# End-to-end baseline pipeline for CXR-report summarization (retrieval method)
#
# Usage:
#   bash scripts/run_baseline.sh \
#        /path/to/raw_images \
#        /path/to/raw_metadata.csv \
#        /path/to/output_root
#
# Arguments:
#   1. RAW_IMAGE_ROOT      Directory containing MIMIC-CXR JPG images
#   2. RAW_META_FILE       CSV or JSONL with image paths + free-text reports
#   3. OUTPUT_ROOT         Folder to store processed data, embeddings, outputs
#
# ---------------------------------------------------------------------------

set -euo pipefail

# -------------------------- Parse CLI arguments -----------------------------
if [[ $# -lt 3 ]]; then
  echo "Usage: bash $0 <RAW_IMAGE_ROOT> <RAW_META_FILE> <OUTPUT_ROOT>"
  exit 1
fi

RAW_IMAGE_ROOT=$1
RAW_META_FILE=$2
OUTPUT_ROOT=$3

# Create directory tree
PROC_DIR="${OUTPUT_ROOT}/data/processed"
EMB_DIR="${OUTPUT_ROOT}/models"
OUT_DIR="${OUTPUT_ROOT}/outputs"
mkdir -p "$PROC_DIR" "$EMB_DIR" "$OUT_DIR"

# -------------------------- 1. Preprocessing --------------------------------
echo ">>> [1/4] Preprocessing dataset ..."
python -m src.preprocess \
  --input_file    "$RAW_META_FILE" \
  --image_root    "$RAW_IMAGE_ROOT" \
  --output_csv    "${PROC_DIR}/clean_meta.csv" \
  --output_imgdir "${PROC_DIR}/images" \
  --img_size      224

# -------------------------- 2. Encode images --------------------------------
echo ">>> [2/4] Encoding training images ..."
python -m src.encode_images \
  --csv_file     "${PROC_DIR}/clean_meta.csv" \
  --image_root   "${PROC_DIR}/images" \
  --out_file     "${EMB_DIR}/embeddings.pt" \
  --model        "ViT-B/32" \
  --batch_size   64

# -------------------------- 3. Retrieval ------------------------------------
echo ">>> [3/4] Retrieving findings for validation set ..."
python -m src.retrieve \
  --embedding_file "${EMB_DIR}/embeddings.pt" \
  --test_csv       "${PROC_DIR}/clean_meta.csv" \
  --image_root     "${PROC_DIR}/images" \
  --output_file    "${OUT_DIR}/predictions.jsonl" \
  --model          "ViT-B/32" \
  --batch_size     64 \
  --top_k          1

# -------------------------- 4. Evaluation -----------------------------------
echo ">>> [4/4] Evaluating predictions ..."
python -m src.evaluate \
  --pred_file "${OUT_DIR}/predictions.jsonl" \
  --ref_file  "${PROC_DIR}/clean_meta.csv" \
  --out_file  "${OUT_DIR}/metrics.jsonl"

echo "Pipeline finished. Results saved to ${OUT_DIR}"
