#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash configs/train_cottonweed.sh
# Optional override:
#   GPU_ID=0 TOTAL_ITRS=640 BATCH_SIZE=8 CROP_SIZE=512 bash configs/train_cottonweed.sh

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/dataset/cottonweed}"
GPU_ID="${GPU_ID:-0}"
MODEL="${MODEL:-deeplabv3plus_mobilenet}"
OUTPUT_STRIDE="${OUTPUT_STRIDE:-16}"
LR="${LR:-0.01}"
TOTAL_ITRS="${TOTAL_ITRS:-640}"
BATCH_SIZE="${BATCH_SIZE:-8}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
CROP_SIZE="${CROP_SIZE:-512}"
VAL_INTERVAL="${VAL_INTERVAL:-100}"
MASK_DIR="${MASK_DIR:-masks_trainid}"

cd "${PROJECT_DIR}"
python main.py \
  --model "${MODEL}" \
  --dataset cottonweed \
  --data_root "${DATA_ROOT}" \
  --cottonweed_mask_dir "${MASK_DIR}" \
  --gpu_id "${GPU_ID}" \
  --output_stride "${OUTPUT_STRIDE}" \
  --lr "${LR}" \
  --total_itrs "${TOTAL_ITRS}" \
  --batch_size "${BATCH_SIZE}" \
  --val_batch_size "${VAL_BATCH_SIZE}" \
  --crop_size "${CROP_SIZE}" \
  --val_interval "${VAL_INTERVAL}"
