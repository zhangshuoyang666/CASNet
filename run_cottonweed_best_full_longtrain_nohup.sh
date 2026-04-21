#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${PROJECT_DIR}/workdirs"

DATA_ROOT="${1:-/root/autodl-tmp/dataset/cottonweed}"
GPU_ID="${2:-0}"
MODEL="${3:-deeplabv3plus_mobilenet}"
BATCH_SIZE="${4:-4}"
VAL_BATCH_SIZE="${5:-2}"
VAL_INTERVAL="${6:-100}"
LR="${7:-0.01}"
OUTPUT_STRIDE="${8:-16}"
CROP_SIZE="${9:-512}"
EARLY_STOP_PATIENCE="${10:-15}"
MIN_ITRS_BEFORE_STOP="${11:-3000}"
EARLY_STOP_MIN_DELTA="${12:-0.0005}"

mkdir -p "${WORK_DIR}"
cd "${PROJECT_DIR}"

timestamp="$(date '+%Y%m%d_%H%M%S')"
EXP_NAME="cottonweed_${MODEL}_full_longtrain_earlystop_${timestamp}"
LOG_FILE="${WORK_DIR}/train_${EXP_NAME}.log"

nohup conda run --no-capture-output -n Depplab python -u main.py \
  --dataset cottonweed \
  --data_root "${DATA_ROOT}" \
  --model "${MODEL}" \
  --gpu_id "${GPU_ID}" \
  --output_stride "${OUTPUT_STRIDE}" \
  --lr "${LR}" \
  --total_itrs -1 \
  --val_interval "${VAL_INTERVAL}" \
  --batch_size "${BATCH_SIZE}" \
  --val_batch_size "${VAL_BATCH_SIZE}" \
  --crop_size "${CROP_SIZE}" \
  --work_dir "${WORK_DIR}" \
  --exp_name "${EXP_NAME}" \
  --aspp_variant deform \
  --enable_fg_fusion \
  --enable_texture_enhance \
  --enable_decoder_detail \
  --enable_early_stop \
  --early_stop_metric "Mean IoU" \
  --early_stop_patience "${EARLY_STOP_PATIENCE}" \
  --early_stop_min_delta "${EARLY_STOP_MIN_DELTA}" \
  --min_itrs_before_early_stop "${MIN_ITRS_BEFORE_STOP}" \
  > "${LOG_FILE}" 2>&1 &

echo "Started long training."
echo "PID: $!"
echo "Exp: ${EXP_NAME}"
echo "Log: ${LOG_FILE}"
