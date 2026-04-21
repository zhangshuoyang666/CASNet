#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${PROJECT_DIR}/workdirs"

DATA_ROOT="${1:-/root/autodl-tmp/dataset/cottonweed}"
GPU_ID="${2:-0}"
TOTAL_ITRS="${3:-1600}"
VAL_INTERVAL="${4:-32}"
BATCH_SIZE="${5:-16}"
VAL_BATCH_SIZE="${6:-4}"
OUTPUT_STRIDE="${7:-16}"
EXP_NAME="${8:-exp_spatial_cbam_boundaryaux_best_itr1600}"

mkdir -p "${WORK_DIR}"
cd "${PROJECT_DIR}"

python -u main.py \
  --dataset cottonweed \
  --data_root "${DATA_ROOT}" \
  --model deeplabv3plus_mobilenet \
  --attention_type spatial_cbam \
  --enable_boundary_aux \
  --edge_loss_type bce \
  --lambda_edge 0.1 \
  --boundary_width 5 \
  --gpu_id "${GPU_ID}" \
  --output_stride "${OUTPUT_STRIDE}" \
  --total_itrs "${TOTAL_ITRS}" \
  --val_interval "${VAL_INTERVAL}" \
  --batch_size "${BATCH_SIZE}" \
  --val_batch_size "${VAL_BATCH_SIZE}" \
  --work_dir "${WORK_DIR}" \
  --exp_name "${EXP_NAME}"
