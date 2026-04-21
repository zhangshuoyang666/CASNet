#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${PROJECT_DIR}/workdirs"
LOG_DIR="${WORK_DIR}"

DATA_ROOT="${1:-/root/autodl-tmp/dataset/cottonweed}"
GPU_ID="${2:-0}"
MODEL="${3:-deeplabv3plus_mobilenet}"
TOTAL_ITRS="${4:-640}"
VAL_INTERVAL="${5:-100}"
BATCH_SIZE="${6:-16}"
VAL_BATCH_SIZE="${7:-4}"
OUTPUT_STRIDE="${8:-16}"

ATTENTIONS=(none channel spatial se cbam cbam_light spatial_cbam ca)

mkdir -p "${WORK_DIR}" "${LOG_DIR}"

cd "${PROJECT_DIR}"
for attn in "${ATTENTIONS[@]}"; do
  EXP_NAME="cottonweed_${MODEL}_${attn}_itr${TOTAL_ITRS}"
  LOG_FILE="${LOG_DIR}/train_${EXP_NAME}.log"

  echo "[$(date '+%F %T')] Start attention=${attn}, log=${LOG_FILE}"
  python -u main.py \
    --dataset cottonweed \
    --data_root "${DATA_ROOT}" \
    --model "${MODEL}" \
    --attention_type "${attn}" \
    --gpu_id "${GPU_ID}" \
    --output_stride "${OUTPUT_STRIDE}" \
    --total_itrs "${TOTAL_ITRS}" \
    --val_interval "${VAL_INTERVAL}" \
    --batch_size "${BATCH_SIZE}" \
    --val_batch_size "${VAL_BATCH_SIZE}" \
    --work_dir "${WORK_DIR}" \
    --exp_name "${EXP_NAME}" > "${LOG_FILE}" 2>&1
  echo "[$(date '+%F %T')] Finished attention=${attn}"
done

echo "All attention experiments finished."
