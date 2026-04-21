#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${PROJECT_DIR}/workdirs"
DATA_ROOT="/root/autodl-tmp/dataset/cottonweed"
TOTAL_ITRS=640
VAL_INTERVAL=100
BATCH_SIZE=4
VAL_BATCH_SIZE=2
GPU_ID=0

mkdir -p "${WORKDIR}"

MODELS=(
  "deeplabv3plus_convnext_tiny"
  "deeplabv3plus_convnext_small"
  "deeplabv3plus_swin_tiny"
  "deeplabv3plus_efficientnet_b3"
)

echo "[$(date '+%F %T')] Start backbone ablation training"
echo "Project: ${PROJECT_DIR}"
echo "Data root: ${DATA_ROOT}"
echo "Logs dir: ${WORKDIR}"

for model in "${MODELS[@]}"; do
  ts="$(date '+%Y%m%d_%H%M%S')"
  exp_name="cottonweed_${model}_itr${TOTAL_ITRS}_${ts}"
  log_file="${WORKDIR}/train_cottonweed_${model}_itr${TOTAL_ITRS}.log"

  echo "[$(date '+%F %T')] Running ${model}"
  echo "[$(date '+%F %T')] Log: ${log_file}"

  conda run -n Depplab python "${PROJECT_DIR}/main.py" \
    --dataset cottonweed \
    --data_root "${DATA_ROOT}" \
    --model "${model}" \
    --total_itrs "${TOTAL_ITRS}" \
    --val_interval "${VAL_INTERVAL}" \
    --batch_size "${BATCH_SIZE}" \
    --val_batch_size "${VAL_BATCH_SIZE}" \
    --gpu_id "${GPU_ID}" \
    --work_dir "${WORKDIR}" \
    --exp_name "${exp_name}" \
    > "${log_file}" 2>&1

  echo "[$(date '+%F %T')] Finished ${model}"
done

echo "[$(date '+%F %T')] All runs finished."
