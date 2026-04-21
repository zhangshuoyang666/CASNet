#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/dataset/SoyCotton_deeplab}"
RAW_IMAGES_DIR="${RAW_IMAGES_DIR:-/root/autodl-tmp/dataset/SoyCotton/images}"
COCO_JSON="${COCO_JSON:-/root/autodl-tmp/dataset/SoyCotton/annotations/coco.json}"
WORKDIR_ROOT="${WORKDIR_ROOT:-/root/autodl-tmp/CODE/DeepLabV3Plus-Pytorch-master/workdirs}"

GPU_ID="${GPU_ID:-0}"
MODEL="${MODEL:-deeplabv3plus_mobilenet}"
OUTPUT_STRIDE="${OUTPUT_STRIDE:-16}"
LR="${LR:-0.01}"
TOTAL_ITRS="${TOTAL_ITRS:-640}"
BATCH_SIZE="${BATCH_SIZE:-8}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
CROP_SIZE="${CROP_SIZE:-512}"
VAL_INTERVAL="${VAL_INTERVAL:-200}"
VAL_RATIO="${VAL_RATIO:-0.1}"
SEED="${SEED:-42}"

EXP_NAME="${EXP_NAME:-soycotton_${MODEL}_itrs${TOTAL_ITRS}_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${WORKDIR_ROOT}/${EXP_NAME}"
LOG_FILE="${RUN_DIR}/train.log"

mkdir -p "${RUN_DIR}" "${WORKDIR_ROOT}"
cd "${PROJECT_DIR}"

python prepare_soycotton_from_coco.py \
  --coco_json "${COCO_JSON}" \
  --images_dir "${RAW_IMAGES_DIR}" \
  --output_root "${DATASET_ROOT}" \
  --val_ratio "${VAL_RATIO}" \
  --seed "${SEED}"

nohup python main.py \
  --model "${MODEL}" \
  --dataset cottonweed \
  --data_root "${DATASET_ROOT}" \
  --cottonweed_mask_dir masks_trainid \
  --class_names "${DATASET_ROOT}/classes.txt" \
  --work_dir "${WORKDIR_ROOT}" \
  --exp_name "${EXP_NAME}" \
  --gpu_id "${GPU_ID}" \
  --output_stride "${OUTPUT_STRIDE}" \
  --lr "${LR}" \
  --total_itrs "${TOTAL_ITRS}" \
  --batch_size "${BATCH_SIZE}" \
  --val_batch_size "${VAL_BATCH_SIZE}" \
  --crop_size "${CROP_SIZE}" \
  --val_interval "${VAL_INTERVAL}" \
  > "${LOG_FILE}" 2>&1 &

echo "Started training with PID $!"
echo "Run dir: ${RUN_DIR}"
echo "Log file: ${LOG_FILE}"
