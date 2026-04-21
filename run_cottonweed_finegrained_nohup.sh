#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${PROJECT_DIR}/workdirs"

DATA_ROOT="${1:-/root/autodl-tmp/dataset/cottonweed}"
GPU_ID="${2:-0}"
MODEL="${3:-deeplabv3plus_mobilenet}"
TOTAL_ITRS="${4:-640}"
VAL_INTERVAL="${5:-100}"
BATCH_SIZE="${6:-8}"
VAL_BATCH_SIZE="${7:-4}"
OUTPUT_STRIDE="${8:-16}"
LR="${9:-0.01}"
CROP_SIZE="${10:-512}"

mkdir -p "${WORK_DIR}"
cd "${PROJECT_DIR}"

timestamp="$(date '+%Y%m%d_%H%M%S')"
MASTER_LOG="${WORK_DIR}/nohup_finegrained_ablation_${timestamp}.log"

echo "[$(date '+%F %T')] Fine-grained ablation starts" | tee -a "${MASTER_LOG}"
echo "Project=${PROJECT_DIR}" | tee -a "${MASTER_LOG}"
echo "DataRoot=${DATA_ROOT}" | tee -a "${MASTER_LOG}"
echo "Model=${MODEL}, TotalItrs=${TOTAL_ITRS}, ValInterval=${VAL_INTERVAL}" | tee -a "${MASTER_LOG}"

RUN_NAMES=(
  "baseline"
  "fg_fusion"
  "improved_aspp_dense"
  "texture_enhance"
  "full_model"
)

RUN_ARGS=(
  "--aspp_variant standard"
  "--aspp_variant standard --enable_fg_fusion"
  "--aspp_variant dense --dense_aspp_rates 1,3,6,12,18"
  "--aspp_variant standard --enable_texture_enhance"
  "--aspp_variant deform --enable_fg_fusion --enable_texture_enhance --enable_decoder_detail"
)

for idx in "${!RUN_NAMES[@]}"; do
  run_name="${RUN_NAMES[$idx]}"
  extra_args="${RUN_ARGS[$idx]}"
  exp_name="cottonweed_${MODEL}_${run_name}_itr${TOTAL_ITRS}_${timestamp}"
  run_log="${WORK_DIR}/train_${exp_name}.log"

  echo "[$(date '+%F %T')] START ${run_name}" | tee -a "${MASTER_LOG}"
  echo "[$(date '+%F %T')] EXP=${exp_name}" | tee -a "${MASTER_LOG}"
  echo "[$(date '+%F %T')] LOG=${run_log}" | tee -a "${MASTER_LOG}"
  echo "[$(date '+%F %T')] ARGS=${extra_args}" | tee -a "${MASTER_LOG}"

  nohup conda run --no-capture-output -n Depplab python -u main.py \
    --dataset cottonweed \
    --data_root "${DATA_ROOT}" \
    --model "${MODEL}" \
    --gpu_id "${GPU_ID}" \
    --output_stride "${OUTPUT_STRIDE}" \
    --lr "${LR}" \
    --total_itrs "${TOTAL_ITRS}" \
    --val_interval "${VAL_INTERVAL}" \
    --batch_size "${BATCH_SIZE}" \
    --val_batch_size "${VAL_BATCH_SIZE}" \
    --crop_size "${CROP_SIZE}" \
    --work_dir "${WORK_DIR}" \
    --exp_name "${exp_name}" \
    ${extra_args} > "${run_log}" 2>&1 &

  pid=$!
  echo "[$(date '+%F %T')] PID=${pid}" | tee -a "${MASTER_LOG}"
  wait "${pid}"
  echo "[$(date '+%F %T')] FINISH ${run_name}" | tee -a "${MASTER_LOG}"
done

python "${PROJECT_DIR}/tools/build_finegrained_report.py" \
  --work_dir "${WORK_DIR}" \
  --timestamp "${timestamp}" \
  --output "${WORK_DIR}/finegrained_report_${timestamp}.md" | tee -a "${MASTER_LOG}"

echo "[$(date '+%F %T')] All runs finished" | tee -a "${MASTER_LOG}"
echo "Master log: ${MASTER_LOG}"
