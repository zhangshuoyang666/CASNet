#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/root/autodl-tmp/CODE/DeepLabV3Plus-Pytorch-master/DeepLabV3Plus-Pytorch-masterPLUS"
EXP_ROOT="${PROJECT_DIR}/1cottonweed-strong"
WORK_DIR="${EXP_ROOT}/workdirs"
EXP_NAME="exp_spatial_cbam_saff_boundaryaux_cottonweed_strong_itr10000"
DATA_ROOT="/root/autodl-tmp/dataset/cottonweed_strong"
GPU_ID="${1:-0}"

# 参考实验：
# - workdirs/train_exp_spatial_cbam_saff_boundaryaux_continue50_es5_step.log
# - workdirs/train_exp_spatial_cbam_saff_boundaryaux_itr1600.log
# 关键设置保持一致：spatial_cbam + SAFF + boundary_aux + bce edge + lambda_edge=0.1 + boundary_width=5
# 本次变更：total_itrs=10000，数据集切换到 cottonweed_strong

TOTAL_ITRS=10000
VAL_INTERVAL=32
BATCH_SIZE=8
VAL_BATCH_SIZE=2
OUTPUT_STRIDE=16
LR=0.01
LR_POLICY=step
STEP_SIZE=1000
WEIGHT_DECAY=0.0005
RANDOM_SEED=1
PRINT_INTERVAL=10

mkdir -p "${EXP_ROOT}" "${WORK_DIR}"
cd "${PROJECT_DIR}"

CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[ERROR] conda init script not found: ${CONDA_SH}" >&2
  exit 1
fi

LOG_FILE="${EXP_ROOT}/train.log"
CMD_FILE="${EXP_ROOT}/launch_command.txt"

cat > "${CMD_FILE}" <<EOF
python -u main.py \\
  --dataset cottonweed \\
  --data_root ${DATA_ROOT} \\
  --cottonweed_mask_dir masks \\
  --model deeplabv3plus_mobilenet \\
  --attention_type spatial_cbam \\
  --enable_boundary_aux \\
  --use_saff \\
  --edge_loss_type bce \\
  --lambda_edge 0.1 \\
  --boundary_width 5 \\
  --gpu_id ${GPU_ID} \\
  --output_stride ${OUTPUT_STRIDE} \\
  --total_itrs ${TOTAL_ITRS} \\
  --val_interval ${VAL_INTERVAL} \\
  --batch_size ${BATCH_SIZE} \\
  --val_batch_size ${VAL_BATCH_SIZE} \\
  --lr ${LR} \\
  --lr_policy ${LR_POLICY} \\
  --step_size ${STEP_SIZE} \\
  --weight_decay ${WEIGHT_DECAY} \\
  --random_seed ${RANDOM_SEED} \\
  --print_interval ${PRINT_INTERVAL} \\
  --work_dir ${WORK_DIR} \\
  --exp_name ${EXP_NAME}
EOF

nohup bash -lc "
  source \"${CONDA_SH}\"
  conda activate Depplab
  cd \"${PROJECT_DIR}\"
  python -u main.py \\
    --dataset cottonweed \\
    --data_root \"${DATA_ROOT}\" \\
    --cottonweed_mask_dir masks \\
    --model deeplabv3plus_mobilenet \\
    --attention_type spatial_cbam \\
    --enable_boundary_aux \\
    --use_saff \\
    --edge_loss_type bce \\
    --lambda_edge 0.1 \\
    --boundary_width 5 \\
    --gpu_id \"${GPU_ID}\" \\
    --output_stride \"${OUTPUT_STRIDE}\" \\
    --total_itrs \"${TOTAL_ITRS}\" \\
    --val_interval \"${VAL_INTERVAL}\" \\
    --batch_size \"${BATCH_SIZE}\" \\
    --val_batch_size \"${VAL_BATCH_SIZE}\" \\
    --lr \"${LR}\" \\
    --lr_policy \"${LR_POLICY}\" \\
    --step_size \"${STEP_SIZE}\" \\
    --weight_decay \"${WEIGHT_DECAY}\" \\
    --random_seed \"${RANDOM_SEED}\" \\
    --print_interval \"${PRINT_INTERVAL}\" \\
    --work_dir \"${WORK_DIR}\" \\
    --exp_name \"${EXP_NAME}\"
" > "${LOG_FILE}" 2>&1 &

echo "Started training"
echo "EXP_ROOT: ${EXP_ROOT}"
echo "RUN_DIR: ${WORK_DIR}/${EXP_NAME}"
echo "LOG: ${LOG_FILE}"
echo "PID: $!"
