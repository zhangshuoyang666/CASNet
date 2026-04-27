#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/root/autodl-tmp/CODE/DeepLabV3Plus-Pytorch-master/DeepLabV3Plus-Pytorch-masterPLUS"
EXP_ROOT="${PROJECT_DIR}/5cottonweedV4（cotton-abuth-others）"
WORK_DIR="${EXP_ROOT}/workdirs"
EXP_NAME="exp_spatial_cbam_saff_boundaryaux_cottonweedV4_4class_itr3000_lr0p02_bs32"
DATA_ROOT="/root/autodl-tmp/dataset/cottonweedV4_train1k_4class"
CLASS_NAMES_FILE="${EXP_ROOT}/class_names_4class.txt"
GPU_ID="${1:-0}"

TOTAL_ITRS=3000
VAL_INTERVAL=32
BATCH_SIZE=32
VAL_BATCH_SIZE=4
OUTPUT_STRIDE=16
LR=0.02
LR_POLICY=step
STEP_SIZE=1000
WEIGHT_DECAY=0.0005
RANDOM_SEED=1
PRINT_INTERVAL=10

mkdir -p "${EXP_ROOT}" "${WORK_DIR}" "${EXP_ROOT}/logs" "${EXP_ROOT}/results"
cd "${PROJECT_DIR}"

CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[ERROR] conda init script not found: ${CONDA_SH}" >&2
  exit 1
fi

LOG_FILE="${EXP_ROOT}/train.log"
CMD_FILE="${EXP_ROOT}/launch_command.txt"
CFG_FILE="${EXP_ROOT}/config_exp_spatial_cbam_saff_boundaryaux_itr3000_lr0p02_bs32_cottonweedV4_4class.txt"
DATASET_CFG_FILE="${EXP_ROOT}/dataset_config_auto_generated.txt"

cat > "${DATASET_CFG_FILE}" <<EOF
[dataset]
name=cottonweedV4_train1k_4class
root=${DATA_ROOT}
mask_dir=masks_4class
num_classes=4
class_names_file=${CLASS_NAMES_FILE}
class_0=background
class_1=cotton
class_2=abuth
class_3=others
EOF

cat > "${CMD_FILE}" <<EOF
python -u main.py \\
  --dataset cottonweed \\
  --data_root ${DATA_ROOT} \\
  --cottonweed_mask_dir masks_4class \\
  --num_classes 4 \\
  --class_names ${CLASS_NAMES_FILE} \\
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

cat > "${CFG_FILE}" <<EOF
python main.py \\
  --dataset cottonweed \\
  --data_root ${DATA_ROOT} \\
  --cottonweed_mask_dir masks_4class \\
  --num_classes 4 \\
  --class_names ${CLASS_NAMES_FILE} \\
  --model deeplabv3plus_mobilenet \\
  --output_stride 16 \\
  --attention_type spatial_cbam \\
  --use_saff \\
  --enable_boundary_aux \\
  --edge_loss_type bce \\
  --boundary_width 5 \\
  --lambda_edge 0.1 \\
  --lr 0.02 \\
  --lr_policy step \\
  --step_size 1000 \\
  --batch_size 32 \\
  --val_batch_size 4 \\
  --crop_size 513 \\
  --total_itrs 3000 \\
  --val_interval 32 \\
  --print_interval 10 \\
  --gpu_id ${GPU_ID} \\
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
    --cottonweed_mask_dir masks_4class \\
    --num_classes 4 \\
    --class_names \"${CLASS_NAMES_FILE}\" \\
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
echo $! > "${EXP_ROOT}/logs/train_${EXP_NAME}.pid"
