#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BEST_DIR="${PROJECT_DIR}/BEST-Last"
POST_SCRIPT="${BEST_DIR}/postprocess_mainlineA.py"

DATA_ROOT="${1:-/root/autodl-tmp/dataset/cottonweed}"
GPU_ID="${2:-0}"
BATCH_SIZE="${3:-8}"
BASE_BATCH="${4:-16}"
BASE_LR="${5:-0.01}"
EARLY_STOP_PATIENCE="${6:-8}"
EARLY_STOP_MIN_DELTA="${7:-0.0003}"

# Linear LR scaling with batch size: lr = base_lr * batch/base_batch
LR="$(python - <<PY
base_lr=${BASE_LR}
batch=${BATCH_SIZE}
base_batch=${BASE_BATCH}
print(f"{base_lr * batch / base_batch:.6f}")
PY
)"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LR_TAG="${LR//./p}"
DELTA_TAG="${EARLY_STOP_MIN_DELTA//./p}"
EXP_NAME="mainlineA_bs${BATCH_SIZE}_lr${LR_TAG}_es${EARLY_STOP_PATIENCE}_md${DELTA_TAG}_${TIMESTAMP}"
RUN_DIR="${BEST_DIR}/${EXP_NAME}"

mkdir -p "${RUN_DIR}"

cat > "${RUN_DIR}/config_snapshot.json" <<EOF
{
  "exp_name": "${EXP_NAME}",
  "profile": "mainline_A_bs8_adaptive",
  "dataset": "cottonweed",
  "data_root": "${DATA_ROOT}",
  "cottonweed_mask_dir": "masks",
  "model": "deeplabv3plus_mobilenet",
  "attention_type": "spatial_cbam",
  "enable_boundary_aux": true,
  "use_saff": true,
  "saff_f1_source": "mid",
  "saff_f2_source": "aspp",
  "edge_loss_type": "bce",
  "lambda_edge": 0.1,
  "boundary_width": 5,
  "output_stride": 16,
  "crop_size": 513,
  "batch_size": ${BATCH_SIZE},
  "val_batch_size": 4,
  "lr": ${LR},
  "lr_scaling_rule": "linear_from_bs16_lr0.01",
  "weight_decay": 1e-4,
  "dense_aspp_rates": "1,2,4,8,16",
  "warmup_iters": 0,
  "aug_vflip": false,
  "aug_rotation": 0,
  "total_itrs": -1,
  "val_interval": 50,
  "print_interval": 10,
  "enable_early_stop": true,
  "early_stop_metric": "Mean IoU",
  "early_stop_patience": ${EARLY_STOP_PATIENCE},
  "early_stop_min_delta": ${EARLY_STOP_MIN_DELTA},
  "min_itrs_before_early_stop": 1000
}
EOF

cat > "${RUN_DIR}/train_and_post.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source "/root/miniconda3/etc/profile.d/conda.sh"
conda activate Depplab
cd "${PROJECT_DIR}"

python -u main.py \\
  --dataset cottonweed \\
  --data_root "${DATA_ROOT}" \\
  --cottonweed_mask_dir masks \\
  --model deeplabv3plus_mobilenet \\
  --attention_type spatial_cbam \\
  --enable_boundary_aux \\
  --use_saff \\
  --saff_f1_source mid \\
  --saff_f2_source aspp \\
  --edge_loss_type bce \\
  --lambda_edge 0.1 \\
  --boundary_width 5 \\
  --gpu_id "${GPU_ID}" \\
  --output_stride 16 \\
  --crop_size 513 \\
  --batch_size "${BATCH_SIZE}" \\
  --val_batch_size 4 \\
  --random_seed 1 \\
  --lr "${LR}" \\
  --weight_decay 1e-4 \\
  --dense_aspp_rates 1,2,4,8,16 \\
  --warmup_iters 0 \\
  --total_itrs -1 \\
  --val_interval 50 \\
  --print_interval 10 \\
  --work_dir "${BEST_DIR}" \\
  --exp_name "${EXP_NAME}" \\
  --enable_early_stop \\
  --early_stop_metric "Mean IoU" \\
  --early_stop_patience "${EARLY_STOP_PATIENCE}" \\
  --early_stop_min_delta "${EARLY_STOP_MIN_DELTA}" \\
  --min_itrs_before_early_stop 1000 \\
  > "${RUN_DIR}/train.log" 2>&1

python "${POST_SCRIPT}" --run_dir "${RUN_DIR}" >> "${RUN_DIR}/train.log" 2>&1
EOF

chmod +x "${RUN_DIR}/train_and_post.sh"
nohup "${RUN_DIR}/train_and_post.sh" > "${RUN_DIR}/nohup.out" 2>&1 &
PID=$!

cat > "${RUN_DIR}/status.txt" <<EOF
started: $(date '+%F %T')
pid: ${PID}
run_dir: ${RUN_DIR}
data_root: ${DATA_ROOT}
gpu_id: ${GPU_ID}
batch_size: ${BATCH_SIZE}
lr: ${LR}
command: nohup ${RUN_DIR}/train_and_post.sh > ${RUN_DIR}/nohup.out 2>&1 &
EOF

cat > "${BEST_DIR}/latest_bs8_run.txt" <<EOF
exp_name=${EXP_NAME}
run_dir=${RUN_DIR}
pid=${PID}
batch_size=${BATCH_SIZE}
lr=${LR}
started=$(date '+%F %T')
EOF

echo "Started Mainline-A bs=${BATCH_SIZE} training with nohup."
echo "PID: ${PID}"
echo "Run dir: ${RUN_DIR}"
echo "Train log: ${RUN_DIR}/train.log"
echo "Nohup: ${RUN_DIR}/nohup.out"
