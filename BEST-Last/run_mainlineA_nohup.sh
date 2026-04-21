#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BEST_DIR="${PROJECT_DIR}/BEST-Last"
POST_SCRIPT="${BEST_DIR}/postprocess_mainlineA.py"

DATA_ROOT="${1:-/root/autodl-tmp/dataset/cottonweedV2}"
GPU_ID="${2:-0}"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
EXP_NAME="mainlineA_es_${TIMESTAMP}"
RUN_DIR="${BEST_DIR}/${EXP_NAME}"

mkdir -p "${RUN_DIR}"

cat > "${RUN_DIR}/config_snapshot.json" <<EOF
{
  "exp_name": "${EXP_NAME}",
  "profile": "mainline_A",
  "dataset": "cottonweedV2",
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
  "batch_size": 32,
  "val_batch_size": 4,
  "lr": 0.02,
  "weight_decay": 1e-4,
  "dense_aspp_rates": "1,3,6,12,18",
  "warmup_iters": 0,
  "aug_vflip": true,
  "aug_rotation": 0,
  "total_itrs": -1,
  "val_interval": 50,
  "print_interval": 10,
  "enable_early_stop": true,
  "early_stop_metric": "Mean IoU",
  "early_stop_patience": 10,
  "early_stop_min_delta": 0.0005,
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
  --batch_size 32 \\
  --val_batch_size 4 \\
  --random_seed 1 \\
  --lr 0.02 \\
  --weight_decay 1e-4 \\
  --dense_aspp_rates 1,3,6,12,18 \\
  --total_itrs -1 \\
  --val_interval 50 \\
  --print_interval 10 \\
  --work_dir "${BEST_DIR}" \\
  --exp_name "${EXP_NAME}" \\
  --enable_early_stop \\
  --early_stop_metric "Mean IoU" \\
  --early_stop_patience 10 \\
  --early_stop_min_delta 0.0005 \\
  --min_itrs_before_early_stop 1000 \\
  --aug_vflip \\
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
command: nohup ${RUN_DIR}/train_and_post.sh > ${RUN_DIR}/nohup.out 2>&1 &
EOF

echo "Started Mainline-A training with nohup."
echo "PID: ${PID}"
echo "Run dir: ${RUN_DIR}"
echo "Log: ${RUN_DIR}/train.log"
echo "Nohup: ${RUN_DIR}/nohup.out"
