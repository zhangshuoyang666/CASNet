#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${PROJECT_DIR}/workdirs"
CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"

# Wait current experiment (default: running mid+aspp SAFF job) then continue.
WAIT_PID="${1:-308036}"
DATA_ROOT="${2:-/root/autodl-tmp/dataset/cottonweed}"
GPU_ID="${3:-0}"
TOTAL_ITRS="${4:-1600}"
VAL_INTERVAL="${5:-32}"
BATCH_SIZE="${6:-16}"
VAL_BATCH_SIZE="${7:-4}"
OUTPUT_STRIDE="${8:-16}"

mkdir -p "${WORK_DIR}"
cd "${PROJECT_DIR}"

nohup bash -lc "
  set -euo pipefail
  source \"${CONDA_SH}\"
  conda activate Depplab
  cd \"${PROJECT_DIR}\"

  echo \"[Queue] waiting pid ${WAIT_PID} ...\"
  while kill -0 \"${WAIT_PID}\" 2>/dev/null; do sleep 60; done
  echo \"[Queue] wait pid finished at \$(date '+%F %T')\"

  run_one () {
    local exp_name=\"\$1\"
    local f1=\"\$2\"
    local f2=\"\$3\"
    local log_file=\"${WORK_DIR}/train_\${exp_name}.log\"
    echo \"[Queue] start \${exp_name} (F1=\${f1}, F2=\${f2}) at \$(date '+%F %T')\"
    python -u main.py \
      --dataset cottonweed \
      --data_root \"${DATA_ROOT}\" \
      --model deeplabv3plus_mobilenet \
      --attention_type spatial_cbam \
      --enable_boundary_aux \
      --use_saff \
      --saff_f1_source \"\${f1}\" \
      --saff_f2_source \"\${f2}\" \
      --edge_loss_type bce \
      --lambda_edge 0.1 \
      --boundary_width 5 \
      --gpu_id \"${GPU_ID}\" \
      --output_stride \"${OUTPUT_STRIDE}\" \
      --total_itrs \"${TOTAL_ITRS}\" \
      --val_interval \"${VAL_INTERVAL}\" \
      --batch_size \"${BATCH_SIZE}\" \
      --val_batch_size \"${VAL_BATCH_SIZE}\" \
      --work_dir \"${WORK_DIR}\" \
      --exp_name \"\${exp_name}\" > \"\${log_file}\" 2>&1
    echo \"[Queue] done  \${exp_name} at \$(date '+%F %T')\"
  }

  # High-gain-first candidates:
  # 1) F1=low, F2=aspp: stronger shallow detail with semantic ASPP branch.
  # 2) F1=mid, F2=high: test whether pre-ASPP semantics preserve class boundaries.
  # 3) F1=low, F2=high: stronger detail + raw semantics (aggressive fusion).
  run_one exp_spatial_cbam_saff_low_aspp_boundaryaux_itr1600 low aspp
  run_one exp_spatial_cbam_saff_mid_high_boundaryaux_itr1600 mid high
  run_one exp_spatial_cbam_saff_low_high_boundaryaux_itr1600 low high

  python tools/summarize_saff_f1f2_results.py \
    --work_dir \"${WORK_DIR}\" \
    --experiments exp_spatial_cbam_boundaryaux_best_itr1600,exp_spatial_cbam_saff_boundaryaux_itr1600,exp_spatial_cbam_saff_low_aspp_boundaryaux_itr1600,exp_spatial_cbam_saff_mid_high_boundaryaux_itr1600,exp_spatial_cbam_saff_low_high_boundaryaux_itr1600 \
    --output \"${WORK_DIR}/saff_f1f2_highgain_report.md\"

  echo \"[Queue] all queued experiments finished at \$(date '+%F %T')\"
" > "${WORK_DIR}/queue_saff_f1f2_highgain.log" 2>&1 &

echo "Queued SAFF F1/F2 experiments."
echo "  queue_log: ${WORK_DIR}/queue_saff_f1f2_highgain.log"
echo "  queue_pid: $!"
