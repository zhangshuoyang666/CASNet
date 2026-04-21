#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${PROJECT_DIR}/workdirs"
CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-Depplab}"

# Queue mode:
#   WAIT_PID > 0: wait this pid to exit
#   WAIT_PID = 0: start immediately
WAIT_PID="${1:-0}"

DATA_ROOT="${2:-/root/autodl-tmp/dataset/cottonweed}"
GPU_ID="${3:-0}"
MODEL="${4:-deeplabv3plus_mobilenet}"
TOTAL_ITRS="${5:-1600}"
VAL_INTERVAL="${6:-32}"
BATCH_SIZE="${7:-16}"
VAL_BATCH_SIZE="${8:-4}"
OUTPUT_STRIDE="${9:-16}"
LR="${10:-0.01}"
WEIGHT_DECAY="${11:-1e-4}"
LAMBDA_EDGE="${12:-0.1}"
BOUNDARY_WIDTH="${13:-5}"

# "fg" -> --enable_fg_fusion
# "saff" -> --use_saff --saff_f1_source mid --saff_f2_source aspp
FUSION_TYPE="${14:-fg}"
DATASET="${15:-cottonweed}"
MASK_DIR="${16:-}"
CLASS_NAMES_FILE="${17:-}"
NUM_CLASSES="${18:-}"
WAIT_KEYWORD="${19:-}"
MAX_RETRIES="${20:-2}"
RETRY_SLEEP_SEC="${21:-30}"

mkdir -p "${WORK_DIR}"
cd "${PROJECT_DIR}"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[ERROR] conda init script not found: ${CONDA_SH}"
  exit 1
fi

TS="$(date '+%Y%m%d_%H%M%S')"
QUEUE_LOG="${WORK_DIR}/queue_ablation_abc_${TS}.log"
REPORT_CSV="${WORK_DIR}/ablation_abc_${TS}.csv"
REPORT_MD="${WORK_DIR}/ablation_abc_${TS}.md"

nohup bash -lc "
  set -euo pipefail
  source \"${CONDA_SH}\"
  conda activate \"${CONDA_ENV}\"
  cd \"${PROJECT_DIR}\"

  if [[ \"${WAIT_PID}\" =~ ^[0-9]+$ ]] && [[ \"${WAIT_PID}\" -gt 0 ]]; then
    echo \"[Queue] waiting pid ${WAIT_PID} ...\"
    while kill -0 \"${WAIT_PID}\" 2>/dev/null; do
      sleep 60
    done
    echo \"[Queue] wait pid finished at \$(date '+%F %T')\"
  elif [[ -n \"${WAIT_KEYWORD}\" ]]; then
    echo \"[Queue] waiting keyword ${WAIT_KEYWORD} ...\"
    while WAIT_KEYWORD="${WAIT_KEYWORD}" python - <<'PY'
import os
import subprocess
kw = os.environ['WAIT_KEYWORD']
out = subprocess.check_output(['ps', '-eo', 'cmd'], text=True, errors='ignore')
for line in out.splitlines():
    if kw in line and 'run_ablation_abc_queue_nohup.sh' not in line:
        raise SystemExit(0)
raise SystemExit(1)
PY
    do
      sleep 60
    done
    echo \"[Queue] wait keyword cleared at \$(date '+%F %T')\"
  else
    echo \"[Queue] WAIT_PID=${WAIT_PID}, start immediately at \$(date '+%F %T')\"
  fi

  echo \"exp_name,best_miou,best_iter,best_epoch\" > \"${REPORT_CSV}\"

  run_one () {
    local exp_name=\"\$1\"
    local extra_args=\"\$2\"
    local log_file=\"${WORK_DIR}/train_\${exp_name}.log\"
    local attempt=1

    while true; do
      echo \"[Queue] start \${exp_name} (attempt=\${attempt}/${MAX_RETRIES}) at \$(date '+%F %T')\"
      echo \"[Queue] args: \${extra_args}\"
      DATASET_EXTRA_ARGS=\"\"
      if [[ -n \"${MASK_DIR}\" ]]; then
        DATASET_EXTRA_ARGS+=\" --cottonweed_mask_dir ${MASK_DIR}\"
      fi
      if [[ -n \"${CLASS_NAMES_FILE}\" ]]; then
        DATASET_EXTRA_ARGS+=\" --class_names ${CLASS_NAMES_FILE}\"
      fi
      if [[ -n \"${NUM_CLASSES}\" ]]; then
        DATASET_EXTRA_ARGS+=\" --num_classes ${NUM_CLASSES}\"
      fi

      run_ok=1
      if ! python -u main.py \
        --dataset \"${DATASET}\" \
        --data_root \"${DATA_ROOT}\" \
        --model \"${MODEL}\" \
        --attention_type spatial_cbam \
        --gpu_id \"${GPU_ID}\" \
        --output_stride \"${OUTPUT_STRIDE}\" \
        --lr \"${LR}\" \
        --weight_decay \"${WEIGHT_DECAY}\" \
        --total_itrs \"${TOTAL_ITRS}\" \
        --val_interval \"${VAL_INTERVAL}\" \
        --batch_size \"${BATCH_SIZE}\" \
        --val_batch_size \"${VAL_BATCH_SIZE}\" \
        --work_dir \"${WORK_DIR}\" \
        --exp_name \"\${exp_name}\" \
        \${DATASET_EXTRA_ARGS} \
        \${extra_args} > \"\${log_file}\" 2>&1; then
        run_ok=0
      fi

      if [[ \"\${run_ok}\" -eq 1 ]]; then
        if METRICS_PATH="${WORK_DIR}/\${exp_name}/metrics.tsv" REPORT_CSV="${REPORT_CSV}" EXP_NAME="\${exp_name}" python - <<'PY'
import csv
import pathlib
import os

metrics_path = pathlib.Path(os.environ['METRICS_PATH'])
csv_path = pathlib.Path(os.environ['REPORT_CSV'])
exp_name = os.environ['EXP_NAME']

best = None
with metrics_path.open('r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\\t')
    for row in reader:
        val = str(row.get('mIoU', '')).strip()
        if not val:
            continue
        cur = {
            'mIoU': float(val),
            'Iter': int(float(row.get('Iter', 0) or 0)),
            'epoch': int(float(row.get('epoch', 0) or 0)),
        }
        if best is None or cur['mIoU'] > best['mIoU']:
            best = cur

if best is None:
    raise SystemExit(f'No valid mIoU rows found: {metrics_path}')

with csv_path.open('a', encoding='utf-8') as f:
    f.write(f\"{exp_name},{best['mIoU']:.6f},{best['Iter']},{best['epoch']}\\n\")
PY
        then
          echo \"[Queue] done  \${exp_name} at \$(date '+%F %T')\"
          break
        else
          echo \"[Queue][WARN] metrics parse failed for \${exp_name}, attempt=\${attempt}\" | tee -a \"\${log_file}\"
        fi
      else
        echo \"[Queue][WARN] train failed for \${exp_name}, attempt=\${attempt}\" | tee -a \"\${log_file}\"
      fi

      if [[ \"\${attempt}\" -ge \"${MAX_RETRIES}\" ]]; then
        echo \"[Queue][ERROR] \${exp_name} failed after ${MAX_RETRIES} attempts\"
        return 1
      fi
      attempt=\$((attempt + 1))
      echo \"[Queue] retry \${exp_name} after ${RETRY_SLEEP_SEC}s ...\"
      sleep \"${RETRY_SLEEP_SEC}\"
    done
  }

  FUSION_ARGS=\"\"
  if [[ \"${FUSION_TYPE}\" == \"fg\" ]]; then
    FUSION_ARGS=\"--enable_fg_fusion\"
  elif [[ \"${FUSION_TYPE}\" == \"saff\" ]]; then
    FUSION_ARGS=\"--use_saff --saff_f1_source mid --saff_f2_source aspp\"
  else
    echo \"[ERROR] unsupported FUSION_TYPE=${FUSION_TYPE}, expected fg|saff\"
    exit 2
  fi

  # 1) A
  EXP_A=\"ablation_A_spatial_cbam_${TS}\"
  run_one \"\${EXP_A}\" \"\"

  # 2) A+B
  EXP_AB=\"ablation_AB_spatial_cbam_fusion_${TS}\"
  run_one \"\${EXP_AB}\" \"\${FUSION_ARGS}\"

  # 3) A+B+C
  EXP_ABC=\"ablation_ABC_spatial_cbam_fusion_boundary_${TS}\"
  run_one \"\${EXP_ABC}\" \"\${FUSION_ARGS} --enable_boundary_aux --edge_loss_type bce --lambda_edge ${LAMBDA_EDGE} --boundary_width ${BOUNDARY_WIDTH}\"

  REPORT_CSV="${REPORT_CSV}" REPORT_MD="${REPORT_MD}" TS="${TS}" FUSION_TYPE="${FUSION_TYPE}" python - <<'PY'
import csv
import pathlib
import os

csv_path = pathlib.Path(os.environ['REPORT_CSV'])
md_path = pathlib.Path(os.environ['REPORT_MD'])
rows = []
with csv_path.open('r', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        row['best_miou'] = float(row['best_miou'])
        row['best_iter'] = int(row['best_iter'])
        row['best_epoch'] = int(row['best_epoch'])
        rows.append(row)

rows.sort(key=lambda x: x['best_miou'], reverse=True)
best = rows[0] if rows else None

lines = [
    '# Ablation Report: A / A+B / A+B+C',
    '',
    f'- generated_at: {os.environ.get(\"TS\", \"\")}',
    f'- fusion_type: {os.environ.get(\"FUSION_TYPE\", \"\")}',
    '',
    '| exp_name | best_mIoU | best_iter | best_epoch |',
    '|---|---:|---:|---:|',
]
for r in rows:
    lines.append(f\"| {r['exp_name']} | {r['best_miou']:.6f} | {r['best_iter']} | {r['best_epoch']} |\")

if best is not None:
    lines += ['', f\"**Best:** {best['exp_name']} (mIoU={best['best_miou']:.6f})\"]

md_path.write_text('\\n'.join(lines) + '\\n', encoding='utf-8')
PY

  echo \"[Queue] all ablation runs finished at \$(date '+%F %T')\"
  echo \"[Queue] report csv: ${REPORT_CSV}\"
  echo \"[Queue] report md : ${REPORT_MD}\"
" > "${QUEUE_LOG}" 2>&1 &

QUEUE_PID=$!

echo "Queued ablation runs: A -> A+B -> A+B+C"
echo "  queue_pid: ${QUEUE_PID}"
echo "  queue_log: ${QUEUE_LOG}"
echo "  report_csv: ${REPORT_CSV}"
echo "  report_md: ${REPORT_MD}"
