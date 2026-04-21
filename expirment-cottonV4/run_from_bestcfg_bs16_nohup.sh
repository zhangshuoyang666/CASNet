#!/usr/bin/env bash
set -euo pipefail

CONDA_SH="/root/miniconda3/etc/profile.d/conda.sh"
source "$CONDA_SH"
conda activate Depplab

PROJECT="/root/autodl-tmp/CODE/DeepLabV3Plus-Pytorch-master/DeepLabV3Plus-Pytorch-master"
EXP_BASE="$PROJECT/expirment-cottonV4"
BEST_CFG="$PROJECT/BEST-Last/bestcfg_nonbaseline_bs8_lr0p005000_es5_md0p0001_20260419_120030/config_snapshot.json"
DATA_ROOT="/root/autodl-tmp/dataset/cottonweedV4_train1k"
MASK_DIR="masks"
GPU_ID="${1:-0}"

mkdir -p "$EXP_BASE"
TS="$(date '+%Y%m%d_%H%M%S')"

BS=16
VAL_BS=1
BASE_BS=8
BASE_LR=$(python - <<PY
import json
cfg=json.load(open("$BEST_CFG","r",encoding="utf-8"))
print(cfg["lr"])
PY
)
LR=$(python - <<PY
print(f"{float('$BASE_LR') * float('$BS') / float('$BASE_BS'):.6f}")
PY
)

WD=$(python - <<PY
import json
cfg=json.load(open("$BEST_CFG","r",encoding="utf-8"))
print(cfg["weight_decay"])
PY
)
LAMBDA_EDGE=$(python - <<PY
import json
cfg=json.load(open("$BEST_CFG","r",encoding="utf-8"))
print(cfg["lambda_edge"])
PY
)
BOUNDARY_WIDTH=$(python - <<PY
import json
cfg=json.load(open("$BEST_CFG","r",encoding="utf-8"))
print(cfg["boundary_width"])
PY
)

RUN_ROOT="$EXP_BASE/frombest_bs${BS}_lr${LR//./p}_wd${WD//./p}_le${LAMBDA_EDGE//./p}_${TS}"
mkdir -p "$RUN_ROOT"
echo "$RUN_ROOT" > "$EXP_BASE/latest_run_dir.txt"

CLASS_FILE="$RUN_ROOT/class_names.txt"
cat > "$CLASS_FILE" <<'EOF'
background
cotton
abuth
canger
machixian
longkui
tianxuanhua
niujincao
EOF

TRAIN_COUNT=$(python - <<PY
import os
root="$DATA_ROOT/train/images"
print(len([f for f in os.listdir(root) if f.lower().endswith((".jpg",".jpeg",".png"))]))
PY
)
STEPS_PER_EPOCH=$(( TRAIN_COUNT / BS ))
PRECHECK_ITRS=$(( STEPS_PER_EPOCH * 2 ))
VAL_INTERVAL=$(( STEPS_PER_EPOCH / 2 ))
if [ "$VAL_INTERVAL" -lt 50 ]; then
  VAL_INTERVAL=50
fi
TOTAL_ITRS=$(( STEPS_PER_EPOCH * 25 ))
EARLY_STOP_PATIENCE=8
EARLY_STOP_MIN_DELTA=0.0001
MIN_ITRS_BEFORE_ES=$(( STEPS_PER_EPOCH * 6 ))

cat > "$RUN_ROOT/config_snapshot.json" <<EOF
{
  "source_config": "$BEST_CFG",
  "dataset": "cottonweedV4_train1k",
  "data_root": "$DATA_ROOT",
  "cottonweed_mask_dir": "$MASK_DIR",
  "class_names_file": "$CLASS_FILE",
  "num_classes": 8,
  "batch_size": $BS,
  "val_batch_size": $VAL_BS,
  "base_batch_for_scaling": $BASE_BS,
  "base_lr_from_config": $BASE_LR,
  "scaled_lr": $LR,
  "weight_decay": "$WD",
  "lambda_edge": $LAMBDA_EDGE,
  "boundary_width": $BOUNDARY_WIDTH,
  "precheck_itrs": $PRECHECK_ITRS,
  "total_itrs": $TOTAL_ITRS,
  "val_interval": $VAL_INTERVAL,
  "early_stop_patience": $EARLY_STOP_PATIENCE,
  "early_stop_min_delta": $EARLY_STOP_MIN_DELTA,
  "min_itrs_before_early_stop": $MIN_ITRS_BEFORE_ES,
  "adaptive_rule_on_precheck_fail": "lr=lr*0.5, warmup+=200"
}
EOF

run_train() {
  local run_dir="$1"
  local exp_name="$2"
  local lr="$3"
  local warmup="$4"
  local total_itrs="$5"
  local patience="$6"

  local warmup_arg=""
  if [ "$warmup" -gt 0 ]; then
    warmup_arg="--warmup_iters $warmup"
  fi

  python -u "$PROJECT/main.py" \
    --dataset cottonweed \
    --data_root "$DATA_ROOT" \
    --cottonweed_mask_dir "$MASK_DIR" \
    --class_names "$CLASS_FILE" \
    --num_classes 8 \
    --model deeplabv3plus_mobilenet \
    --attention_type spatial_cbam \
    --enable_boundary_aux \
    --use_saff \
    --saff_f1_source mid \
    --saff_f2_source aspp \
    --edge_loss_type bce \
    --lambda_edge "$LAMBDA_EDGE" \
    --boundary_width "$BOUNDARY_WIDTH" \
    --gpu_id "$GPU_ID" \
    --output_stride 16 \
    --crop_size 513 \
    --batch_size "$BS" \
    --val_batch_size "$VAL_BS" \
    --random_seed 1 \
    --lr "$lr" \
    --weight_decay "$WD" \
    --aspp_variant standard \
    --dense_aspp_rates 1,3,6,12,18 \
    $warmup_arg \
    --total_itrs "$total_itrs" \
    --val_interval "$VAL_INTERVAL" \
    --print_interval 10 \
    --work_dir "$RUN_ROOT" \
    --exp_name "$exp_name" \
    --enable_early_stop \
    --early_stop_metric "Mean IoU" \
    --early_stop_patience "$patience" \
    --early_stop_min_delta "$EARLY_STOP_MIN_DELTA" \
    --min_itrs_before_early_stop "$MIN_ITRS_BEFORE_ES" \
    > "$run_dir/train.log" 2>&1
}

WARMUP=0
TRY_LR="$LR"
PRE_NAME="precheck_bs${BS}_lr${TRY_LR//./p}_${TS}"
PRE_DIR="$RUN_ROOT/$PRE_NAME"
mkdir -p "$PRE_DIR"
echo "[INFO] precheck start: $PRE_NAME"
run_train "$PRE_DIR" "$PRE_NAME" "$TRY_LR" "$WARMUP" "$PRECHECK_ITRS" 3

if ! python "$EXP_BASE/metrics_guard.py" --metrics_tsv "$PRE_DIR/metrics.tsv" --json_out "$PRE_DIR/health_check.json"; then
  echo "[WARN] precheck failed, applying adaptive adjustment."
  TRY_LR=$(python - <<PY
print(f"{float('$TRY_LR')*0.5:.6f}")
PY
)
  WARMUP=200
  PRE_NAME="precheck_retry_bs${BS}_lr${TRY_LR//./p}_wu${WARMUP}_${TS}"
  PRE_DIR="$RUN_ROOT/$PRE_NAME"
  mkdir -p "$PRE_DIR"
  run_train "$PRE_DIR" "$PRE_NAME" "$TRY_LR" "$WARMUP" "$PRECHECK_ITRS" 3
  if ! python "$EXP_BASE/metrics_guard.py" --metrics_tsv "$PRE_DIR/metrics.tsv" --json_out "$PRE_DIR/health_check.json"; then
    echo "[ERROR] precheck failed twice, stop this job."
    exit 2
  fi
fi

FULL_NAME="fulltrain_bs${BS}_lr${TRY_LR//./p}_wu${WARMUP}_es${EARLY_STOP_PATIENCE}_${TS}"
FULL_DIR="$RUN_ROOT/$FULL_NAME"
mkdir -p "$FULL_DIR"
echo "[INFO] full training start: $FULL_NAME"
run_train "$FULL_DIR" "$FULL_NAME" "$TRY_LR" "$WARMUP" "$TOTAL_ITRS" "$EARLY_STOP_PATIENCE"
python "$EXP_BASE/postprocess_cottonv4.py" --run_dir "$FULL_DIR" >> "$FULL_DIR/train.log" 2>&1

python - <<PY
import csv, json, os
run_root="$RUN_ROOT"
full_dir="$FULL_DIR"
best=None
with open(os.path.join(full_dir,"metrics.tsv"),encoding="utf-8") as f:
    for row in csv.DictReader(f,delimiter="\\t"):
        if not str(row.get("mIoU","")).strip():
            continue
        cur={
            "mIoU": float(row["mIoU"]),
            "aAcc": float(row["aAcc"]),
            "mAcc": float(row["mAcc"]),
            "iter": int(row["Iter"] or 0),
            "epoch": int(row["epoch"] or 0),
        }
        if best is None or cur["mIoU"] > best["mIoU"]:
            best=cur
if best is None:
    raise SystemExit("no best row found")
out={
    "run_root": run_root,
    "full_run_dir": full_dir,
    "best_mIoU": best["mIoU"],
    "best_aAcc": best["aAcc"],
    "best_mAcc": best["mAcc"],
    "best_iter": best["iter"],
    "best_epoch": best["epoch"],
    "final_lr": float("$TRY_LR"),
    "final_warmup": int("$WARMUP"),
}
with open(os.path.join(run_root,"best_config.json"),"w",encoding="utf-8") as f:
    json.dump(out,f,ensure_ascii=False,indent=2)

report=[
    "# CottonweedV4 Retrain Report",
    "",
    f"- run_root: {run_root}",
    f"- full_run_dir: {full_dir}",
    f"- best_mIoU: {best['mIoU']:.6f}",
    f"- best_aAcc: {best['aAcc']:.6f}",
    f"- best_mAcc: {best['mAcc']:.6f}",
    f"- best_iter/epoch: {best['iter']}/{best['epoch']}",
    f"- final_lr: {float('$TRY_LR')}",
    f"- final_warmup: {int('$WARMUP')}",
    "",
    "## Files",
    "- precheck logs: precheck*/train.log",
    "- full train log: fulltrain*/train.log",
    "- metrics: fulltrain*/metrics.tsv",
    "- figure: fulltrain*/loss_and_metrics.png",
    "- summary: fulltrain*/result_summary.md",
]
with open(os.path.join(run_root,"report.md"),"w",encoding="utf-8") as f:
    f.write("\\n".join(report)+"\\n")
print("done:", os.path.join(run_root,"report.md"))
PY

echo "[DONE] run completed: $RUN_ROOT"
