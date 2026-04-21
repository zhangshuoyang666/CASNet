#!/usr/bin/env bash
set -euo pipefail

CONDA_SH="/root/miniconda3/etc/profile.d/conda.sh"
source "$CONDA_SH"
conda activate Depplab

PROJECT="/root/autodl-tmp/CODE/DeepLabV3Plus-Pytorch-master/DeepLabV3Plus-Pytorch-master"
BASE="$PROJECT/best-model-超参"

run_one() {
  local dir="$1"
  local name
  name="$(basename "$dir")"
  echo "========================================"
  echo "[Queue] START $name at $(date '+%F %T')"
  echo "========================================"

  cd "$PROJECT"
  bash "$dir/run.sh" > "$dir/train.log" 2>&1 || true

  echo "[Queue] Training done for $name at $(date '+%F %T')"
  python "$BASE/parse_and_plot.py" "$dir" "$name"
  echo "[Queue] Parsed $name"
  echo ""
}

echo "[Queue] Step 1 LR screening started at $(date '+%F %T')"

run_one "$BASE/01_lr_0p001_bs32"
run_one "$BASE/01_lr_0p005_bs32"
run_one "$BASE/01_lr_0p01_bs32"
run_one "$BASE/01_lr_0p02_bs32"

echo "[Queue] All 4 LR experiments done at $(date '+%F %T')"

python "$BASE/compare_experiments.py" \
  "$BASE/01_lr_0p001_bs32" \
  "$BASE/01_lr_0p005_bs32" \
  "$BASE/01_lr_0p01_bs32" \
  "$BASE/01_lr_0p02_bs32" \
  --output "$BASE/step1_lr_comparison.png" \
  --title "Step 1: LR Screening (bs=32, 800 iters)"

echo "[Queue] Comparison chart generated. Step 1 queue COMPLETE."
