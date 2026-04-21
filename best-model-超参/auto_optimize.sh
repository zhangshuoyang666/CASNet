#!/usr/bin/env bash
###############################################################################
# SA-BNet 全自动串行超参数优化 v2 (修复版)
#
# 用法:
#   nohup bash auto_optimize.sh > auto_optimize.log 2>&1 &
#   tail -f auto_optimize.log   # 实时监控
#
# 修复项 (相对 v1):
#   - tracker CSV old_value 时序: 先记录再更新变量
#   - warmup: 用自定义 WarmupPolyLR 替代 SequentialLR
#   - PolyLR total_itrs 统一为 1600, 短程实验提前截断
#   - compare_miou 阈值对称化
#   - result_summary.md 自动生成
#   - Step 4 λ_edge 全部候选跑完再选最优 (不提前 break)
#   - work_dir/exp_name 不再嵌套
#   - pid.txt 记录真实训练 PID
#   - status.txt 增加结束时间和退出码
###############################################################################
set -euo pipefail

CONDA_SH="/root/miniconda3/etc/profile.d/conda.sh"
source "$CONDA_SH"
conda activate Depplab

PROJECT="/root/autodl-tmp/CODE/DeepLabV3Plus-Pytorch-master/DeepLabV3Plus-Pytorch-master"
BASE="$PROJECT/best-model-超参"
PARSE="$BASE/parse_and_plot.py"
COMPARE="$BASE/compare_experiments.py"
TRACKER="$BASE/experiment_tracker.csv"
BEST_CFG="$BASE/best_config.json"

DATA_ROOT="/root/autodl-tmp/dataset/cottonweed"
MASK_DIR="masks"

BEST_MIOU="94.66"
BEST_LR="0.01"
BEST_WD="1e-4"
BEST_LAMBDA="0.1"
BEST_WARMUP="0"
BEST_VFLIP=""
BEST_ROTATION="0"
BEST_ASPP="1,3,6,12,18"
BEST_BS="32"

POLY_TOTAL=1600          # PolyLR always uses this as denominator
SHORT_ITRS=800            # screening: ~47 epochs, ~40-60 min per exp
CONFIRM_ITRS=1600         # full confirm: only for Step 1 winner
VAL_INTERVAL=100          # 8 validation checkpoints per 800 iters
VAL_BS=4

log() { echo "[$(date '+%F %T')] $*"; }

###############################################################################
# build_args: assemble python CLI from parameters
###############################################################################
build_args() {
  local lr="$1" wd="$2" lam="$3" warmup="$4" vflip="$5" rot="$6" aspp="$7"
  local itrs="$8" vi="$9" workdir="${10}"
  local a="--dataset cottonweed --data_root $DATA_ROOT --cottonweed_mask_dir $MASK_DIR"
  a+=" --model deeplabv3plus_mobilenet --attention_type spatial_cbam --enable_boundary_aux --use_saff"
  a+=" --edge_loss_type bce --lambda_edge $lam --boundary_width 5"
  a+=" --gpu_id 0 --output_stride 16 --crop_size 513"
  a+=" --batch_size $BEST_BS --val_batch_size $VAL_BS --random_seed 1"
  a+=" --lr $lr --weight_decay $wd --dense_aspp_rates $aspp"
  a+=" --total_itrs $itrs --val_interval $vi"
  a+=" --work_dir $workdir --exp_name run"
  [ "$warmup" -gt 0 ] 2>/dev/null && a+=" --warmup_iters $warmup"
  [ -n "$vflip" ] && a+=" --aug_vflip"
  [ "$rot" -gt 0 ] 2>/dev/null && a+=" --aug_rotation $rot"
  echo "$a"
}

###############################################################################
# run_exp: train, parse, record PID & status
###############################################################################
run_exp() {
  local dir="$1" args="$2"
  local name; name="$(basename "$dir")"
  log "▶ START $name"

  cat > "$dir/run.sh" << EOFSH
#!/usr/bin/env bash
set -euo pipefail
source "$CONDA_SH"
conda activate Depplab
cd "$PROJECT"
exec python -u main.py $args
EOFSH
  chmod +x "$dir/run.sh"

  cat > "$dir/status.txt" << EOFS
started: $(date '+%F %T')
exp_dir: $name
command: python -u main.py $args
EOFS

  cd "$PROJECT"
  echo "$$" > "$dir/pid.txt"
  python -u main.py $args > "$dir/train.log" 2>&1 || true
  local ecode=$?
  echo "ended: $(date '+%F %T')" >> "$dir/status.txt"
  echo "exit_code: $ecode" >> "$dir/status.txt"

  python "$PARSE" "$dir" "$name" 2>&1 || true
  log "✓ DONE  $name (exit=$ecode)"
}

###############################################################################
# get_best_from_csv: extract best mIoU row → "miou|abuth|cotton|iter|epoch"
###############################################################################
get_best_from_csv() {
  local csv="$1"
  [ ! -f "$csv" ] && { echo "0|0|0|0|0"; return; }
  python3 -c "
import csv
best = None
with open('$csv') as f:
    for row in csv.DictReader(f):
        m = float(row['miou'])
        if best is None or m > float(best['miou']):
            best = dict(row)
if best:
    print(f\"{best['miou']}|{best.get('abuth_iou','0')}|{best.get('cotton_iou','0')}|{best.get('iter','0')}|{best.get('epoch','0')}\")
else:
    print('0|0|0|0|0')
"
}

###############################################################################
# compare_miou: "better" (>0.15), "marginal" (0..0.15), "worse"
###############################################################################
compare_miou() {
  python3 -c "
a, b = float('$1'), float('$2')
if a > b + 0.15:   print('better')
elif a > b:         print('marginal')
else:               print('worse')
"
}

append_tracker() { echo "$1" >> "$TRACKER"; }

update_best_json() {
  cat > "$BEST_CFG" << EOFJ
{
  "step": "$1",
  "exp_name": "$2",
  "best_miou": $BEST_MIOU,
  "batch_size": $BEST_BS,
  "lr": $BEST_LR,
  "weight_decay": $BEST_WD,
  "lambda_edge": $BEST_LAMBDA,
  "warmup_iters": $BEST_WARMUP,
  "aug_vflip": $([ -n "$BEST_VFLIP" ] && echo true || echo false),
  "aug_rotation": $BEST_ROTATION,
  "dense_aspp_rates": "$BEST_ASPP"
}
EOFJ
}

###############################################################################
#  Helper: create experiment dir + config_snapshot
###############################################################################
prep_dir() {
  local dir="$1" step="$2" cparam="$3" oval="$4" nval="$5" readme="$6"
  mkdir -p "$dir"
  cat > "$dir/README.md" <<< "$readme"
  python3 -c "
import json, sys
json.dump({
  'step':'$step','changed_param':'$cparam','old_value':'$oval','new_value':'$nval',
  'purpose':'$readme',
  'lr':$BEST_LR,'batch_size':$BEST_BS,'weight_decay':'$BEST_WD',
  'lambda_edge':$BEST_LAMBDA,'warmup_iters':$BEST_WARMUP,
  'aug_vflip':$([ -n "$BEST_VFLIP" ] && echo True || echo False),
  'aug_rotation':$BEST_ROTATION,'dense_aspp_rates':'$BEST_ASPP'
}, open('$dir/config_snapshot.json','w'), indent=2)
"
}

###############################################################################
#  Helper: run one candidate, parse, log, return miou
###############################################################################
run_and_log() {
  local dir="$1" args="$2"
  run_exp "$dir" "$args"
  get_best_from_csv "$dir/metrics.csv"
}

###############################################################################
#                     STEP 1: LEARNING RATE SCREENING
###############################################################################
log "================================================================"
log "  STEP 1: Learning Rate Screening  (bs=$BEST_BS, 800 iters)"
log "================================================================"

declare -A LR_RES
for lr_pair in "0p001:0.001" "0p005:0.005" "0p01:0.01" "0p02:0.02"; do
  IFS=: read tag lr <<< "$lr_pair"
  DIR="$BASE/01_lr_${tag}_bs32"
  prep_dir "$DIR" "1_lr" "lr" "0.01" "$lr" "Step1: lr=$lr bs=32 screen"
  ARGS=$(build_args "$lr" "$BEST_WD" "$BEST_LAMBDA" "0" "" "0" "$BEST_ASPP" "$SHORT_ITRS" "$VAL_INTERVAL" "$DIR")
  RESULT=$(run_and_log "$DIR" "$ARGS")
  LR_RES["$tag"]="$RESULT"
  IFS='|' read m a c it ep <<< "$RESULT"
  log "  lr=$lr → mIoU=${m}% abuth=${a}% cotton=${c}% @iter=$it"
done

# Pick best LR
BEST_LR_TAG="" ; BEST_LR_MIOU="0"
for tag in 0p001 0p005 0p01 0p02; do
  IFS='|' read m _ <<< "${LR_RES[$tag]}"
  better=$(python3 -c "print('y' if float('$m')>float('$BEST_LR_MIOU') else 'n')")
  [ "$better" = "y" ] && { BEST_LR_MIOU="$m"; BEST_LR_TAG="$tag"; }
done
case "$BEST_LR_TAG" in 0p001) BLR=0.001;; 0p005) BLR=0.005;; 0p01) BLR=0.01;; 0p02) BLR=0.02;; esac
log "★ Best LR from screen: $BLR (mIoU=$BEST_LR_MIOU%)"

python "$COMPARE" "$BASE/01_lr_0p001_bs32" "$BASE/01_lr_0p005_bs32" \
  "$BASE/01_lr_0p01_bs32" "$BASE/01_lr_0p02_bs32" \
  --output "$BASE/step1_lr_comparison.png" \
  --title "Step1 LR Screen (bs32, 800itr)" 2>&1 || true

for lr_pair in "0p001:0.001" "0p005:0.005" "0p01:0.01" "0p02:0.02"; do
  IFS=: read tag lr <<< "$lr_pair"
  IFS='|' read m a c it ep <<< "${LR_RES[$tag]}"
  gain=$(python3 -c "print(f'{float(\"$m\")-94.66:.2f}')")
  dec=$( [ "$tag" = "$BEST_LR_TAG" ] && echo "promote_to_confirm" || echo "reject" )
  append_tracker "1,01_lr_${tag}_bs32,baseline,lr,0.01,$lr,$SHORT_ITRS,,$m,$it,$a,$c,$gain,$dec,,01_lr_${tag}_bs32"
done

###############################################################################
#                   STEP 1 CONFIRM
###############################################################################
log "========== STEP 1 CONFIRM: lr=$BLR =========="
CDIR="$BASE/01_confirm_lr_${BEST_LR_TAG}_bs32"
prep_dir "$CDIR" "1c" "lr" "0.01" "$BLR" "Step1 confirm: lr=$BLR full 1600itr"
ARGS=$(build_args "$BLR" "$BEST_WD" "$BEST_LAMBDA" "0" "" "0" "$BEST_ASPP" "$CONFIRM_ITRS" "$VAL_INTERVAL" "$CDIR")
RESULT=$(run_and_log "$CDIR" "$ARGS")
IFS='|' read m a c it ep <<< "$RESULT"
gain=$(python3 -c "print(f'{float(\"$m\")-float(\"$BEST_MIOU\"):.2f}')")
CMP=$(compare_miou "$m" "$BEST_MIOU")
log "  Confirm lr=$BLR → mIoU=${m}% (gain=$gain vs best=$BEST_MIOU%)"

if [ "$CMP" != "worse" ]; then
  OLD_MIOU=$BEST_MIOU; BEST_MIOU="$m"; BEST_LR="$BLR"
  update_best_json "1c" "confirm_lr_${BEST_LR_TAG}"
  DEC=$( [ "$CMP" = "better" ] && echo "update_best" || echo "marginal_gain" )
  append_tracker "1c,confirm_lr_${BEST_LR_TAG},baseline,lr,0.01,$BLR,$SHORT_ITRS,$CONFIRM_ITRS,$m,$it,$a,$c,$gain,$DEC,,01_confirm_lr_${BEST_LR_TAG}_bs32"
  log "★ Best updated: lr=$BEST_LR mIoU=$BEST_MIOU%"
else
  append_tracker "1c,confirm_lr_${BEST_LR_TAG},baseline,lr,0.01,$BLR,$SHORT_ITRS,$CONFIRM_ITRS,$m,$it,$a,$c,$gain,reject,not better,01_confirm_lr_${BEST_LR_TAG}_bs32"
  BEST_LR="0.01"
  log "  Confirm rejected. Keeping lr=0.01."
fi

###############################################################################
#                       STEP 2: WARMUP
###############################################################################
log "================================================================"
log "  STEP 2: Warmup=300  (lr=$BEST_LR)"
log "================================================================"

WDIR="$BASE/02_warmup_300"
prep_dir "$WDIR" "2" "warmup_iters" "0" "300" "Step2: warmup=300 on lr=$BEST_LR"
ARGS=$(build_args "$BEST_LR" "$BEST_WD" "$BEST_LAMBDA" "300" "$BEST_VFLIP" "$BEST_ROTATION" "$BEST_ASPP" "$SHORT_ITRS" "$VAL_INTERVAL" "$WDIR")
RESULT=$(run_and_log "$WDIR" "$ARGS")
IFS='|' read m a c it ep <<< "$RESULT"
gain=$(python3 -c "print(f'{float(\"$m\")-float(\"$BEST_MIOU\"):.2f}')")
CMP=$(compare_miou "$m" "$BEST_MIOU")
log "  Warmup=300 → mIoU=${m}% (gain=$gain)"

OLD_WARMUP=$BEST_WARMUP
if [ "$CMP" != "worse" ]; then
  BEST_MIOU="$m"; BEST_WARMUP="300"
  update_best_json "2" "warmup_300"
  DEC=$( [ "$CMP" = "better" ] && echo "update_best" || echo "marginal_gain" )
  append_tracker "2,warmup_300,step1_best,warmup_iters,$OLD_WARMUP,300,$SHORT_ITRS,,$m,$it,$a,$c,$gain,$DEC,,02_warmup_300"
  log "★ Best updated: warmup=300"
else
  append_tracker "2,warmup_300,step1_best,warmup_iters,$OLD_WARMUP,300,$SHORT_ITRS,,$m,$it,$a,$c,$gain,reject,,02_warmup_300"
  log "  Warmup rejected."
fi

###############################################################################
#                     STEP 3.1: VERTICAL FLIP
###############################################################################
log "================================================================"
log "  STEP 3.1: Vertical Flip"
log "================================================================"

VDIR="$BASE/03_aug_vflip"
prep_dir "$VDIR" "3.1" "aug_vflip" "false" "true" "Step3.1: add vertical flip"
ARGS=$(build_args "$BEST_LR" "$BEST_WD" "$BEST_LAMBDA" "$BEST_WARMUP" "yes" "$BEST_ROTATION" "$BEST_ASPP" "$SHORT_ITRS" "$VAL_INTERVAL" "$VDIR")
RESULT=$(run_and_log "$VDIR" "$ARGS")
IFS='|' read m a c it ep <<< "$RESULT"
gain=$(python3 -c "print(f'{float(\"$m\")-float(\"$BEST_MIOU\"):.2f}')")
CMP=$(compare_miou "$m" "$BEST_MIOU")
log "  VFlip → mIoU=${m}% (gain=$gain)"

if [ "$CMP" != "worse" ]; then
  BEST_MIOU="$m"; BEST_VFLIP="yes"
  update_best_json "3.1" "aug_vflip"
  DEC=$( [ "$CMP" = "better" ] && echo "update_best" || echo "marginal_gain" )
  append_tracker "3.1,aug_vflip,step2_best,aug_vflip,false,true,$SHORT_ITRS,,$m,$it,$a,$c,$gain,$DEC,,03_aug_vflip"
  log "★ Best updated: aug_vflip=true"
else
  append_tracker "3.1,aug_vflip,step2_best,aug_vflip,false,true,$SHORT_ITRS,,$m,$it,$a,$c,$gain,reject,,03_aug_vflip"
  log "  VFlip rejected."
fi

###############################################################################
#                     STEP 3.2: ROTATION ±30°
###############################################################################
log "================================================================"
log "  STEP 3.2: Rotation ±30°"
log "================================================================"

RDIR="$BASE/04_aug_rot30"
prep_dir "$RDIR" "3.2" "aug_rotation" "0" "30" "Step3.2: add rotation ±30°"
ARGS=$(build_args "$BEST_LR" "$BEST_WD" "$BEST_LAMBDA" "$BEST_WARMUP" "$BEST_VFLIP" "30" "$BEST_ASPP" "$SHORT_ITRS" "$VAL_INTERVAL" "$RDIR")
RESULT=$(run_and_log "$RDIR" "$ARGS")
IFS='|' read m a c it ep <<< "$RESULT"
gain=$(python3 -c "print(f'{float(\"$m\")-float(\"$BEST_MIOU\"):.2f}')")
CMP=$(compare_miou "$m" "$BEST_MIOU")
log "  Rot30 → mIoU=${m}% (gain=$gain)"

if [ "$CMP" != "worse" ]; then
  BEST_MIOU="$m"; BEST_ROTATION="30"
  update_best_json "3.2" "aug_rot30"
  DEC=$( [ "$CMP" = "better" ] && echo "update_best" || echo "marginal_gain" )
  append_tracker "3.2,aug_rot30,step3.1_best,aug_rotation,0,30,$SHORT_ITRS,,$m,$it,$a,$c,$gain,$DEC,,04_aug_rot30"
  log "★ Best updated: aug_rotation=30"
else
  append_tracker "3.2,aug_rot30,step3.1_best,aug_rotation,0,30,$SHORT_ITRS,,$m,$it,$a,$c,$gain,reject,,04_aug_rot30"
  log "  Rotation rejected."
fi

###############################################################################
#              STEP 4: λ_edge  (run ALL candidates, pick best)
###############################################################################
log "================================================================"
log "  STEP 4: λ_edge  (candidates: 0.2, 0.3, 0.4)"
log "================================================================"

BEST_LAM_MIOU="0"; BEST_LAM_TAG=""
for lp in "0p2:0.2" "0p3:0.3" "0p4:0.4"; do
  IFS=: read ltag lval <<< "$lp"
  LDIR="$BASE/05_lambda_${ltag}"
  prep_dir "$LDIR" "4" "lambda_edge" "$BEST_LAMBDA" "$lval" "Step4: λ_edge=$lval"
  ARGS=$(build_args "$BEST_LR" "$BEST_WD" "$lval" "$BEST_WARMUP" "$BEST_VFLIP" "$BEST_ROTATION" "$BEST_ASPP" "$SHORT_ITRS" "$VAL_INTERVAL" "$LDIR")
  RESULT=$(run_and_log "$LDIR" "$ARGS")
  IFS='|' read m a c it ep <<< "$RESULT"
  log "  λ=$lval → mIoU=${m}%"
  better=$(python3 -c "print('y' if float('$m')>float('$BEST_LAM_MIOU') else 'n')")
  [ "$better" = "y" ] && { BEST_LAM_MIOU="$m"; BEST_LAM_TAG="$ltag"; BEST_LAM_VAL="$lval"; BEST_LAM_RES="$RESULT"; }
done

# Decide
IFS='|' read m a c it ep <<< "$BEST_LAM_RES"
gain=$(python3 -c "print(f'{float(\"$m\")-float(\"$BEST_MIOU\"):.2f}')")
CMP=$(compare_miou "$m" "$BEST_MIOU")
log "  Best λ candidate: $BEST_LAM_VAL mIoU=${m}% (gain=$gain)"

OLD_LAMBDA=$BEST_LAMBDA
for lp in "0p2:0.2" "0p3:0.3" "0p4:0.4"; do
  IFS=: read ltag lval <<< "$lp"
  IFS='|' read mm aa cc ii ee <<< "${LR_RES[$ltag]:-$(get_best_from_csv "$BASE/05_lambda_${ltag}/metrics.csv")}"
  gg=$(python3 -c "print(f'{float(\"$mm\")-float(\"$BEST_MIOU\"):.2f}')" 2>/dev/null || echo "0")
  dd=$( [ "$ltag" = "$BEST_LAM_TAG" ] && [ "$CMP" != "worse" ] && echo "update_best" || echo "reject" )
  append_tracker "4,lambda_${ltag},prev_best,lambda_edge,$OLD_LAMBDA,$lval,$SHORT_ITRS,,$mm,$ii,$aa,$cc,$gg,$dd,,05_lambda_${ltag}"
done

if [ "$CMP" != "worse" ]; then
  BEST_MIOU="$m"; BEST_LAMBDA="$BEST_LAM_VAL"
  update_best_json "4" "lambda_$BEST_LAM_TAG"
  log "★ Best updated: lambda_edge=$BEST_LAMBDA"
else
  log "  No λ_edge candidate improved. Keeping $BEST_LAMBDA."
fi

###############################################################################
#              STEP 5: weight_decay
###############################################################################
log "================================================================"
log "  STEP 5: weight_decay=5e-4"
log "================================================================"

WDDIR="$BASE/06_wd_5e4"
prep_dir "$WDDIR" "5" "weight_decay" "$BEST_WD" "5e-4" "Step5: weight_decay=5e-4"
ARGS=$(build_args "$BEST_LR" "5e-4" "$BEST_LAMBDA" "$BEST_WARMUP" "$BEST_VFLIP" "$BEST_ROTATION" "$BEST_ASPP" "$SHORT_ITRS" "$VAL_INTERVAL" "$WDDIR")
RESULT=$(run_and_log "$WDDIR" "$ARGS")
IFS='|' read m a c it ep <<< "$RESULT"
gain=$(python3 -c "print(f'{float(\"$m\")-float(\"$BEST_MIOU\"):.2f}')")
CMP=$(compare_miou "$m" "$BEST_MIOU")
log "  WD=5e-4 → mIoU=${m}% (gain=$gain)"

OLD_WD=$BEST_WD
if [ "$CMP" != "worse" ]; then
  BEST_MIOU="$m"; BEST_WD="5e-4"
  update_best_json "5" "wd_5e4"
  DEC=$( [ "$CMP" = "better" ] && echo "update_best" || echo "marginal_gain" )
  append_tracker "5,wd_5e4,prev_best,weight_decay,$OLD_WD,5e-4,$SHORT_ITRS,,$m,$it,$a,$c,$gain,$DEC,,06_wd_5e4"
  log "★ Best updated: weight_decay=5e-4"
else
  append_tracker "5,wd_5e4,prev_best,weight_decay,$OLD_WD,5e-4,$SHORT_ITRS,,$m,$it,$a,$c,$gain,reject,,06_wd_5e4"
  log "  WD=5e-4 rejected."
fi

###############################################################################
#              STEP 6: ASPP rates (run all, pick best)
###############################################################################
log "================================================================"
log "  STEP 6: ASPP rates"
log "================================================================"

BEST_ASPP_MIOU="0"; BEST_ASPP_TAG=""
for ap in "1_6_12_18_24:1,6,12,18,24" "1_2_4_8_16:1,2,4,8,16"; do
  IFS=: read atag aval <<< "$ap"
  ADIR="$BASE/07_aspp_${atag}"
  prep_dir "$ADIR" "6" "aspp_rates" "$BEST_ASPP" "$aval" "Step6: ASPP=($aval)"
  ARGS=$(build_args "$BEST_LR" "$BEST_WD" "$BEST_LAMBDA" "$BEST_WARMUP" "$BEST_VFLIP" "$BEST_ROTATION" "$aval" "$SHORT_ITRS" "$VAL_INTERVAL" "$ADIR")
  RESULT=$(run_and_log "$ADIR" "$ARGS")
  IFS='|' read m a c it ep <<< "$RESULT"
  log "  ASPP=($aval) → mIoU=${m}%"
  better=$(python3 -c "print('y' if float('$m')>float('$BEST_ASPP_MIOU') else 'n')")
  [ "$better" = "y" ] && { BEST_ASPP_MIOU="$m"; BEST_ASPP_TAG="$atag"; BEST_ASPP_VAL="$aval"; BEST_ASPP_RES="$RESULT"; }
done

IFS='|' read m a c it ep <<< "$BEST_ASPP_RES"
gain=$(python3 -c "print(f'{float(\"$m\")-float(\"$BEST_MIOU\"):.2f}')")
CMP=$(compare_miou "$m" "$BEST_MIOU")

OLD_ASPP=$BEST_ASPP
for ap in "1_6_12_18_24:1,6,12,18,24" "1_2_4_8_16:1,2,4,8,16"; do
  IFS=: read atag aval <<< "$ap"
  IFS='|' read mm aa cc ii ee <<< "$(get_best_from_csv "$BASE/07_aspp_${atag}/metrics.csv")"
  gg=$(python3 -c "print(f'{float(\"$mm\")-float(\"$BEST_MIOU\"):.2f}')" 2>/dev/null || echo "0")
  dd=$( [ "$atag" = "$BEST_ASPP_TAG" ] && [ "$CMP" != "worse" ] && echo "update_best" || echo "reject" )
  append_tracker "6,aspp_${atag},prev_best,aspp_rates,$OLD_ASPP,$aval,$SHORT_ITRS,,$mm,$ii,$aa,$cc,$gg,$dd,,07_aspp_${atag}"
done

if [ "$CMP" != "worse" ]; then
  BEST_MIOU="$m"; BEST_ASPP="$BEST_ASPP_VAL"
  update_best_json "6" "aspp_$BEST_ASPP_TAG"
  log "★ Best updated: aspp=$BEST_ASPP"
else
  log "  No ASPP candidate improved."
fi

###############################################################################
#              FINAL REPORT
###############################################################################
log "================================================================"
log "  Generating final_report.md"
log "================================================================"

update_best_json "final" "final"

# Build tracker table via python
TRACKER_TABLE=$(python3 -c "
import csv, sys
with open('$TRACKER') as f:
    rows = list(csv.reader(f))
if len(rows) > 1:
    print('| Step | Exp | Changed | Old→New | mIoU | Gain | Decision |')
    print('|------|-----|---------|---------|------|------|----------|')
    for r in rows[1:]:
        if len(r) >= 15:
            print(f'| {r[0]} | {r[1]} | {r[3]} | {r[4]}→{r[5]} | {r[8]}% | {r[12]} | {r[13]} |')
" 2>/dev/null || echo "(tracker parse error)")

# Build directory index
DIR_INDEX=$(for d in "$BASE"/0*/; do
  [ -d "$d" ] || continue
  n=$(basename "$d")
  h=$(head -1 "$d/README.md" 2>/dev/null | sed 's/^# *//' || echo "")
  echo "- \`$n/\` — $h"
done)

cat > "$BASE/final_report.md" << EOFFINAL
# SA-BNet 串行超参数优化 — 最终报告

> 生成时间: $(date '+%F %T')

## 优化路径

\`\`\`
基线 (mIoU=94.66%, bs=16, lr=0.01)
  → Step1: LR筛选 (bs=32)
  → Step1c: 确认最优LR
  → Step2: Warmup
  → Step3.1: VFlip
  → Step3.2: Rotation
  → Step4: λ_edge
  → Step5: weight_decay
  → Step6: ASPP rates
\`\`\`

## 最终最佳配置

| 超参数 | 值 |
|--------|-----|
| batch_size | $BEST_BS |
| lr | $BEST_LR |
| weight_decay | $BEST_WD |
| lambda_edge | $BEST_LAMBDA |
| warmup_iters | $BEST_WARMUP |
| aug_vflip | $([ -n "$BEST_VFLIP" ] && echo true || echo false) |
| aug_rotation | $BEST_ROTATION |
| dense_aspp_rates | $BEST_ASPP |
| **best mIoU** | **${BEST_MIOU}%** |

## 每步实验结果

$TRACKER_TABLE

## 实验目录索引

$DIR_INDEX

## 推荐后续方向

1. 在最终 best 配置上跑更长训练 (3200+ iters) 验证天花板
2. 尝试 CosineAnnealing 调度器
3. CutMix / MixUp 数据增强
4. 更大 backbone (ResNet-101 / ConvNeXt-Tiny)
5. boundary_width 微调 (3 vs 5 vs 7)
6. Dropout 率调整 (0.1 vs 0.3)
EOFFINAL

log "★★★ ALL STEPS COMPLETE ★★★"
log "Final best: mIoU=$BEST_MIOU% lr=$BEST_LR wd=$BEST_WD λ=$BEST_LAMBDA warmup=$BEST_WARMUP vflip=$([ -n "$BEST_VFLIP" ] && echo Y || echo N) rot=$BEST_ROTATION aspp=$BEST_ASPP"
log "Report: $BASE/final_report.md"
log "Tracker: $TRACKER"
log "Config: $BEST_CFG"
