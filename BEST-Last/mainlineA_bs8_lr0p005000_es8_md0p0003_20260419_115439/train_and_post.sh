#!/usr/bin/env bash
set -euo pipefail
source "/root/miniconda3/etc/profile.d/conda.sh"
conda activate Depplab
cd "/root/autodl-tmp/CODE/DeepLabV3Plus-Pytorch-master/DeepLabV3Plus-Pytorch-master"

python -u main.py \
  --dataset cottonweed \
  --data_root "/root/autodl-tmp/dataset/cottonweed" \
  --cottonweed_mask_dir masks \
  --model deeplabv3plus_mobilenet \
  --attention_type spatial_cbam \
  --enable_boundary_aux \
  --use_saff \
  --saff_f1_source mid \
  --saff_f2_source aspp \
  --edge_loss_type bce \
  --lambda_edge 0.1 \
  --boundary_width 5 \
  --gpu_id "0" \
  --output_stride 16 \
  --crop_size 513 \
  --batch_size "8" \
  --val_batch_size 4 \
  --random_seed 1 \
  --lr "0.005000" \
  --weight_decay 1e-4 \
  --dense_aspp_rates 1,2,4,8,16 \
  --warmup_iters 0 \
  --total_itrs -1 \
  --val_interval 50 \
  --print_interval 10 \
  --work_dir "/root/autodl-tmp/CODE/DeepLabV3Plus-Pytorch-master/DeepLabV3Plus-Pytorch-master/BEST-Last" \
  --exp_name "mainlineA_bs8_lr0p005000_es8_md0p0003_20260419_115439" \
  --enable_early_stop \
  --early_stop_metric "Mean IoU" \
  --early_stop_patience "8" \
  --early_stop_min_delta "0.0003" \
  --min_itrs_before_early_stop 1000 \
  > "/root/autodl-tmp/CODE/DeepLabV3Plus-Pytorch-master/DeepLabV3Plus-Pytorch-master/BEST-Last/mainlineA_bs8_lr0p005000_es8_md0p0003_20260419_115439/train.log" 2>&1

python "/root/autodl-tmp/CODE/DeepLabV3Plus-Pytorch-master/DeepLabV3Plus-Pytorch-master/BEST-Last/postprocess_mainlineA.py" --run_dir "/root/autodl-tmp/CODE/DeepLabV3Plus-Pytorch-master/DeepLabV3Plus-Pytorch-master/BEST-Last/mainlineA_bs8_lr0p005000_es8_md0p0003_20260419_115439" >> "/root/autodl-tmp/CODE/DeepLabV3Plus-Pytorch-master/DeepLabV3Plus-Pytorch-master/BEST-Last/mainlineA_bs8_lr0p005000_es8_md0p0003_20260419_115439/train.log" 2>&1
