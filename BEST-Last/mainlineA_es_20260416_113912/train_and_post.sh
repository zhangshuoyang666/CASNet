#!/usr/bin/env bash
set -euo pipefail
source "/root/miniconda3/etc/profile.d/conda.sh"
conda activate Depplab
cd "/root/autodl-tmp/CODE/DeepLabV3Plus-Pytorch-master/DeepLabV3Plus-Pytorch-master"

python -u main.py \
  --dataset cottonweed \
  --data_root "/root/autodl-tmp/dataset/cottonweedV2" \
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
  --batch_size 32 \
  --val_batch_size 4 \
  --random_seed 1 \
  --lr 0.02 \
  --weight_decay 1e-4 \
  --dense_aspp_rates 1,3,6,12,18 \
  --total_itrs -1 \
  --val_interval 50 \
  --print_interval 10 \
  --work_dir "/root/autodl-tmp/CODE/DeepLabV3Plus-Pytorch-master/DeepLabV3Plus-Pytorch-master/BEST-Last" \
  --exp_name "mainlineA_es_20260416_113912" \
  --enable_early_stop \
  --early_stop_metric "Mean IoU" \
  --early_stop_patience 10 \
  --early_stop_min_delta 0.0005 \
  --min_itrs_before_early_stop 1000 \
  --aug_vflip \
  > "/root/autodl-tmp/CODE/DeepLabV3Plus-Pytorch-master/DeepLabV3Plus-Pytorch-master/BEST-Last/mainlineA_es_20260416_113912/train.log" 2>&1

python "/root/autodl-tmp/CODE/DeepLabV3Plus-Pytorch-master/DeepLabV3Plus-Pytorch-master/BEST-Last/postprocess_mainlineA.py" --run_dir "/root/autodl-tmp/CODE/DeepLabV3Plus-Pytorch-master/DeepLabV3Plus-Pytorch-master/BEST-Last/mainlineA_es_20260416_113912" >> "/root/autodl-tmp/CODE/DeepLabV3Plus-Pytorch-master/DeepLabV3Plus-Pytorch-master/BEST-Last/mainlineA_es_20260416_113912/train.log" 2>&1
