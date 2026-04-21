#!/usr/bin/env bash
set -euo pipefail
source "/root/miniconda3/etc/profile.d/conda.sh"
conda activate Depplab
cd "/root/autodl-tmp/CODE/DeepLabV3Plus-Pytorch-master/DeepLabV3Plus-Pytorch-master"
exec python -u main.py --dataset cottonweed --data_root /root/autodl-tmp/dataset/cottonweed --cottonweed_mask_dir masks --model deeplabv3plus_mobilenet --attention_type spatial_cbam --enable_boundary_aux --use_saff --edge_loss_type bce --lambda_edge 0.1 --boundary_width 5 --gpu_id 0 --output_stride 16 --crop_size 513 --batch_size 32 --val_batch_size 4 --random_seed 1 --lr 0.02 --weight_decay 1e-4 --dense_aspp_rates 1,3,6,12,18 --aspp_variant standard --total_itrs 400 --val_interval 50 --print_interval 10 --work_dir /root/autodl-tmp/CODE/DeepLabV3Plus-Pytorch-master/DeepLabV3Plus-Pytorch-master/超参数/03_aug_vflip --exp_name run --aug_vflip
