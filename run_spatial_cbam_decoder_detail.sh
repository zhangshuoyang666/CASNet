#!/bin/bash
# Spatial-CBAM + decoder_detail (mid-level enabled)
# 对比实验：在 exp_spatial_cbam_e50_itr1600 基础上，开启 mid-level 参与解码

source /root/miniconda3/etc/profile.d/conda.sh
conda activate Depplab

nohup python -u main.py \
    --dataset cottonweed \
    --data_root /root/autodl-tmp/dataset/cottonweed \
    --model deeplabv3plus_mobilenet \
    --attention_type spatial_cbam \
    --enable_decoder_detail \
    --gpu_id 0 \
    --output_stride 16 \
    --total_itrs 1600 \
    --val_interval 32 \
    --batch_size 16 \
    --val_batch_size 4 \
    --work_dir ./workdirs \
    --exp_name exp_spatial_cbam_decoder_detail_e50_itr1600 \
    > ./workdirs/train_spatial_cbam_decoder_detail_e50_itr1600.log 2>&1 &

echo "PID: $!"
echo "Log: ./workdirs/train_spatial_cbam_decoder_detail_e50_itr1600.log"
