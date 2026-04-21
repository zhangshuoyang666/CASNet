# CottonWeed Fine-grained Segmentation Experiment Report

## Overview
- 目标：增强 cotton 与 abuth 的细粒度可分性（非泛化型提升）。
- 模块：FineGrainedFusion / TextureEnhance / DenseASPP or DeformASPP / ImprovedDecoder(detail path).

## Key Results (best checkpoint by mIoU)
| Experiment | mIoU | mF1 | aAcc | cotton IoU | cotton Acc | abuth IoU | abuth Acc | cotton->abuth | abuth->cotton |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cottonweed_deeplabv3plus_mobilenet_baseline_itr40_20260411_013332 | 32.70 | 37.81 | 87.12 | 3.50 | 3.55 | 7.16 | 7.51 | 9.88 | 0.47 |
| cottonweed_deeplabv3plus_mobilenet_fg_fusion_itr40_20260411_013332 | 37.41 | 45.31 | 88.11 | 9.10 | 10.37 | 14.45 | 14.88 | 6.00 | 6.44 |
| cottonweed_deeplabv3plus_mobilenet_full_model_itr40_20260411_013332 | 40.68 | 48.48 | 89.32 | 2.34 | 2.42 | 30.25 | 31.39 | 7.81 | 1.58 |
| cottonweed_deeplabv3plus_mobilenet_improved_aspp_dense_itr40_20260411_013332 | 37.23 | 43.55 | 88.54 | 0.02 | 0.02 | 21.99 | 26.05 | 35.09 | 0.01 |
| cottonweed_deeplabv3plus_mobilenet_texture_enhance_itr40_20260411_013332 | 39.32 | 46.13 | 88.56 | 0.03 | 0.03 | 27.82 | 37.89 | 48.02 | 0.04 |

## Per-class Snapshot
### cottonweed_deeplabv3plus_mobilenet_baseline_itr40_20260411_013332
- Best Iter/Epoch: 40/1
- background: IoU 87.43 / Acc 99.96 / F1 93.29
- cotton: IoU 3.50 / Acc 3.55 / F1 6.77
- abuth: IoU 7.16 / Acc 7.51 / F1 13.36
- confusion: cotton->abuth 9.88 | abuth->cotton 0.47

### cottonweed_deeplabv3plus_mobilenet_fg_fusion_itr40_20260411_013332
- Best Iter/Epoch: 40/1
- background: IoU 88.67 / Acc 99.97 / F1 93.99
- cotton: IoU 9.10 / Acc 10.37 / F1 16.68
- abuth: IoU 14.45 / Acc 14.88 / F1 25.25
- confusion: cotton->abuth 6.00 | abuth->cotton 6.44

### cottonweed_deeplabv3plus_mobilenet_full_model_itr40_20260411_013332
- Best Iter/Epoch: 40/1
- background: IoU 89.44 / Acc 99.99 / F1 94.42
- cotton: IoU 2.34 / Acc 2.42 / F1 4.58
- abuth: IoU 30.25 / Acc 31.39 / F1 46.45
- confusion: cotton->abuth 7.81 | abuth->cotton 1.58

### cottonweed_deeplabv3plus_mobilenet_improved_aspp_dense_itr40_20260411_013332
- Best Iter/Epoch: 40/1
- background: IoU 89.67 / Acc 99.79 / F1 94.56
- cotton: IoU 0.02 / Acc 0.02 / F1 0.04
- abuth: IoU 21.99 / Acc 26.05 / F1 36.06
- confusion: cotton->abuth 35.09 | abuth->cotton 0.01

### cottonweed_deeplabv3plus_mobilenet_texture_enhance_itr40_20260411_013332
- Best Iter/Epoch: 40/1
- background: IoU 90.11 / Acc 98.53 / F1 94.80
- cotton: IoU 0.03 / Acc 0.03 / F1 0.06
- abuth: IoU 27.82 / Acc 37.89 / F1 43.53
- confusion: cotton->abuth 48.02 | abuth->cotton 0.04

