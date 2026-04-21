# SA-BNet 串行超参数优化 — 最终报告

> 生成时间: 2026-04-16 06:17:09

## 优化路径

```
基线 (mIoU=94.66%, bs=16, lr=0.01)
  → Step1: LR筛选 (bs=32)
  → Step1c: 确认最优LR
  → Step2: Warmup
  → Step3.1: VFlip
  → Step3.2: Rotation
  → Step4: λ_edge
  → Step5: weight_decay
  → Step6: ASPP rates
```

## 最终最佳配置

| 超参数 | 值 |
|--------|-----|
| batch_size | 32 |
| lr | 0.02 |
| weight_decay | 1e-4 |
| lambda_edge | 0.1 |
| warmup_iters | 0 |
| aug_vflip | false |
| aug_rotation | 0 |
| dense_aspp_rates | 1,2,4,8,16 |
| **best mIoU** | **93.83%** |

## 每步实验结果

| Step | Exp | Changed | Old→New | mIoU | Gain | Decision |
|------|-----|---------|---------|------|------|----------|
| 1 | 01_lr_0p005_bs32 | lr | 0.01→0.005 | 90.68% | 0.68 | reject |
| 1 | 01_lr_0p01_bs32 | lr | 0.01→0.01 | 92.4% | 2.40 | reject |
| 1 | 01_lr_0p02_bs32 | lr | 0.01→0.02 | 93.05% | 3.05 | promote_to_confirm |
| 1c | confirm_lr_0p02 | lr | 0.01→0.02 | 93.69% | 3.69 | update_best |
| 2 | warmup_300 | warmup_iters | 0→300 | 90.26% | -3.43 | reject |
| 3.1 | aug_vflip | aug_vflip | false→true | 92.03% | -1.66 | reject |
| 3.2 | aug_rot30 | aug_rotation | 0→30 | 90.54% | -3.15 | reject |
| 4 | lambda_0p2 | lambda_edge | 0.1→0.2 | 92.01% | -1.68 | reject |
| 4 | lambda_0p3 | lambda_edge | 0.1→0.3 | 91.7% | -1.99 | reject |
| 4 | lambda_0p4 | lambda_edge | 0.1→0.4 | 91.73% | -1.96 | reject |
| 5 | wd_5e4 | weight_decay | 1e-4→5e-4 | 92.52% | -1.17 | reject |
| 6 | aspp_1_6_12_18_24 | aspp_rates | 1→3 | 18% | 18 | 24 |
| 6 | aspp_1_2_4_8_16 | aspp_rates | 1→3 | 18% | 8 | 16 |

## 有改进的实验

  - confirm_lr_0p02: mIoU=93.69% (gain=3.69)

## 实验目录索引

- `01_confirm_lr_0p02_bs32/` — Step1 confirm: lr=0.02 full 800itr
- `01_lr_0p001_bs32/` — Step1: lr=0.001 bs=32 screen 800itr
- `01_lr_0p005_bs32/` — Step1: lr=0.005 bs=32 screen 800itr
- `01_lr_0p01_bs32/` — Step1: lr=0.01 bs=32 screen 800itr
- `01_lr_0p02_bs32/` — Step1: lr=0.02 bs=32 screen 800itr
- `02_warmup_300/` — Step2: warmup=300 on lr=0.02
- `03_aug_vflip/` — Step3.1: add vertical flip
- `04_aug_rot30/` — Step3.2: add rotation ±30°
- `05_lambda_0p2/` — Step4: λ_edge=0.2
- `05_lambda_0p3/` — Step4: λ_edge=0.3
- `05_lambda_0p4/` — Step4: λ_edge=0.4
- `06_wd_5e4/` — Step5: weight_decay=5e-4
- `07_aspp_1_2_4_8_16/` — Step6: DenseASPP rates=(1,2,4,8,16)
- `07_aspp_1_6_12_18_24/` — Step6: DenseASPP rates=(1,6,12,18,24)

## 推荐后续方向

1. 在最终 best 配置上跑更长训练 (3200+ iters) 验证天花板
2. 尝试 CosineAnnealing 调度器
3. CutMix / MixUp 数据增强
4. 更大 backbone (ResNet-101 / ConvNeXt-Tiny)
5. boundary_width 微调 (3 vs 5 vs 7)
6. Dropout 率调整 (0.1 vs 0.3)
7. 多种 edge_loss_type 比较 (bce vs bce_dice)
