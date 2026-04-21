# SA-BNet 串行超参数优化实验

## 策略

基于当前最优基线 `exp_spatial_cbam_saff_boundaryaux_itr1600` (mIoU=94.66%, bs=16, lr=0.01)，
以 batch_size=32 为起点，执行**串行单变量优化**。

## 优化顺序

| Step | 超参数 | 优先级 | 候选值 |
|------|--------|--------|--------|
| 1 | 学习率 lr | 高 | 0.001, 0.005, 0.01, 0.02 |
| 2 | Warmup | 中 | 0, 300 iters |
| 3.1 | 数据增强 VFlip | 中 | off, on |
| 3.2 | 数据增强 Rotation | 中 | 0, ±30° |
| 4 | λ_edge | 中 | 0.1, 0.2, 0.3, 0.4 |
| 5 | weight_decay | 中 | 1e-4, 5e-4 |
| 6 | ASPP rates | 中 | (1,3,6,12,18), (1,6,12,18,24), (1,2,4,8,16) |

## 筛选原则

- 短程筛选：800 iters, val_interval=100
- 确认阶段：1600 iters（与基线同等长度）
- 决策：reject / promote_to_confirm / update_best / marginal_gain / fail_fast

## 文件结构

```
best-model-超参/
  README.md
  best_config.json
  experiment_tracker.csv
  next_actions.md
  parse_and_plot.py
  01_lr_0p001_bs32/
  01_lr_0p005_bs32/
  ...
```
