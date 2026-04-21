# 最优实验报告：exp_spatial_cbam_boundaryaux_best_itr1600

## 1. 实验目的
在 cottonweed 三分类分割任务（background / abuth / cotton）上，验证 `spatial_cbam` 注意力 + 边界辅助分支（Boundary Auxiliary）对细粒度类别区分能力的提升效果，重点关注 `abuth` 与 `cotton` 两类指标及相互混淆。

## 2. 最优实验配置
- 实验名：`exp_spatial_cbam_boundaryaux_best_itr1600`
- 数据集：`cottonweed`（Train=513，Val=146）
- 模型：`deeplabv3plus_mobilenet`
- 注意力：`spatial_cbam`
- 边界辅助分支：开启
- 边界损失：`edge_loss_type=bce`
- 边界损失权重：`lambda_edge=0.1`
- 边界宽度：`boundary_width=5`
- 训练总迭代：`1600`
- 训练总轮次：`50 epoch`

## 3. 核心结果（最优轮次）
### 3.1 全局最优点（按 mIoU）
- 最优 mIoU：`0.945542`（epoch `50` / iter `1600`）
- 该点分类别 IoU：
  - background IoU：`0.989551`
  - abuth IoU：`0.947352`
  - cotton IoU：`0.899724`

### 3.2 关注类别最优点
- abuth 最优 IoU：`0.947793`（epoch `46` / iter `1472`）
- cotton 最优 IoU：`0.899724`（epoch `50` / iter `1600`）

## 4. 相似类别混淆分析
基于 `similarity_confusion.tsv`（值越低越好）：
- 在最终轮次（epoch 50）：
  - cotton -> abuth：`0.004966`
  - abuth -> cotton：`0.000446`
- 训练中最小 `cotton -> abuth` 出现在 epoch 36（iter 1152），为 `0.001400`。

说明：在后期收敛阶段，`cotton` 与 `abuth` 的双向误分率均维持在较低水平，表明边界辅助分支对细粒度区分有稳定贡献。

## 5. 对比实验（同注意力、无边界辅助）
对比对象：`exp_spatial_cbam_e50_itr1600`（`spatial_cbam`，无边界辅助分支）

| 指标 | 最优实验（本报告） | 对比实验 | 变化量 |
|---|---:|---:|---:|
| 最优 mIoU | 0.945542 | 0.936318 | +0.009224 |
| abuth 最优 IoU | 0.947793 | 0.945095 | +0.002698 |
| cotton 最优 IoU | 0.899724 | 0.876635 | +0.023089 |
| 末轮 cotton->abuth | 0.004966 | 0.008174 | -0.003208 |
| 末轮 abuth->cotton | 0.000446 | 0.001053 | -0.000607 |

结论：在保持 `spatial_cbam` 注意力不变时，引入边界辅助分支后，整体 mIoU、`abuth` IoU、`cotton` IoU 均提升，其中 `cotton` 类提升最明显；同时两类间双向混淆进一步下降。

## 6. 最终结论
- 该阶段最佳方案为：`spatial_cbam + boundary auxiliary(bce, lambda=0.1, bw=5)`。
- 你关心的问题直接回答：
  - 加的注意力：`spatial_cbam`
  - `abuth` 最优：第 `46` 轮（iter `1472`）
  - `cotton` 最优：第 `50` 轮（iter `1600`）
  - 实验总轮次：`50` 轮
