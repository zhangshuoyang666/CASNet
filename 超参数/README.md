# SA-BNet Serial Hyperparameter Optimization

## Baseline
- **Model**: DeepLabV3+ MobileNet + SpatialCBAM + SAFF + BoundaryAux
- **Best mIoU**: 94.66% (exp_spatial_cbam_saff_boundaryaux_itr1600 @ iter 1216)
- **Config**: bs=16, lr=0.01, wd=1e-4, λ_edge=0.1, boundary_width=5

## Optimization Strategy
Serial single-variable optimization:
1. **Step 1**: Learning rate (0.001, 0.005, 0.01, 0.02) with bs=32
2. **Step 2**: Warmup (0 vs 300 iters) on best LR
3. **Step 3.1**: Vertical flip augmentation
4. **Step 3.2**: Rotation ±30° augmentation
5. **Step 4**: λ_edge (0.2, 0.3, 0.4) 
6. **Step 5**: weight_decay (1e-4 vs 5e-4)
7. **Step 6**: ASPP rates variants

## Protocol
- Short screening: 800 iterations per candidate
- Confirmation: 1600 iterations for winners
- Val interval: 100 iterations (8 checkpoints per screen)
- Decision: better (>0.15% gain), marginal (0~0.15%), worse
- Only promote if better or marginal; reject if worse

## Files
- `best_config.json` — current best configuration
- `experiment_tracker.csv` — all experiment results
- `auto_optimize.sh` — main automation script
- `parse_and_plot.py` — log parser and curve generator
- `compare_experiments.py` — multi-experiment comparison plotter
- `final_report.md` — generated after all steps complete
