# Next Actions

## Current Best
- exp: `exp_spatial_cbam_saff_boundaryaux_itr1600`
- mIoU: 94.66%
- bs=16, lr=0.01

## Next Step
- Step 1: LR screening with bs=32
- Candidates: 0.001, 0.005, 0.01, 0.02
- Run length: 800 iters, val_interval=100
