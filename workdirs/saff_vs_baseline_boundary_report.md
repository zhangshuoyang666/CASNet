# SAFF vs Baseline (cottonweed)

| Experiment | Best Iter | Best Epoch | best mIoU | best mPA | best mF1 | cotton IoU | abuth IoU | cotton->abuth | abuth->cotton |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exp_spatial_cbam_boundaryaux_best_itr1600 | 1600 | 50 | 94.55% | 96.69% | 97.16% | 89.97% | 94.74% | 0.50% | 0.04% |
| exp_spatial_cbam_saff_boundaryaux_itr1600 | 1216 | 38 | 94.66% | 97.37% | 97.23% | 90.86% | 94.21% | 0.14% | 0.08% |

## SAFF Gain (SAFF - Baseline)
- mIoU: 0.10%
- mPA: 0.68%
- mF1: 0.06%
- cotton IoU: 0.89%
- abuth IoU: -0.53%
- cotton->abuth confusion: -0.36%
- abuth->cotton confusion: 0.03%
