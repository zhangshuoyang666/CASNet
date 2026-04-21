# Result Summary: 05_lambda_0p4

## Purpose
Step4: λ_edge=0.4

## Changed Parameter
- **lambda_edge**: 0.1 → 0.4

## Best Validation Results
| Metric | Value |
|--------|-------|
| **mIoU** | **91.73%** |
| mAcc | 94.86% |
| OA | 98.76% |
| Abuth IoU | 84.15% |
| Cotton IoU | 92.18% |
| Best Iter | 250 |
| Best Epoch | 16 |

## Confusion
- cotton→abuth: 0.51%
- abuth→cotton: 4.74%

## Files
- `train.log` — full training log
- `metrics.csv` — parsed validation metrics
- `curves.png` — loss/mIoU/per-class curves
- `config_snapshot.json` — experiment config
- `run.sh` — launch script
