# Result Summary: 05_lambda_0p3

## Purpose
Step4: λ_edge=0.3

## Changed Parameter
- **lambda_edge**: 0.1 → 0.3

## Best Validation Results
| Metric | Value |
|--------|-------|
| **mIoU** | **91.70%** |
| mAcc | 95.09% |
| OA | 98.75% |
| Abuth IoU | 84.08% |
| Cotton IoU | 92.15% |
| Best Iter | 250 |
| Best Epoch | 16 |

## Confusion
- cotton→abuth: 0.42%
- abuth→cotton: 4.95%

## Files
- `train.log` — full training log
- `metrics.csv` — parsed validation metrics
- `curves.png` — loss/mIoU/per-class curves
- `config_snapshot.json` — experiment config
- `run.sh` — launch script
