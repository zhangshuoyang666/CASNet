# Result Summary: 05_lambda_0p2

## Purpose
Step4: λ_edge=0.2

## Changed Parameter
- **lambda_edge**: 0.1 → 0.2

## Best Validation Results
| Metric | Value |
|--------|-------|
| **mIoU** | **92.01%** |
| mAcc | 95.07% |
| OA | 98.78% |
| Abuth IoU | 84.72% |
| Cotton IoU | 92.46% |
| Best Iter | 250 |
| Best Epoch | 16 |

## Confusion
- cotton→abuth: 0.27%
- abuth→cotton: 4.39%

## Files
- `train.log` — full training log
- `metrics.csv` — parsed validation metrics
- `curves.png` — loss/mIoU/per-class curves
- `config_snapshot.json` — experiment config
- `run.sh` — launch script
