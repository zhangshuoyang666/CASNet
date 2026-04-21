# Result Summary: 06_wd_5e4

## Purpose
Step5: weight_decay=5e-4

## Changed Parameter
- **weight_decay**: 1e-4 → 5e-4

## Best Validation Results
| Metric | Value |
|--------|-------|
| **mIoU** | **92.52%** |
| mAcc | 95.55% |
| OA | 98.84% |
| Abuth IoU | 85.73% |
| Cotton IoU | 92.96% |
| Best Iter | 250 |
| Best Epoch | 16 |

## Confusion
- cotton→abuth: 0.58%
- abuth→cotton: 2.92%

## Files
- `train.log` — full training log
- `metrics.csv` — parsed validation metrics
- `curves.png` — loss/mIoU/per-class curves
- `config_snapshot.json` — experiment config
- `run.sh` — launch script
