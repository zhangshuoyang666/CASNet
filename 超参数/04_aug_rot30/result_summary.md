# Result Summary: 04_aug_rot30

## Purpose
Step3.2: add rotation ±30°

## Changed Parameter
- **aug_rotation**: 0 → 30

## Best Validation Results
| Metric | Value |
|--------|-------|
| **mIoU** | **90.54%** |
| mAcc | 94.29% |
| OA | 98.60% |
| Abuth IoU | 82.37% |
| Cotton IoU | 90.47% |
| Best Iter | 400 |
| Best Epoch | 25 |

## Confusion
- cotton→abuth: 0.82%
- abuth→cotton: 6.44%

## Files
- `train.log` — full training log
- `metrics.csv` — parsed validation metrics
- `curves.png` — loss/mIoU/per-class curves
- `config_snapshot.json` — experiment config
- `run.sh` — launch script
