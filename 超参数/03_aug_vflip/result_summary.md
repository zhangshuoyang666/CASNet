# Result Summary: 03_aug_vflip

## Purpose
Step3.1: add vertical flip

## Changed Parameter
- **aug_vflip**: false → true

## Best Validation Results
| Metric | Value |
|--------|-------|
| **mIoU** | **92.03%** |
| mAcc | 96.06% |
| OA | 98.68% |
| Abuth IoU | 86.47% |
| Cotton IoU | 90.91% |
| Best Iter | 300 |
| Best Epoch | 19 |

## Confusion
- cotton→abuth: 0.34%
- abuth→cotton: 4.12%

## Files
- `train.log` — full training log
- `metrics.csv` — parsed validation metrics
- `curves.png` — loss/mIoU/per-class curves
- `config_snapshot.json` — experiment config
- `run.sh` — launch script
