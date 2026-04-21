# Result Summary: 01_lr_0p005_bs32

## Purpose
Step1: lr=0.005 bs=32 screen 800itr

## Changed Parameter
- **lr**: 0.01 → 0.005

## Best Validation Results
| Metric | Value |
|--------|-------|
| **mIoU** | **90.68%** |
| mAcc | 94.40% |
| OA | 98.60% |
| Abuth IoU | 82.52% |
| Cotton IoU | 90.75% |
| Best Iter | 800 |
| Best Epoch | 50 |

## Confusion
- cotton→abuth: 0.99%
- abuth→cotton: 5.43%

## Files
- `train.log` — full training log
- `metrics.csv` — parsed validation metrics
- `curves.png` — loss/mIoU/per-class curves
- `config_snapshot.json` — experiment config
- `run.sh` — launch script
