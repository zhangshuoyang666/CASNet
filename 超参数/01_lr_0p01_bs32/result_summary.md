# Result Summary: 01_lr_0p01_bs32

## Purpose
Step1: lr=0.01 bs=32 screen 800itr

## Changed Parameter
- **lr**: 0.01 → 0.01

## Best Validation Results
| Metric | Value |
|--------|-------|
| **mIoU** | **92.40%** |
| mAcc | 95.35% |
| OA | 98.83% |
| Abuth IoU | 85.64% |
| Cotton IoU | 92.67% |
| Best Iter | 600 |
| Best Epoch | 38 |

## Confusion
- cotton→abuth: 0.29%
- abuth→cotton: 3.99%

## Files
- `train.log` — full training log
- `metrics.csv` — parsed validation metrics
- `curves.png` — loss/mIoU/per-class curves
- `config_snapshot.json` — experiment config
- `run.sh` — launch script
