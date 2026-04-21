# Result Summary: 01_lr_0p001_bs32

## Purpose
Step1: lr=0.001 bs=32 screen 800itr

## Changed Parameter
- **lr**: 0.01 → 0.001

## Best Validation Results
| Metric | Value |
|--------|-------|
| **mIoU** | **74.95%** |
| mAcc | 82.16% |
| OA | 96.47% |
| Abuth IoU | 52.61% |
| Cotton IoU | 74.37% |
| Best Iter | 800 |
| Best Epoch | 50 |

## Confusion
- cotton→abuth: 6.17%
- abuth→cotton: 24.56%

## Files
- `train.log` — full training log
- `metrics.csv` — parsed validation metrics
- `curves.png` — loss/mIoU/per-class curves
- `config_snapshot.json` — experiment config
- `run.sh` — launch script
