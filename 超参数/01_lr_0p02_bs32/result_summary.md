# Result Summary: 01_lr_0p02_bs32

## Purpose
Step1: lr=0.02 bs=32 screen 800itr

## Changed Parameter
- **lr**: 0.01 → 0.02

## Best Validation Results
| Metric | Value |
|--------|-------|
| **mIoU** | **93.05%** |
| mAcc | 96.15% |
| OA | 98.91% |
| Abuth IoU | 86.79% |
| Cotton IoU | 93.42% |
| Best Iter | 500 |
| Best Epoch | 32 |

## Confusion
- cotton→abuth: 0.47%
- abuth→cotton: 2.71%

## Files
- `train.log` — full training log
- `metrics.csv` — parsed validation metrics
- `curves.png` — loss/mIoU/per-class curves
- `config_snapshot.json` — experiment config
- `run.sh` — launch script
