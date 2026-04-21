# Result Summary: 01_confirm_lr_0p02_bs32

## Purpose
Step1 confirm: lr=0.02 full 800itr

## Changed Parameter
- **lr**: 0.01 → 0.02

## Best Validation Results
| Metric | Value |
|--------|-------|
| **mIoU** | **93.69%** |
| mAcc | 96.38% |
| OA | 99.00% |
| Abuth IoU | 87.93% |
| Cotton IoU | 94.15% |
| Best Iter | 450 |
| Best Epoch | 29 |

## Confusion
- cotton→abuth: 0.19%
- abuth→cotton: 2.3%

## Files
- `train.log` — full training log
- `metrics.csv` — parsed validation metrics
- `curves.png` — loss/mIoU/per-class curves
- `config_snapshot.json` — experiment config
- `run.sh` — launch script
