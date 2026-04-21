# Result Summary: 07_aspp_1_6_12_18_24

## Purpose
Step6: DenseASPP rates=(1,6,12,18,24)

## Changed Parameter
- **aspp_rates**: 1,3,6,12,18(standard) → 1,6,12,18,24(dense)

## Best Validation Results
| Metric | Value |
|--------|-------|
| **mIoU** | **93.69%** |
| mAcc | 96.37% |
| OA | 98.94% |
| Abuth IoU | 88.36% |
| Cotton IoU | 93.84% |
| Best Iter | 400 |
| Best Epoch | 25 |

## Confusion
- cotton→abuth: 0.23%
- abuth→cotton: 1.04%

## Files
- `train.log` — full training log
- `metrics.csv` — parsed validation metrics
- `curves.png` — loss/mIoU/per-class curves
- `config_snapshot.json` — experiment config
- `run.sh` — launch script
