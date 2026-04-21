# Result Summary: 07_aspp_1_2_4_8_16

## Purpose
Step6: DenseASPP rates=(1,2,4,8,16)

## Changed Parameter
- **aspp_rates**: 1,3,6,12,18(standard) → 1,2,4,8,16(dense)

## Best Validation Results
| Metric | Value |
|--------|-------|
| **mIoU** | **93.83%** |
| mAcc | 96.23% |
| OA | 98.97% |
| Abuth IoU | 88.71% |
| Cotton IoU | 93.87% |
| Best Iter | 400 |
| Best Epoch | 25 |

## Confusion
- cotton→abuth: 0.11%
- abuth→cotton: 1.58%

## Files
- `train.log` — full training log
- `metrics.csv` — parsed validation metrics
- `curves.png` — loss/mIoU/per-class curves
- `config_snapshot.json` — experiment config
- `run.sh` — launch script
