# Result Summary: 02_warmup_300

## Purpose
Step2: warmup=300 on lr=0.02

## Changed Parameter
- **warmup_iters**: 0 → 300

## Best Validation Results
| Metric | Value |
|--------|-------|
| **mIoU** | **90.26%** |
| mAcc | 93.29% |
| OA | 98.61% |
| Abuth IoU | 80.72% |
| Cotton IoU | 91.27% |
| Best Iter | 400 |
| Best Epoch | 25 |

## Confusion
- cotton→abuth: 0.39%
- abuth→cotton: 7.01%

## Files
- `train.log` — full training log
- `metrics.csv` — parsed validation metrics
- `curves.png` — loss/mIoU/per-class curves
- `config_snapshot.json` — experiment config
- `run.sh` — launch script
