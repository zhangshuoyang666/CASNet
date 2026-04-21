# CottonWeedV4 Experiment Report

## Best Validation
- mIoU: 0.806524
- aAcc: 0.986546
- mAcc: 0.867526
- iter/epoch: 6758/16

### Per-Class Metrics (Best)
| class | IoU | Acc | F1 |
|---|---:|---:|---:|
| background | 0.990268 | 0.996870 | 0.995110 |
| cotton | 0.943482 | 0.963576 | 0.970919 |
| abuth | 0.831492 | 0.924585 | 0.907994 |
| canger | 0.772306 | 0.890538 | 0.871527 |
| machixian | 0.841867 | 0.926977 | 0.914145 |
| longkui | 0.924536 | 0.946215 | 0.960789 |
| tianxuanhua | 0.440951 | 0.456729 | 0.612028 |
| niujincao | 0.707288 | 0.834719 | 0.828552 |

## Last Validation
- mIoU: 0.773667
- aAcc: 0.983560
- mAcc: 0.838952
- iter/epoch: 8502/20

## Artifacts
- train log: `train.log`
- metric table: `metrics.tsv`
- curves: `loss_and_metrics.png`
- summary json: `summary.json`
