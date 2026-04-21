# Mainline-A Result Summary

## Validation Metrics
- best mIoU: 89.08 @ iter 896 (epoch 14)
- best mAcc: 93.94
- best aAcc: 98.36
- best F1: 94.05
- best class IoU (bg/abuth/cotton): 98.62/79.79/88.82

## Last Validation Point
- last mIoU: 87.73 @ iter 1056 (epoch 17)
- last mAcc: 94.79
- last aAcc: 98.03
- last F1: 93.24

## Complexity & Speed
- parameters: 6715624 (6.7156 M)
- GFLOPs (513x513, bs=1): 23.5734
- FPS (513x513, bs=1): 275.5216
- benchmark device: cuda

## Artifacts
- checkpoint used: `/root/autodl-tmp/CODE/DeepLabV3Plus-Pytorch-master/DeepLabV3Plus-Pytorch-master/BEST-Last/bestcfg_nonbaseline_bs8_lr0p005000_es5_md0p0001_20260419_120030/checkpoints/best_deeplabv3plus_mobilenet_cottonweed_os16.pth`
- train log: `train.log`
- validation table: `metrics.tsv`
- parsed records: `metrics.json`
- curve image: `curves.png`
- model stats: `model_stats.json`
