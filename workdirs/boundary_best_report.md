# Boundary Auxiliary Branch Experiment Report

## Overview

This report summarizes all boundary-auxiliary related runs on `cottonweed` in the current project and highlights the best configuration.

## Consolidated Results

| Stage | Experiment | Core Config | aAcc | mIoU | abuth IoU | cotton IoU | cotton->abuth |
|---|---|---|---:|---:|---:|---:|---:|
| quick | baseline_quick | no boundary aux | 0.875778 | 0.340187 | 0.130180 | 0.011041 | 0.128000 |
| quick | boundaryaux_old_quick | bce+dice, lambda=0.3, bw=3 | 0.919800 | 0.530610 | 0.516071 | 0.144522 | 0.333800 |
| 5ep | baseline_5ep | no boundary aux | 0.977239 | 0.847813 | 0.842186 | 0.719893 | 0.068584 |
| 5ep | boundaryaux_old_5ep | bce+dice, lambda=0.3, bw=3 | 0.971019 | 0.783909 | 0.793861 | 0.575304 | 0.246583 |
| 30ep | boundaryaux_bce_lambda005_bw3_e30 | bce, lambda=0.05, bw=3 | 0.986716 | 0.920015 | 0.918310 | 0.855526 | 0.029237 |
| 30ep | boundaryaux_bce_lambda01_bw3_e30 | bce, lambda=0.1, bw=3 | 0.988113 | 0.926227 | 0.931976 | 0.859135 | 0.023541 |
| 30ep | boundaryaux_bce_lambda01_bw1_e30 | bce, lambda=0.1, bw=1 | 0.988151 | 0.927755 | 0.931122 | 0.864667 | 0.020202 |
| 30ep | boundaryaux_bce_lambda01_bw5_e30_rerun | bce, lambda=0.1, bw=5 | **0.988692** | **0.932283** | **0.938931** | **0.870398** | **0.008601** |

## Best Run

- **Experiment**: `boundaryaux_bce_lambda01_bw5_e30_rerun`
- **Run directory**: `workdirs/boundaryaux_bce_lambda01_bw5_e30_rerun`
- **Log file**: `workdirs/nohup_boundaryaux_bce_lambda01_bw5_e30_rerun.log`

### Best metrics

- aAcc: `0.988692`
- mIoU: `0.932283`
- background IoU: `0.987519`
- abuth IoU: `0.938931`
- cotton IoU: `0.870398`
- abuth Acc: `0.971474`
- cotton Acc: `0.900020`
- cotton->abuth confusion: `0.008601`
- abuth->cotton confusion: `0.000837`

## Best Configuration

- `enable_boundary_aux=True`
- `edge_loss_type=bce`
- `lambda_edge=0.1`
- `boundary_width=5`
- `model=deeplabv3plus_mobilenet`
- `dataset=cottonweed`
- `output_stride=16`
- `batch_size=8`
- `val_batch_size=1`
- `crop_size=513`
- `total_itrs=1920` (about 30 epochs)

