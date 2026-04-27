实验名称: exp_spatial_cbam_saff_boundaryaux_cottonweed_strong_itr10000
时间: 2026-04-24

数据集:
- /root/autodl-tmp/dataset/cottonweed_strong
- train: 2000 (cotton=1000, abuth=1000)
- val: 74
- test: 74

配置来源:
- 参考日志1: workdirs/train_exp_spatial_cbam_saff_boundaryaux_continue50_es5_step.log
- 参考日志2: workdirs/train_exp_spatial_cbam_saff_boundaryaux_itr1600.log

本次关键参数:
- model=deeplabv3plus_mobilenet
- attention_type=spatial_cbam
- use_saff=true
- enable_boundary_aux=true
- edge_loss_type=bce
- lambda_edge=0.1
- boundary_width=5
- total_itrs=10000
- lr=0.01
- lr_policy=step
- step_size=1000
- weight_decay=5e-4
- batch_size=8
- val_batch_size=2

启动脚本:
- /root/autodl-tmp/CODE/DeepLabV3Plus-Pytorch-master/DeepLabV3Plus-Pytorch-masterPLUS/1cottonweed-strong/train_cottonweed_strong_itr10000_nohup.sh

运行日志:
- /root/autodl-tmp/CODE/DeepLabV3Plus-Pytorch-master/DeepLabV3Plus-Pytorch-masterPLUS/1cottonweed-strong/train.log

运行目录:
- /root/autodl-tmp/CODE/DeepLabV3Plus-Pytorch-master/DeepLabV3Plus-Pytorch-masterPLUS/1cottonweed-strong/workdirs/exp_spatial_cbam_saff_boundaryaux_cottonweed_strong_itr10000

问题修复记录:
1) 首次启动报错: val/masks_trainid 不存在
   - 处理: 使用 --cottonweed_mask_dir masks
2) 二次启动报错: CUDA OOM（另有大进程占用 + 批量较大）
   - 处理: 结束占用进程 PID 142862；batch_size 16->8, val_batch_size 4->2

待训练结束后补充:
- 最优指标（mIoU, IoU/Acc/F1分项, aAcc）
- 参数量 / GFLOPs / FPS
- 最优权重路径
- 曲线图与结果图路径
