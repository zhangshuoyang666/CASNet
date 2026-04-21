from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
from datetime import datetime

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, CottonWeedSegmentation
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt


def unpack_model_outputs(outputs):
    if isinstance(outputs, dict):
        return outputs["out"], outputs.get("edge", None)
    return outputs, None


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes', 'cottonweed'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")
    parser.add_argument("--cottonweed_mask_dir", type=str, default='masks_trainid',
                        help='mask directory under each split when dataset=cottonweed')
    parser.add_argument("--class_names", type=str, default='',
                        help='optional txt file, one class name per line (including background)')
    parser.add_argument("--work_dir", type=str, default='./workdirs',
                        help='directory to save checkpoints and metric logs')
    parser.add_argument("--exp_name", type=str, default='',
                        help='experiment name under work_dir')

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    parser.add_argument("--attention_type", type=str, default='none',
                        choices=['none', 'channel', 'spatial', 'cbam', 'cbam_light', 'spatial_cbam', 'se', 'ca', 'coordinate'],
                        help='attention module before ASPP 1x1 projection')
    parser.add_argument("--enable_fg_fusion", action='store_true', default=False,
                        help="enable FineGrainedFusionModule")
    parser.add_argument("--enable_texture_enhance", action='store_true', default=False,
                        help="enable TextureEnhanceModule before ASPP")
    parser.add_argument("--aspp_variant", type=str, default='standard',
                        choices=['standard', 'dense', 'deform'],
                        help='ASPP variant: baseline standard / dense / deform-like')
    parser.add_argument("--dense_aspp_rates", type=str, default='1,3,6,12,18',
                        help='dense ASPP rates, comma separated')
    parser.add_argument("--enable_decoder_detail", action='store_true', default=False,
                        help='enable low-level detail compensation path in decoder')
    parser.add_argument("--enable_boundary_aux", action='store_true', default=False,
                        help='enable boundary auxiliary branch from decoder features')
    parser.add_argument("--use_saff", action='store_true', default=False,
                        help='enable SAFF fusion module between ASPP and decoder')
    parser.add_argument("--saff_f1_source", type=str, default='mid',
                        choices=['low', 'mid', 'high'],
                        help='SAFF F1 source feature level')
    parser.add_argument("--saff_f2_source", type=str, default='aspp',
                        choices=['aspp', 'high'],
                        help='SAFF F2 source feature branch')
    parser.add_argument("--boundary_width", type=int, default=3,
                        help='boundary width used for auto-generated edge targets')
    parser.add_argument("--lambda_edge", type=float, default=0.3,
                        help='edge loss weight in total loss')
    parser.add_argument("--use_seg_dice", action='store_true', default=False,
                        help='use CE/Focal + Dice as segmentation loss')
    parser.add_argument("--edge_loss_type", type=str, default='bce_dice',
                        choices=['bce', 'bce_dice'], help='edge loss composition')
    parser.add_argument("--save_edge_results", action='store_true', default=False,
                        help='save predicted boundary maps during validation')
    parser.add_argument("--abuth_class_id", type=int, default=1,
                        help='class id for abuth in cottonweed dataset')
    parser.add_argument("--cotton_class_id", type=int, default=2,
                        help='class id for cotton in cottonweed dataset')

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=30000,
                        help="max training iterations; <=0 means no hard limit")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--enable_early_stop", action='store_true', default=False,
                        help="enable early stopping on validation metric")
    parser.add_argument("--early_stop_patience", type=int, default=12,
                        help="number of validation rounds without improvement before stop")
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4,
                        help="minimum metric improvement to reset early-stop counter")
    parser.add_argument("--early_stop_metric", type=str, default='Mean IoU',
                        choices=['Mean IoU', 'Overall Acc', 'Mean Acc', 'FreqW Acc'],
                        help="validation metric used by early stopping")
    parser.add_argument("--min_itrs_before_early_stop", type=int, default=2000,
                        help="minimum iterations before early stopping can trigger")
    parser.add_argument("--poly_max_itrs_for_early_stop", type=int, default=100000,
                        help="virtual max itrs for PolyLR when total_itrs<=0")
    parser.add_argument("--warmup_iters", type=int, default=0,
                        help="linear LR warm-up iterations (0 = disabled)")
    parser.add_argument("--aug_vflip", action='store_true', default=False,
                        help="add RandomVerticalFlip to training augmentation")
    parser.add_argument("--aug_rotation", type=int, default=0,
                        help="add RandomRotation with given max degrees (0 = disabled)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'voc':
        train_transform = et.ExtCompose([
            # et.ExtResize(size=opts.crop_size),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                    image_set='train', download=opts.download, transform=train_transform)
        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                  image_set='val', download=False, transform=val_transform)

    if opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Cityscapes(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root,
                             split='val', transform=val_transform)
    if opts.dataset == 'cottonweed':
        _cotton_aug = [
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            et.ExtRandomHorizontalFlip(),
        ]
        if getattr(opts, 'aug_vflip', False):
            _cotton_aug.append(et.ExtRandomVerticalFlip())
        if getattr(opts, 'aug_rotation', 0) > 0:
            _cotton_aug.append(et.ExtRandomRotation(degrees=opts.aug_rotation))
        _cotton_aug += [
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ]
        train_transform = et.ExtCompose(_cotton_aug)
        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        train_dst = CottonWeedSegmentation(root=opts.data_root,
                                           split='train',
                                           mask_dir=opts.cottonweed_mask_dir,
                                           transform=train_transform)
        val_dst = CottonWeedSegmentation(root=opts.data_root,
                                         split='val',
                                         mask_dir=opts.cottonweed_mask_dir,
                                         transform=val_transform)
    return train_dst, val_dst


def load_class_names(opts):
    if opts.class_names:
        if not os.path.isfile(opts.class_names):
            raise FileNotFoundError("class names file not found: {}".format(opts.class_names))
        with open(opts.class_names, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        if len(names) != opts.num_classes:
            raise ValueError(
                "class names count ({}) does not match num_classes ({})".format(len(names), opts.num_classes)
            )
        return names

    if opts.dataset.lower() == 'cottonweed':
        if opts.num_classes == 8:
            return ['background', 'cotton', 'abuth', 'canger', 'machixian', 'longkui', 'tianxuanhua', 'niujincao']
        if opts.num_classes == 3:
            return ['background', 'abuth', 'cotton']
        return ['class_{}'.format(i) for i in range(opts.num_classes)]
    if opts.dataset.lower() == 'voc':
        return ['class_{}'.format(i) for i in range(opts.num_classes)]
    if opts.dataset.lower() == 'cityscapes':
        return ['class_{}'.format(i) for i in range(opts.num_classes)]
    return ['class_{}'.format(i) for i in range(opts.num_classes)]


def parse_int_list(text):
    values = [v.strip() for v in str(text).split(",") if v.strip()]
    if not values:
        raise ValueError("empty integer list")
    return tuple(int(v) for v in values)


def compute_detailed_metrics(confusion_matrix, class_names):
    cm = confusion_matrix.astype(np.float64)
    eps = 1e-12

    tp = np.diag(cm)
    gt = cm.sum(axis=1)
    pred = cm.sum(axis=0)

    class_iou = tp / np.maximum(gt + pred - tp, eps)
    class_acc = tp / np.maximum(gt, eps)
    class_precision = tp / np.maximum(pred, eps)
    class_f1 = 2.0 * class_precision * class_acc / np.maximum(class_precision + class_acc, eps)
    aacc = tp.sum() / np.maximum(cm.sum(), eps)
    miou = np.nanmean(class_iou)
    macc = np.nanmean(class_acc)
    mf1 = np.nanmean(class_f1)

    rows = []
    for idx, class_name in enumerate(class_names):
        rows.append({
            "class": class_name,
            "IoU": class_iou[idx],
            "Acc": class_acc[idx],
            "F1": class_f1[idx],
        })

    return rows, aacc, miou, macc, mf1


def format_metric_table(rows, aacc, miou, macc, mf1, cur_itrs, cur_epochs):
    header = "{:<14}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<8}".format(
        "class", "IoU", "Acc", "F1", "aAcc", "mIoU", "mAcc", "mF1", "Iter", "epoch"
    )
    lines = [header]
    for i, row in enumerate(rows):
        aacc_text = "{:.2f}".format(aacc * 100.0) if i == 0 else ""
        miou_text = "{:.2f}".format(miou * 100.0) if i == 0 else ""
        macc_text = "{:.2f}".format(macc * 100.0) if i == 0 else ""
        mf1_text = "{:.2f}".format(mf1 * 100.0) if i == 0 else ""
        iter_text = str(cur_itrs) if i == 0 else ""
        epoch_text = str(cur_epochs) if i == 0 else ""
        lines.append(
            "{:<14}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<8}".format(
                row["class"],
                "{:.2f}".format(row["IoU"] * 100.0),
                "{:.2f}".format(row["Acc"] * 100.0),
                "{:.2f}".format(row["F1"] * 100.0),
                aacc_text,
                miou_text,
                macc_text,
                mf1_text,
                iter_text,
                epoch_text,
            )
        )
    return "\n".join(lines)


def append_metric_tsv(metrics_tsv_path, rows, aacc, miou, macc, mf1, cur_itrs, cur_epochs):
    with open(metrics_tsv_path, "a", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            f.write(
                "{class_name}\t{iou:.6f}\t{acc:.6f}\t{f1:.6f}\t{aacc}\t{miou}\t{macc}\t{mf1}\t{itrs}\t{epochs}\n".format(
                    class_name=row["class"],
                    iou=row["IoU"],
                    acc=row["Acc"],
                    f1=row["F1"],
                    aacc="{:.6f}".format(aacc) if i == 0 else "",
                    miou="{:.6f}".format(miou) if i == 0 else "",
                    macc="{:.6f}".format(macc) if i == 0 else "",
                    mf1="{:.6f}".format(mf1) if i == 0 else "",
                    itrs=cur_itrs if i == 0 else "",
                    epochs=cur_epochs if i == 0 else "",
                )
            )


def find_class_id(class_names, target_name, default_id):
    for i, name in enumerate(class_names):
        if name.lower() == target_name.lower():
            return i
    return default_id


def format_similarity_metrics(confusion_matrix, class_names, cotton_id, abuth_id):
    cm = confusion_matrix.astype(np.float64)
    eps = 1e-12
    cotton_name = class_names[cotton_id] if cotton_id < len(class_names) else "cotton"
    abuth_name = class_names[abuth_id] if abuth_id < len(class_names) else "abuth"
    cotton_gt = max(cm[cotton_id].sum(), eps)
    abuth_gt = max(cm[abuth_id].sum(), eps)
    cotton_as_abuth = cm[cotton_id, abuth_id] / cotton_gt
    abuth_as_cotton = cm[abuth_id, cotton_id] / abuth_gt
    lines = [
        "[Similarity-Focus]",
        "  {} -> {} confusion: {:.2f}%".format(cotton_name, abuth_name, cotton_as_abuth * 100.0),
        "  {} -> {} confusion: {:.2f}%".format(abuth_name, cotton_name, abuth_as_cotton * 100.0),
    ]
    return "\n".join(lines), cotton_as_abuth, abuth_as_cotton


def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            seg_logits, edge_logits = unpack_model_outputs(outputs)
            preds = seg_logits.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)
                    if opts.save_edge_results and edge_logits is not None:
                        edge_prob = torch.sigmoid(edge_logits[i]).detach().cpu().numpy()[0]
                        edge_uint8 = np.clip(edge_prob * 255.0, 0, 255).astype(np.uint8)
                        Image.fromarray(edge_uint8).save('results/%d_edge_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples


def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
    elif opts.dataset.lower() == 'cottonweed':
        if opts.num_classes is None:
            if opts.class_names and os.path.isfile(opts.class_names):
                with open(opts.class_names, "r", encoding="utf-8") as f:
                    opts.num_classes = len([line.strip() for line in f if line.strip()])
            else:
                opts.num_classes = 3

    if not opts.exp_name:
        opts.exp_name = "{}_{}_{}_{}".format(
            opts.dataset,
            opts.model,
            opts.attention_type,
            datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    run_dir = os.path.join(opts.work_dir, opts.exp_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    utils.mkdir(opts.work_dir)
    utils.mkdir(run_dir)
    utils.mkdir(ckpt_dir)
    metrics_tsv_path = os.path.join(run_dir, "metrics.tsv")
    if not os.path.isfile(metrics_tsv_path):
        with open(metrics_tsv_path, "w", encoding="utf-8") as f:
            f.write("class\tIoU\tAcc\tF1\taAcc\tmIoU\tmAcc\tmF1\tIter\tepoch\n")
    confusion_tsv_path = os.path.join(run_dir, "confusion.tsv")
    legacy_confusion_tsv_path = os.path.join(run_dir, "similarity_confusion.tsv")
    if not os.path.isfile(confusion_tsv_path):
        with open(confusion_tsv_path, "w", encoding="utf-8") as f:
            f.write("Iter\tepoch\tcotton_to_abuth\tabuth_to_cotton\n")
    if not os.path.isfile(legacy_confusion_tsv_path):
        with open(legacy_confusion_tsv_path, "w", encoding="utf-8") as f:
            f.write("Iter\tepoch\tcotton_to_abuth\tabuth_to_cotton\n")
    print("Run directory: {}".format(run_dir))

    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset == 'voc' and not opts.crop_val:
        opts.val_batch_size = 1

    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2,
        drop_last=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model (all models are 'constructed at network.modeling)
    dense_aspp_rates = parse_int_list(opts.dense_aspp_rates)
    print(
        "[Ablation-Config] attention_type={}, fg_fusion={}, texture_enhance={}, aspp_variant={}, dense_rates={}, decoder_detail={}, use_saff={}, saff_f1_source={}, saff_f2_source={}".format(
            opts.attention_type,
            opts.enable_fg_fusion,
            opts.enable_texture_enhance,
            opts.aspp_variant,
            dense_aspp_rates,
            opts.enable_decoder_detail,
            opts.use_saff,
            opts.saff_f1_source,
            opts.saff_f2_source,
        )
    )
    model = network.modeling.__dict__[opts.model](
        num_classes=opts.num_classes,
        output_stride=opts.output_stride,
        attention_type=opts.attention_type,
        aspp_variant=opts.aspp_variant,
        dense_aspp_rates=dense_aspp_rates,
        enable_fg_fusion=opts.enable_fg_fusion,
        enable_texture_enhance=opts.enable_texture_enhance,
        enable_decoder_detail=opts.enable_decoder_detail,
        enable_boundary_aux=opts.enable_boundary_aux,
        use_saff=opts.use_saff,
        saff_f1_source=opts.saff_f1_source,
        saff_f2_source=opts.saff_f2_source,
    )
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)
    class_names = load_class_names(opts)
    cotton_id = find_class_id(class_names, "cotton", opts.cotton_class_id)
    abuth_id = find_class_id(class_names, "abuth", opts.abuth_class_id)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    poly_total_itrs = opts.total_itrs
    if opts.total_itrs <= 0:
        poly_total_itrs = max(10000, opts.poly_max_itrs_for_early_stop)
    warmup_iters = getattr(opts, 'warmup_iters', 0)
    if opts.lr_policy == 'poly' and warmup_iters > 0:
        scheduler = utils.WarmupPolyLR(optimizer, poly_total_itrs,
                                       warmup_iters=warmup_iters, power=0.9)
    elif opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, poly_total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)
    else:
        scheduler = utils.PolyLR(optimizer, 30000, power=0.9)

    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        seg_primary_criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        seg_primary_criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    seg_dice_criterion = utils.MultiClassDiceLoss(ignore_index=255)
    edge_bce_criterion = nn.BCEWithLogitsLoss()
    edge_dice_criterion = utils.BinaryDiceLoss()

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    # Restore
    best_score = 0.0
    early_stop_best = float("-inf")
    no_improve_evals = 0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        load_ret = model.load_state_dict(checkpoint["model_state"], strict=not opts.enable_boundary_aux)
        if opts.enable_boundary_aux and (
            len(getattr(load_ret, "missing_keys", [])) > 0 or len(getattr(load_ret, "unexpected_keys", [])) > 0
        ):
            print(
                "[Checkpoint] strict=False due to boundary aux. missing_keys={}, unexpected_keys={}".format(
                    len(load_ret.missing_keys), len(load_ret.unexpected_keys)
                )
            )
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    # ==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        return

    interval_loss = 0
    interval_seg_loss = 0
    interval_edge_loss = 0
    while True:  # stop by max itrs or early stopping
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for (images, labels) in train_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            seg_logits, edge_logits = unpack_model_outputs(outputs)

            seg_loss = seg_primary_criterion(seg_logits, labels)
            if opts.use_seg_dice:
                seg_loss = seg_loss + seg_dice_criterion(seg_logits, labels)

            edge_loss = torch.tensor(0.0, device=device)
            if opts.enable_boundary_aux and edge_logits is not None:
                edge_target = utils.generate_boundary_target(
                    labels, num_classes=opts.num_classes, boundary_width=opts.boundary_width, ignore_index=255
                )
                edge_loss = edge_bce_criterion(edge_logits, edge_target)
                if opts.edge_loss_type == 'bce_dice':
                    edge_loss = edge_loss + edge_dice_criterion(edge_logits, edge_target)

            loss = seg_loss + opts.lambda_edge * edge_loss
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            seg_np_loss = seg_loss.detach().cpu().numpy()
            edge_np_loss = edge_loss.detach().cpu().numpy() if torch.is_tensor(edge_loss) else 0.0
            interval_seg_loss += seg_np_loss
            interval_edge_loss += edge_np_loss
            if vis is not None:
                vis.vis_scalar('Loss', cur_itrs, np_loss)
                vis.vis_scalar('Seg Loss', cur_itrs, seg_np_loss)
                if opts.enable_boundary_aux:
                    vis.vis_scalar('Edge Loss', cur_itrs, edge_np_loss)

            if (cur_itrs) % opts.print_interval == 0:
                interval_loss = interval_loss / opts.print_interval
                interval_seg_loss = interval_seg_loss / opts.print_interval
                interval_edge_loss = interval_edge_loss / opts.print_interval
                print(
                    "Epoch {}, Itrs {}/{}, total_loss={:.6f}, seg_loss={:.6f}, edge_loss={:.6f}, "
                    "lambda_edge={}, boundary_width={}, edge_loss_type={}".format(
                        cur_epochs,
                        cur_itrs,
                        opts.total_itrs,
                        interval_loss,
                        interval_seg_loss,
                        interval_edge_loss,
                        opts.lambda_edge,
                        opts.boundary_width,
                        opts.edge_loss_type,
                    )
                )
                interval_loss = 0.0
                interval_seg_loss = 0.0
                interval_edge_loss = 0.0

            if (cur_itrs) % opts.val_interval == 0:
                save_ckpt(os.path.join(ckpt_dir, 'latest_%s_%s_os%d.pth' %
                          (opts.model, opts.dataset, opts.output_stride)))
                print("validation...")
                model.eval()
                val_score, ret_samples = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
                    ret_samples_ids=vis_sample_id)
                print(metrics.to_str(val_score))
                detail_rows, aacc, miou, macc, mf1 = compute_detailed_metrics(metrics.confusion_matrix, class_names)
                print(format_metric_table(detail_rows, aacc, miou, macc, mf1, cur_itrs, cur_epochs))
                similarity_text, cotton_as_abuth, abuth_as_cotton = format_similarity_metrics(
                    metrics.confusion_matrix, class_names, cotton_id, abuth_id
                )
                print(similarity_text)
                append_metric_tsv(metrics_tsv_path, detail_rows, aacc, miou, macc, mf1, cur_itrs, cur_epochs)
                confusion_line = "{}\t{}\t{:.6f}\t{:.6f}\n".format(
                    cur_itrs, cur_epochs, cotton_as_abuth, abuth_as_cotton
                )
                with open(confusion_tsv_path, "a", encoding="utf-8") as f:
                    f.write(confusion_line)
                with open(legacy_confusion_tsv_path, "a", encoding="utf-8") as f:
                    f.write(confusion_line)
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt(os.path.join(ckpt_dir, 'best_%s_%s_os%d.pth' %
                              (opts.model, opts.dataset, opts.output_stride)))

                if opts.enable_early_stop:
                    cur_metric = val_score[opts.early_stop_metric]
                    if cur_metric > early_stop_best + opts.early_stop_min_delta:
                        early_stop_best = cur_metric
                        no_improve_evals = 0
                    else:
                        no_improve_evals += 1
                    print(
                        "[EarlyStop] metric={} current={:.6f} best={:.6f} no_improve={}/{} min_delta={:.6f}".format(
                            opts.early_stop_metric,
                            cur_metric,
                            early_stop_best,
                            no_improve_evals,
                            opts.early_stop_patience,
                            opts.early_stop_min_delta,
                        )
                    )
                    if cur_itrs >= opts.min_itrs_before_early_stop and no_improve_evals >= opts.early_stop_patience:
                        print(
                            "[EarlyStop] Triggered at iter {}, epoch {}. Stop training.".format(
                                cur_itrs, cur_epochs
                            )
                        )
                        return

                if vis is not None:  # visualize validation score and samples
                    vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                    vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                    vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                    for k, (img, target, lbl) in enumerate(ret_samples):
                        img = (denorm(img) * 255).astype(np.uint8)
                        target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                        lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                        concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                        vis.vis_image('Sample %d' % k, concat_img)
                model.train()
            scheduler.step()

            if opts.total_itrs > 0 and cur_itrs >= opts.total_itrs:
                return


if __name__ == '__main__':
    main()
