#!/usr/bin/env python3
import re
import json
from pathlib import Path

import matplotlib.pyplot as plt

EXP_ROOT = Path('/root/autodl-tmp/CODE/DeepLabV3Plus-Pytorch-master/DeepLabV3Plus-Pytorch-masterPLUS/5cottonweedV4（cotton-abuth-others）')
RUN_DIR = EXP_ROOT / 'workdirs' / 'exp_spatial_cbam_saff_boundaryaux_cottonweedV4_4class_itr3000_lr0p02_bs32'
LOG_PATH = EXP_ROOT / 'train.log'
METRICS_TSV = RUN_DIR / 'metrics.tsv'
MODEL_STATS_JSON = EXP_ROOT / 'model_stats.json'


def parse_losses_from_log(text: str):
    pattern = re.compile(
        r"Epoch\s+(\d+),\s+Itrs\s+(\d+)/(\d+),\s+total_loss=([0-9.]+),\s+seg_loss=([0-9.]+),\s+edge_loss=([0-9.]+)"
    )
    xs, total, seg, edge = [], [], [], []
    for m in pattern.finditer(text):
        xs.append(int(m.group(2)))
        total.append(float(m.group(4)))
        seg.append(float(m.group(5)))
        edge.append(float(m.group(6)))
    return xs, total, seg, edge


def plot_losses(xs, total, seg, edge, out_png: Path):
    if not xs:
        return False
    plt.figure(figsize=(9, 5))
    plt.plot(xs, total, label='total_loss')
    plt.plot(xs, seg, label='seg_loss')
    plt.plot(xs, edge, label='edge_loss')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('Training Loss Curves')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return True


def parse_metrics_tsv(path: Path):
    rows = []
    if not path.exists():
        return rows
    lines = path.read_text(encoding='utf-8').strip().splitlines()
    if len(lines) <= 1:
        return rows
    for line in lines[1:]:
        parts = line.split('\t')
        if len(parts) < 10:
            continue
        cls, iou, acc, f1, aacc, miou, macc, mf1, itr, epoch = parts[:10]
        row = {
            'class': cls,
            'IoU': float(iou),
            'Acc': float(acc),
            'F1': float(f1),
            'aAcc': float(aacc) if aacc else None,
            'mIoU': float(miou) if miou else None,
            'mAcc': float(macc) if macc else None,
            'mF1': float(mf1) if mf1 else None,
            'Iter': int(itr) if itr else None,
            'epoch': int(epoch) if epoch else None,
        }
        rows.append(row)
    return rows


def plot_miou_curve(rows, out_png: Path):
    eval_rows = [r for r in rows if r['mIoU'] is not None and r['Iter'] is not None]
    if not eval_rows:
        return False
    xs = [r['Iter'] for r in eval_rows]
    ys = [r['mIoU'] * 100.0 for r in eval_rows]
    plt.figure(figsize=(8, 4.6))
    plt.plot(xs, ys, marker='o', label='mIoU (%)')
    plt.xlabel('iteration')
    plt.ylabel('mIoU (%)')
    plt.title('Validation mIoU Curve')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return True


def build_summary(log_text: str, rows, model_stats):
    overall_acc = re.findall(r'Overall Acc:\s*([0-9.]+)', log_text)
    mean_acc = re.findall(r'Mean Acc:\s*([0-9.]+)', log_text)
    mean_iou = re.findall(r'Mean IoU:\s*([0-9.]+)', log_text)

    latest_eval = None
    for r in reversed(rows):
        if r['aAcc'] is not None:
            latest_eval = r
            break

    summary = {
        'latest_from_log': {
            'Overall Acc': float(overall_acc[-1]) if overall_acc else None,
            'Mean Acc': float(mean_acc[-1]) if mean_acc else None,
            'Mean IoU': float(mean_iou[-1]) if mean_iou else None,
        },
        'latest_from_metrics_tsv': latest_eval,
        'model_complexity': model_stats,
        'paths': {
            'log': str(LOG_PATH),
            'metrics_tsv': str(METRICS_TSV),
            'run_dir': str(RUN_DIR),
        },
    }
    return summary


def main():
    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    log_text = LOG_PATH.read_text(encoding='utf-8', errors='ignore') if LOG_PATH.exists() else ''

    xs, total, seg, edge = parse_losses_from_log(log_text)
    plot_losses(xs, total, seg, edge, EXP_ROOT / 'train_loss_curve.png')

    rows = parse_metrics_tsv(METRICS_TSV)
    plot_miou_curve(rows, EXP_ROOT / 'val_miou_curve.png')

    model_stats = {}
    if MODEL_STATS_JSON.exists():
        model_stats = json.loads(MODEL_STATS_JSON.read_text(encoding='utf-8'))

    summary = build_summary(log_text, rows, model_stats)
    (EXP_ROOT / 'metrics_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    txt_lines = [
        'Training Summary (auto-generated)',
        f'Log: {LOG_PATH}',
        f'Run dir: {RUN_DIR}',
        '',
        f"Latest Overall Acc: {summary['latest_from_log']['Overall Acc']}",
        f"Latest Mean Acc: {summary['latest_from_log']['Mean Acc']}",
        f"Latest Mean IoU: {summary['latest_from_log']['Mean IoU']}",
        '',
        f"Params: {model_stats.get('params')}",
        f"GFLOPs: {model_stats.get('GFLOPs')}",
        f"FPS: {model_stats.get('FPS')}",
    ]
    (EXP_ROOT / 'metrics_summary.txt').write_text('\n'.join(txt_lines) + '\n', encoding='utf-8')

    print('Saved:', EXP_ROOT / 'train_loss_curve.png')
    print('Saved:', EXP_ROOT / 'val_miou_curve.png')
    print('Saved:', EXP_ROOT / 'metrics_summary.json')
    print('Saved:', EXP_ROOT / 'metrics_summary.txt')


if __name__ == '__main__':
    main()
