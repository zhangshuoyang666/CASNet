#!/usr/bin/env python3
"""Compare multiple experiments by parsing their train.log files.
Usage: python compare_experiments.py dir1 dir2 dir3 ... [--output compare.png]
"""
import sys, os, re, argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from parse_and_plot import parse_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--output", default=None)
    ap.add_argument("--title", default="LR Comparison")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(args.title, fontsize=13, fontweight="bold")

    summary_lines = []
    for d in args.dirs:
        log_path = os.path.join(d, "train.log")
        if not os.path.isfile(log_path):
            continue
        name = os.path.basename(d)
        loss_records, val_records = parse_log(log_path)
        if not val_records:
            continue

        vi = [r["iter"] for r in val_records]
        axes[0].plot([r["iter"] for r in loss_records],
                     [r["total_loss"] for r in loss_records], lw=1.2, label=name)
        axes[1].plot(vi, [r["miou"] for r in val_records], "-o", ms=3, label=name)
        if val_records[0].get("abuth_iou") is not None:
            axes[2].plot(vi, [r["abuth_iou"] for r in val_records], "-o", ms=3, label=f"{name} abuth")
            axes[2].plot(vi, [r["cotton_iou"] for r in val_records], "-s", ms=3, label=f"{name} cotton")

        best_idx = max(range(len(val_records)), key=lambda i: val_records[i]["miou"])
        best = val_records[best_idx]
        summary_lines.append(
            f"{name}: mIoU={best['miou']:.2f}% @iter{best['iter']} "
            f"abuth={best.get('abuth_iou','?')}% cotton={best.get('cotton_iou','?')}%"
        )

    for ax, title in zip(axes, ["Training Loss", "Val mIoU (%)", "Per-Class IoU (%)"]):
        ax.set_xlabel("Iteration")
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = args.output or os.path.join(os.path.dirname(args.dirs[0]), "comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison to {out}")
    print("\n=== Summary ===")
    for line in summary_lines:
        print(line)


if __name__ == "__main__":
    main()
