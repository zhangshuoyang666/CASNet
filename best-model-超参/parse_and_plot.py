#!/usr/bin/env python3
"""Parse a train.log and generate curves.png + metrics.csv for an experiment directory."""
import re, os, sys, json, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def parse_log(log_path):
    loss_records = []
    val_records = []
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        m = re.search(
            r"Epoch\s+(\d+),\s*Itrs\s+(\d+)/(\d+),\s*total_loss=([\d.]+),\s*seg_loss=([\d.]+),\s*edge_loss=([\d.]+)",
            line,
        )
        if m:
            loss_records.append({
                "epoch": int(m.group(1)),
                "iter": int(m.group(2)),
                "total_itr": int(m.group(3)),
                "total_loss": float(m.group(4)),
                "seg_loss": float(m.group(5)),
                "edge_loss": float(m.group(6)),
            })

        m2 = re.search(
            r"background\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+(\d+)",
            line,
        )
        if m2:
            rec = {
                "bg_iou": float(m2.group(1)),
                "oa": float(m2.group(4)),
                "miou": float(m2.group(5)),
                "macc": float(m2.group(6)),
                "mf1": float(m2.group(7)),
                "iter": int(m2.group(8)),
                "epoch": int(m2.group(9)),
                "abuth_iou": None,
                "cotton_iou": None,
                "cotton_as_abuth": None,
                "abuth_as_cotton": None,
            }
            for j in range(i + 1, min(i + 10, len(lines))):
                ma = re.search(r"^abuth\s+([\d.]+)", lines[j])
                if ma:
                    rec["abuth_iou"] = float(ma.group(1))
                mc = re.search(r"^cotton\s+([\d.]+)", lines[j])
                if mc:
                    rec["cotton_iou"] = float(mc.group(1))
                mc2 = re.search(r"cotton\s*->\s*abuth\s+confusion:\s+([\d.]+)%", lines[j])
                if mc2:
                    rec["cotton_as_abuth"] = float(mc2.group(1))
                mc3 = re.search(r"abuth\s*->\s*cotton\s+confusion:\s+([\d.]+)%", lines[j])
                if mc3:
                    rec["abuth_as_cotton"] = float(mc3.group(1))
            val_records.append(rec)

    return loss_records, val_records


def save_metrics_csv(val_records, out_path):
    if not val_records:
        return
    fields = ["epoch", "iter", "miou", "macc", "oa", "mf1", "bg_iou", "abuth_iou", "cotton_iou",
              "cotton_as_abuth", "abuth_as_cotton"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in val_records:
            w.writerow(r)


def plot_curves(loss_records, val_records, out_path, title=""):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title or "Training Curves", fontsize=13, fontweight="bold")

    # Loss curves
    ax = axes[0]
    if loss_records:
        iters = [r["iter"] for r in loss_records]
        ax.plot(iters, [r["total_loss"] for r in loss_records], "b-", lw=1.2, label="Total")
        ax.plot(iters, [r["seg_loss"] for r in loss_records], "g-", lw=1, alpha=0.7, label="Seg")
        ax.plot(iters, [r["edge_loss"] for r in loss_records], "r-", lw=1, alpha=0.7, label="Edge")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # mIoU / mAcc
    ax2 = axes[1]
    if val_records:
        vi = [r["iter"] for r in val_records]
        ax2.plot(vi, [r["miou"] for r in val_records], "b-o", ms=4, label="mIoU")
        ax2.plot(vi, [r["macc"] for r in val_records], "g-s", ms=4, label="mAcc")
        ax2.plot(vi, [r["oa"] for r in val_records], "r-^", ms=4, label="OA")
        best_idx = max(range(len(val_records)), key=lambda i: val_records[i]["miou"])
        best = val_records[best_idx]
        ax2.axhline(y=best["miou"], color="blue", ls="--", alpha=0.4)
        ax2.annotate(f"best={best['miou']:.2f}%@{best['iter']}",
                     (best["iter"], best["miou"]),
                     textcoords="offset points", xytext=(5, 8), fontsize=8, color="blue")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Score (%)")
    ax2.set_title("Validation Metrics")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Per-class IoU
    ax3 = axes[2]
    if val_records and val_records[0]["abuth_iou"] is not None:
        vi = [r["iter"] for r in val_records]
        ax3.plot(vi, [r["abuth_iou"] for r in val_records], "b-o", ms=4, label="Abuth IoU")
        ax3.plot(vi, [r["cotton_iou"] for r in val_records], "g-s", ms=4, label="Cotton IoU")
        ax3.plot(vi, [r["bg_iou"] for r in val_records], "k--", lw=1, alpha=0.5, label="BG IoU")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("IoU (%)")
    ax3.set_title("Per-Class IoU")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_result_summary(val_records, exp_dir, exp_name, changed_param, old_val, new_val, purpose):
    if not val_records:
        return "No validation records found."
    best_idx = max(range(len(val_records)), key=lambda i: val_records[i]["miou"])
    best = val_records[best_idx]
    md = f"""# Result Summary: {exp_name}

## Purpose
{purpose}

## Changed Parameter
- **{changed_param}**: {old_val} → {new_val}

## Best Validation Results
| Metric | Value |
|--------|-------|
| **mIoU** | **{best['miou']:.2f}%** |
| mAcc | {best['macc']:.2f}% |
| OA | {best['oa']:.2f}% |
| Abuth IoU | {best.get('abuth_iou', 'N/A')}% |
| Cotton IoU | {best.get('cotton_iou', 'N/A')}% |
| Best Iter | {best['iter']} |
| Best Epoch | {best['epoch']} |

## Confusion
- cotton→abuth: {best.get('cotton_as_abuth', 'N/A')}%
- abuth→cotton: {best.get('abuth_as_cotton', 'N/A')}%

## Files
- `train.log` — full training log
- `metrics.csv` — parsed validation metrics
- `curves.png` — loss/mIoU/per-class curves
- `config_snapshot.json` — experiment config
- `run.sh` — launch script
"""
    path = os.path.join(exp_dir, "result_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    return md


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_and_plot.py <exp_dir> [title]")
        sys.exit(1)
    exp_dir = sys.argv[1]
    title = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(exp_dir)
    log_path = os.path.join(exp_dir, "train.log")
    if not os.path.isfile(log_path):
        print(f"No train.log found in {exp_dir}")
        sys.exit(1)

    loss_records, val_records = parse_log(log_path)
    save_metrics_csv(val_records, os.path.join(exp_dir, "metrics.csv"))
    plot_curves(loss_records, val_records, os.path.join(exp_dir, "curves.png"), title)

    cfg_path = os.path.join(exp_dir, "config_snapshot.json")
    changed_param, old_val, new_val, purpose = "N/A", "N/A", "N/A", title
    if os.path.isfile(cfg_path):
        with open(cfg_path) as cf:
            cfg = json.load(cf)
            changed_param = cfg.get("changed_param", changed_param)
            old_val = cfg.get("old_value", old_val)
            new_val = cfg.get("new_value", new_val)
            purpose = cfg.get("purpose", purpose)
    generate_result_summary(val_records, exp_dir, title, changed_param, old_val, new_val, purpose)

    print(f"Parsed: {len(loss_records)} loss records, {len(val_records)} val records")
    if val_records:
        best_idx = max(range(len(val_records)), key=lambda i: val_records[i]["miou"])
        best = val_records[best_idx]
        print(f"Best mIoU: {best['miou']:.2f}% @ iter {best['iter']} (epoch {best['epoch']})")
        print(f"  Abuth IoU: {best.get('abuth_iou', 'N/A')}%")
        print(f"  Cotton IoU: {best.get('cotton_iou', 'N/A')}%")
        return best
    return None


if __name__ == "__main__":
    main()
