#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOSS_RE = re.compile(
    r"Epoch\s+(\d+),\s*Itrs\s+(\d+)/(-?\d+),\s*total_loss=([\d.]+),\s*seg_loss=([\d.]+),\s*edge_loss=([\d.]+)"
)


def to_float(text):
    s = str(text).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_loss(log_path):
    records = []
    if not os.path.isfile(log_path):
        return records
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LOSS_RE.search(line)
            if not m:
                continue
            records.append(
                {
                    "epoch": int(m.group(1)),
                    "iter": int(m.group(2)),
                    "total_loss": float(m.group(4)),
                    "seg_loss": float(m.group(5)),
                    "edge_loss": float(m.group(6)),
                }
            )
    return records


def parse_rounds(metrics_tsv):
    rounds = []
    if not os.path.isfile(metrics_tsv):
        return rounds
    with open(metrics_tsv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    current = None
    for row in rows:
        if str(row.get("mIoU", "")).strip():
            if current is not None:
                rounds.append(current)
            current = {
                "iter": int(row.get("Iter", 0) or 0),
                "epoch": int(row.get("epoch", 0) or 0),
                "aAcc": to_float(row.get("aAcc")),
                "mIoU": to_float(row.get("mIoU")),
                "mAcc": to_float(row.get("mAcc")),
                "classes": {},
            }
        if current is None:
            continue
        cname = str(row.get("class", "")).strip()
        if not cname:
            continue
        current["classes"][cname] = {
            "IoU": to_float(row.get("IoU")),
            "Acc": to_float(row.get("Acc")),
            "F1": to_float(row.get("F1")),
        }
    if current is not None:
        rounds.append(current)
    return rounds


def draw_plot(loss_records, rounds, out_png):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    if loss_records:
        iters = [x["iter"] for x in loss_records]
        ax.plot(iters, [x["total_loss"] for x in loss_records], label="total_loss")
        ax.plot(iters, [x["seg_loss"] for x in loss_records], label="seg_loss")
        ax.plot(iters, [x["edge_loss"] for x in loss_records], label="edge_loss")
    ax.set_title("Training Loss")
    ax.set_xlabel("Iteration")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    if rounds:
        iters = [x["iter"] for x in rounds]
        ax.plot(iters, [x["mIoU"] for x in rounds], "o-", label="mIoU")
        ax.plot(iters, [x["aAcc"] for x in rounds], "s-", label="aAcc")
        ax.plot(iters, [x["mAcc"] for x in rounds], "^-", label="mAcc")
    ax.set_title("Global Metrics")
    ax.set_xlabel("Iteration")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[2]
    if rounds:
        names = list(rounds[-1]["classes"].keys())
        for n in names:
            ys = [r["classes"].get(n, {}).get("IoU") for r in rounds]
            ax.plot([r["iter"] for r in rounds], ys, label=n)
        ax.legend(fontsize=8)
    ax.set_title("Per-Class IoU")
    ax.set_xlabel("Iteration")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()
    run_dir = args.run_dir

    log_path = os.path.join(run_dir, "train.log")
    metrics_tsv = os.path.join(run_dir, "metrics.tsv")
    loss_records = parse_loss(log_path)
    rounds = parse_rounds(metrics_tsv)
    if not rounds:
        raise RuntimeError("No validation rows in metrics.tsv")

    best = max(rounds, key=lambda x: x["mIoU"])
    last = rounds[-1]
    summary = {
        "best": best,
        "last": last,
        "validation_rounds": len(rounds),
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    draw_plot(loss_records, rounds, os.path.join(run_dir, "loss_and_metrics.png"))

    lines = [
        "# CottonweedV4 Experiment Report",
        "",
        "## Best",
        f"- mIoU: {best['mIoU']:.6f}",
        f"- aAcc: {best['aAcc']:.6f}",
        f"- mAcc: {best['mAcc']:.6f}",
        f"- iter/epoch: {best['iter']}/{best['epoch']}",
        "",
        "## Per-Class (Best)",
        "| class | IoU | Acc | F1 |",
        "|---|---:|---:|---:|",
    ]
    for cname, m in best["classes"].items():
        lines.append(f"| {cname} | {m['IoU']:.6f} | {m['Acc']:.6f} | {m['F1']:.6f} |")
    lines.extend(
        [
            "",
            "## Last",
            f"- mIoU: {last['mIoU']:.6f}",
            f"- aAcc: {last['aAcc']:.6f}",
            f"- mAcc: {last['mAcc']:.6f}",
            f"- iter/epoch: {last['iter']}/{last['epoch']}",
            "",
            "## Artifacts",
            "- train log: `train.log`",
            "- metrics: `metrics.tsv`",
            "- figure: `loss_and_metrics.png`",
            "- summary json: `summary.json`",
        ]
    )
    with open(os.path.join(run_dir, "result_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[postprocess] done: {run_dir}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOSS_RE = re.compile(
    r"Epoch\s+(\d+),\s*Itrs\s+(\d+)/(-?\d+),\s*total_loss=([\d.]+),\s*seg_loss=([\d.]+),\s*edge_loss=([\d.]+)"
)


def to_float(text):
    s = str(text).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_loss(log_path):
    out = []
    if not os.path.isfile(log_path):
        return out
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LOSS_RE.search(line)
            if not m:
                continue
            out.append(
                {
                    "epoch": int(m.group(1)),
                    "iter": int(m.group(2)),
                    "total_loss": float(m.group(4)),
                    "seg_loss": float(m.group(5)),
                    "edge_loss": float(m.group(6)),
                }
            )
    return out


def parse_metrics_rounds(metrics_tsv):
    rounds = []
    if not os.path.isfile(metrics_tsv):
        return rounds
    with open(metrics_tsv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    cur = None
    for row in rows:
        if str(row.get("mIoU", "")).strip():
            if cur is not None:
                rounds.append(cur)
            cur = {
                "iter": int(row.get("Iter", 0) or 0),
                "epoch": int(row.get("epoch", 0) or 0),
                "aAcc": to_float(row.get("aAcc")),
                "mIoU": to_float(row.get("mIoU")),
                "mAcc": to_float(row.get("mAcc")),
                "mF1": to_float(row.get("mF1")),
                "classes": {},
            }
        if cur is None:
            continue
        cname = str(row.get("class", "")).strip()
        if not cname:
            continue
        cur["classes"][cname] = {
            "IoU": to_float(row.get("IoU")),
            "Acc": to_float(row.get("Acc")),
            "F1": to_float(row.get("F1")),
        }
    if cur is not None:
        rounds.append(cur)
    return rounds


def draw_curves(loss_records, rounds, png_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    if loss_records:
        it = [x["iter"] for x in loss_records]
        ax.plot(it, [x["total_loss"] for x in loss_records], label="total_loss")
        ax.plot(it, [x["seg_loss"] for x in loss_records], label="seg_loss")
        ax.plot(it, [x["edge_loss"] for x in loss_records], label="edge_loss")
    ax.set_title("Training Loss")
    ax.set_xlabel("Iteration")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    if rounds:
        it = [x["iter"] for x in rounds]
        ax.plot(it, [x["mIoU"] for x in rounds], "o-", label="mIoU")
        ax.plot(it, [x["aAcc"] for x in rounds], "s-", label="aAcc")
        ax.plot(it, [x["mAcc"] for x in rounds], "^-", label="mAcc")
    ax.set_title("Global Metrics")
    ax.set_xlabel("Iteration")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[2]
    if rounds:
        class_names = list(rounds[-1]["classes"].keys())
        for cname in class_names:
            ax.plot([x["iter"] for x in rounds], [x["classes"].get(cname, {}).get("IoU") for x in rounds], label=cname)
    ax.set_title("Per-Class IoU")
    ax.set_xlabel("Iteration")
    ax.grid(True, alpha=0.3)
    if rounds:
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(png_path, dpi=140)
    plt.close()


def write_report(run_dir, rounds, report_path, best_idx):
    best = rounds[best_idx] if rounds else None
    last = rounds[-1] if rounds else None
    lines = []
    lines.append("# CottonWeedV4 Experiment Report")
    lines.append("")
    if best is not None:
        lines.append("## Best Validation")
        lines.append(f"- mIoU: {best['mIoU']:.6f}")
        lines.append(f"- aAcc: {best['aAcc']:.6f}")
        lines.append(f"- mAcc: {best['mAcc']:.6f}")
        lines.append(f"- iter/epoch: {best['iter']}/{best['epoch']}")
        lines.append("")
        lines.append("### Per-Class Metrics (Best)")
        lines.append("| class | IoU | Acc | F1 |")
        lines.append("|---|---:|---:|---:|")
        for cname, m in best["classes"].items():
            lines.append(f"| {cname} | {m['IoU']:.6f} | {m['Acc']:.6f} | {m['F1']:.6f} |")
    if last is not None:
        lines.append("")
        lines.append("## Last Validation")
        lines.append(f"- mIoU: {last['mIoU']:.6f}")
        lines.append(f"- aAcc: {last['aAcc']:.6f}")
        lines.append(f"- mAcc: {last['mAcc']:.6f}")
        lines.append(f"- iter/epoch: {last['iter']}/{last['epoch']}")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("- train log: `train.log`")
    lines.append("- metric table: `metrics.tsv`")
    lines.append("- curves: `loss_and_metrics.png`")
    lines.append("- summary json: `summary.json`")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()

    run_dir = args.run_dir
    log_path = os.path.join(run_dir, "train.log")
    metrics_tsv = os.path.join(run_dir, "metrics.tsv")
    loss_records = parse_loss(log_path)
    rounds = parse_metrics_rounds(metrics_tsv)
    if not rounds:
        raise RuntimeError(f"No validation data found in {metrics_tsv}")

    best_idx = max(range(len(rounds)), key=lambda i: rounds[i]["mIoU"])
    summary = {
        "best": rounds[best_idx],
        "last": rounds[-1],
        "validation_rounds": len(rounds),
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    draw_curves(loss_records, rounds, os.path.join(run_dir, "loss_and_metrics.png"))
    write_report(run_dir, rounds, os.path.join(run_dir, "result_summary.md"), best_idx)
    print(f"[postprocess] done: {run_dir}")


if __name__ == "__main__":
    main()
