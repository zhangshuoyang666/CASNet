#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import time
from glob import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from thop import profile

# Ensure project root is importable regardless of launch cwd.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(THIS_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import network


LOSS_RE = re.compile(
    r"Epoch\s+(\d+),\s*Itrs\s+(\d+)/(-?\d+),\s*total_loss=([\d.]+),\s*seg_loss=([\d.]+),\s*edge_loss=([\d.]+)"
)
VAL_RE = re.compile(
    r"background\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+(\d+)"
)


def parse_log(log_path):
    loss_records = []
    val_records = []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        m = LOSS_RE.search(line)
        if m:
            loss_records.append(
                {
                    "epoch": int(m.group(1)),
                    "iter": int(m.group(2)),
                    "total_itr": int(m.group(3)),
                    "total_loss": float(m.group(4)),
                    "seg_loss": float(m.group(5)),
                    "edge_loss": float(m.group(6)),
                }
            )

        m2 = VAL_RE.search(line)
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
            }
            for j in range(i + 1, min(i + 10, len(lines))):
                ma = re.search(r"^abuth\s+([\d.]+)", lines[j])
                if ma:
                    rec["abuth_iou"] = float(ma.group(1))
                mc = re.search(r"^cotton\s+([\d.]+)", lines[j])
                if mc:
                    rec["cotton_iou"] = float(mc.group(1))
            val_records.append(rec)
    return loss_records, val_records


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_curves(loss_records, val_records, out_png):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Mainline-A Training Curves", fontsize=13, fontweight="bold")

    ax = axes[0]
    if loss_records:
        iters = [r["iter"] for r in loss_records]
        ax.plot(iters, [r["total_loss"] for r in loss_records], "b-", lw=1.2, label="Total")
        ax.plot(iters, [r["seg_loss"] for r in loss_records], "g-", lw=1.0, alpha=0.8, label="Seg")
        ax.plot(iters, [r["edge_loss"] for r in loss_records], "r-", lw=1.0, alpha=0.8, label="Edge")
    ax.set_title("Training Loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    if val_records:
        vi = [r["iter"] for r in val_records]
        ax.plot(vi, [r["miou"] for r in val_records], "b-o", ms=4, label="mIoU")
        ax.plot(vi, [r["macc"] for r in val_records], "g-s", ms=4, label="mAcc")
        ax.plot(vi, [r["oa"] for r in val_records], "r-^", ms=4, label="OA")
    ax.set_title("Validation Metrics")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Score (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[2]
    if val_records and val_records[0]["abuth_iou"] is not None:
        vi = [r["iter"] for r in val_records]
        ax.plot(vi, [r["bg_iou"] for r in val_records], "k--", lw=1.0, label="background IoU")
        ax.plot(vi, [r["abuth_iou"] for r in val_records], "b-o", ms=4, label="abuth IoU")
        ax.plot(vi, [r["cotton_iou"] for r in val_records], "g-s", ms=4, label="cotton IoU")
    ax.set_title("Per-Class IoU")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("IoU (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


def build_model_for_mainline_a():
    model = network.modeling.__dict__["deeplabv3plus_mobilenet"](
        num_classes=3,
        output_stride=16,
        attention_type="spatial_cbam",
        aspp_variant="standard",
        dense_aspp_rates=(1, 3, 6, 12, 18),
        enable_fg_fusion=False,
        enable_texture_enhance=False,
        enable_decoder_detail=False,
        enable_boundary_aux=True,
        use_saff=True,
        saff_f1_source="mid",
        saff_f2_source="aspp",
    )
    return model


def pick_checkpoint(run_dir):
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    bests = sorted(glob(os.path.join(ckpt_dir, "best_*.pth")))
    if bests:
        return bests[-1]
    latest = sorted(glob(os.path.join(ckpt_dir, "latest_*.pth")))
    return latest[-1] if latest else None


def benchmark_model(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    dummy = torch.randn(1, 3, 513, 513, device=device)

    with torch.no_grad():
        flops, params = profile(model, inputs=(dummy,), verbose=False)

    # FPS benchmark
    warmup = 20
    loops = 100
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(loops):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0
    fps = loops / max(elapsed, 1e-8)

    return {
        "parameters": int(params),
        "parameters_million": round(params / 1e6, 4),
        "flops": int(flops),
        "gflops": round(flops / 1e9, 4),
        "fps_bs1_513": round(fps, 4),
        "device": str(device),
    }


def write_summary(run_dir, loss_records, val_records, model_stats, checkpoint_path):
    best = max(val_records, key=lambda x: x["miou"]) if val_records else None
    last = val_records[-1] if val_records else None
    summary_path = os.path.join(run_dir, "result_summary.md")

    lines = [
        "# Mainline-A Result Summary",
        "",
        "## Validation Metrics",
    ]
    if best:
        lines.extend(
            [
                f"- best mIoU: {best['miou']:.2f} @ iter {best['iter']} (epoch {best['epoch']})",
                f"- best mAcc: {best['macc']:.2f}",
                f"- best aAcc: {best['oa']:.2f}",
                f"- best F1: {best['mf1']:.2f}",
                f"- best class IoU (bg/abuth/cotton): {best['bg_iou']:.2f}/{best.get('abuth_iou', 0):.2f}/{best.get('cotton_iou', 0):.2f}",
            ]
        )
    if last:
        lines.extend(
            [
                "",
                "## Last Validation Point",
                f"- last mIoU: {last['miou']:.2f} @ iter {last['iter']} (epoch {last['epoch']})",
                f"- last mAcc: {last['macc']:.2f}",
                f"- last aAcc: {last['oa']:.2f}",
                f"- last F1: {last['mf1']:.2f}",
            ]
        )
    lines.extend(
        [
            "",
            "## Complexity & Speed",
            f"- parameters: {model_stats['parameters']} ({model_stats['parameters_million']} M)",
            f"- GFLOPs (513x513, bs=1): {model_stats['gflops']}",
            f"- FPS (513x513, bs=1): {model_stats['fps_bs1_513']}",
            f"- benchmark device: {model_stats['device']}",
            "",
            "## Artifacts",
            f"- checkpoint used: `{checkpoint_path or 'N/A'}`",
            "- train log: `train.log`",
            "- validation table: `metrics.tsv`",
            "- parsed records: `metrics.json`",
            "- curve image: `curves.png`",
            "- model stats: `model_stats.json`",
        ]
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()
    run_dir = args.run_dir
    log_path = os.path.join(run_dir, "train.log")
    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"train.log not found: {log_path}")

    loss_records, val_records = parse_log(log_path)
    save_json(os.path.join(run_dir, "metrics.json"), {"loss": loss_records, "val": val_records})
    save_curves(loss_records, val_records, os.path.join(run_dir, "curves.png"))

    checkpoint_path = pick_checkpoint(run_dir)
    model = build_model_for_mainline_a()
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state, strict=False)
    model_stats = benchmark_model(model)
    save_json(os.path.join(run_dir, "model_stats.json"), model_stats)

    write_summary(run_dir, loss_records, val_records, model_stats, checkpoint_path)
    print(f"[postprocess] done: {run_dir}")


if __name__ == "__main__":
    main()
