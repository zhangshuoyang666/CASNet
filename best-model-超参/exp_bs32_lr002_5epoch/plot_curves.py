"""Parse training log and generate loss/mIoU curve plots."""
import re
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

LOG_FILE = os.path.join(os.path.dirname(__file__), "train_exp_bs32_lr002_5epoch.log")
OUT_DIR = os.path.dirname(__file__)

# ── parse training loss lines ─────────────────────────────────────────────────
loss_iter, total_loss, seg_loss, edge_loss = [], [], [], []
val_iter, val_miou, val_macc, val_oa = [], [], [], []
val_bg_iou, val_abuth_iou, val_cotton_iou = [], [], []
val_epoch = []

with open(LOG_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    # Training loss: "Epoch X, Itrs N/85, total_loss=..., seg_loss=..., edge_loss=..."
    m = re.search(
        r"Itrs\s+(\d+)/\d+,\s*total_loss=([\d.]+),\s*seg_loss=([\d.]+),\s*edge_loss=([\d.]+)",
        line,
    )
    if m:
        loss_iter.append(int(m.group(1)))
        total_loss.append(float(m.group(2)))
        seg_loss.append(float(m.group(3)))
        edge_loss.append(float(m.group(4)))

    # Val summary line: "background  IoU  Acc  F1  aAcc  mIoU  mAcc  mF1  Iter  epoch"
    m2 = re.search(
        r"background\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+(\d+)",
        line,
    )
    if m2:
        val_bg_iou.append(float(m2.group(1)))
        val_oa.append(float(m2.group(4)))
        val_miou.append(float(m2.group(5)))
        val_macc.append(float(m2.group(6)))
        val_iter.append(int(m2.group(8)))
        val_epoch.append(int(m2.group(9)))

    # Per-class IoU for abuth / cotton
    m3 = re.search(r"abuth\s+([\d.]+)", line)
    if m3:
        val_abuth_iou.append(float(m3.group(1)))
    m4 = re.search(r"cotton\s+([\d.]+)", line)
    if m4:
        val_cotton_iou.append(float(m4.group(1)))

# ── Figure 1: Training Loss Curves ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("BS=32 / LR=0.02  |  Training Loss (5 Epochs)", fontsize=13, fontweight="bold")

ax = axes[0]
ax.plot(loss_iter, total_loss, "b-o", ms=4, label="Total Loss")
ax.plot(loss_iter, seg_loss,   "g-s", ms=4, label="Seg Loss")
ax.plot(loss_iter, edge_loss,  "r-^", ms=4, label="Edge Loss")
ax.set_xlabel("Iteration")
ax.set_ylabel("Loss")
ax.set_title("All Losses vs. Iteration")
ax.legend()
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

ax2 = axes[1]
ax2.plot(loss_iter, total_loss, "b-o", ms=4)
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Total Loss")
ax2.set_title("Total Loss Trend")
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_locator(ticker.MultipleLocator(10))

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "loss_curves.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: loss_curves.png")

# ── Figure 2: Validation Metrics ─────────────────────────────────────────────
epochs = list(range(1, len(val_iter) + 1))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("BS=32 / LR=0.02  |  Validation Metrics per Epoch", fontsize=13, fontweight="bold")

ax = axes[0]
ax.plot(epochs, val_miou, "b-o", ms=6, label="mIoU")
ax.plot(epochs, val_macc, "g-s", ms=6, label="mAcc")
ax.plot(epochs, val_oa,   "r-^", ms=6, label="Overall Acc")
for ep, v in zip(epochs, val_miou):
    ax.annotate(f"{v:.1f}%", (ep, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
ax.set_xlabel("Epoch")
ax.set_ylabel("Score (%)")
ax.set_title("mIoU / mAcc / Overall Acc")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(epochs)
ax.set_ylim(40, 100)

ax2 = axes[1]
ax2.plot(epochs, val_bg_iou,     "k-o",  ms=6, label="Background IoU")
ax2.plot(epochs, val_abuth_iou,  "b-s",  ms=6, label="Abuth IoU")
ax2.plot(epochs, val_cotton_iou, "g-^",  ms=6, label="Cotton IoU")
for ep, v in zip(epochs, val_cotton_iou):
    ax2.annotate(f"{v:.1f}%", (ep, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("IoU (%)")
ax2.set_title("Per-Class IoU")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xticks(epochs)
ax2.set_ylim(0, 105)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "val_metrics.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: val_metrics.png")

# ── Figure 3: Loss decomposition (stacked area) ──────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
ax.stackplot(loss_iter, seg_loss, edge_loss,
             labels=["Seg Loss", "Edge Loss (×0.1)"],
             colors=["#4C8BE8", "#F06040"], alpha=0.75)
ax.set_xlabel("Iteration")
ax.set_ylabel("Loss")
ax.set_title("Stacked Loss Decomposition (5 Epochs, BS=32)")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "loss_stacked.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: loss_stacked.png")

print("\n=== Parsed Summary ===")
print(f"Training steps: {len(loss_iter)}")
print(f"Val checkpoints: {len(val_miou)}")
for ep, it, miu, mac, oa in zip(epochs, val_iter, val_miou, val_macc, val_oa):
    print(f"  Epoch {ep} (iter {it}): mIoU={miu:.2f}%  mAcc={mac:.2f}%  OA={oa:.2f}%")
print(f"\nBest mIoU: {max(val_miou):.2f}% @ Epoch {val_miou.index(max(val_miou))+1}")
