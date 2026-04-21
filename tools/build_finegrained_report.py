import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def read_tsv(path):
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        return rows
    header = lines[0].split("\t")
    for line in lines[1:]:
        values = line.split("\t")
        data = {}
        for i, key in enumerate(header):
            data[key] = values[i] if i < len(values) else ""
        rows.append(data)
    return rows


def to_float(text, default=0.0):
    try:
        return float(text)
    except Exception:
        return default


def select_experiments(work_dir, timestamp):
    exp_dirs = []
    for name in sorted(os.listdir(work_dir)):
        full = os.path.join(work_dir, name)
        if os.path.isdir(full) and name.endswith(timestamp) and "cottonweed_" in name:
            exp_dirs.append(full)
    return exp_dirs


def summarize_experiment(exp_dir):
    metrics_path = os.path.join(exp_dir, "metrics.tsv")
    confusion_path = os.path.join(exp_dir, "similarity_confusion.tsv")
    rows = read_tsv(metrics_path)
    confusion_rows = read_tsv(confusion_path)
    if not rows:
        return None

    grouped = []
    current = []
    for row in rows:
        if row.get("Iter", ""):
            if current:
                grouped.append(current)
            current = [row]
        else:
            if current:
                current.append(row)
    if current:
        grouped.append(current)

    best = {
        "mIoU": -1.0,
        "mF1": -1.0,
        "aAcc": 0.0,
        "iter": "",
        "epoch": "",
        "class_rows": {},
    }
    for group_rows in grouped:
        first = group_rows[0]
        if first is None:
            continue
        miou = to_float(first.get("mIoU", "0"))
        if miou > best["mIoU"]:
            best["mIoU"] = miou
            best["mF1"] = to_float(first.get("mF1", "0"))
            best["aAcc"] = to_float(first.get("aAcc", "0"))
            best["iter"] = first.get("Iter", "")
            best["epoch"] = first.get("epoch", "")
            best["class_rows"] = {r.get("class", ""): r for r in group_rows}

    confusion_map = {row.get("Iter", ""): row for row in confusion_rows}
    confusion_best = confusion_map.get(best["iter"], {})

    def get_cls(name):
        return best["class_rows"].get(name, {"IoU": "", "Acc": "", "F1": ""})

    return {
        "exp_name": os.path.basename(exp_dir),
        "iter": best["iter"],
        "epoch": best["epoch"],
        "mIoU": best["mIoU"],
        "mF1": best["mF1"],
        "aAcc": best["aAcc"],
        "background": get_cls("background"),
        "cotton": get_cls("cotton"),
        "abuth": get_cls("abuth"),
        "cotton_to_abuth": to_float(confusion_best.get("cotton_to_abuth", "0")),
        "abuth_to_cotton": to_float(confusion_best.get("abuth_to_cotton", "0")),
    }


def pct(x):
    return "{:.2f}".format(float(x) * 100.0)


def build_report(results):
    lines = []
    lines.append("# CottonWeed Fine-grained Segmentation Experiment Report")
    lines.append("")
    lines.append("## Overview")
    lines.append("- 目标：增强 cotton 与 abuth 的细粒度可分性（非泛化型提升）。")
    lines.append("- 模块：FineGrainedFusion / TextureEnhance / DenseASPP or DeformASPP / ImprovedDecoder(detail path).")
    lines.append("")
    lines.append("## Key Results (best checkpoint by mIoU)")
    lines.append("| Experiment | mIoU | mF1 | aAcc | cotton IoU | cotton Acc | abuth IoU | abuth Acc | cotton->abuth | abuth->cotton |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                r["exp_name"],
                pct(r["mIoU"]),
                pct(r["mF1"]),
                pct(r["aAcc"]),
                pct(to_float(r["cotton"].get("IoU", "0"))),
                pct(to_float(r["cotton"].get("Acc", "0"))),
                pct(to_float(r["abuth"].get("IoU", "0"))),
                pct(to_float(r["abuth"].get("Acc", "0"))),
                pct(r["cotton_to_abuth"]),
                pct(r["abuth_to_cotton"]),
            )
        )
    lines.append("")
    lines.append("## Per-class Snapshot")
    for r in results:
        lines.append("### {}".format(r["exp_name"]))
        lines.append("- Best Iter/Epoch: {}/{}".format(r["iter"], r["epoch"]))
        lines.append("- background: IoU {} / Acc {} / F1 {}".format(
            pct(to_float(r["background"].get("IoU", "0"))),
            pct(to_float(r["background"].get("Acc", "0"))),
            pct(to_float(r["background"].get("F1", "0"))),
        ))
        lines.append("- cotton: IoU {} / Acc {} / F1 {}".format(
            pct(to_float(r["cotton"].get("IoU", "0"))),
            pct(to_float(r["cotton"].get("Acc", "0"))),
            pct(to_float(r["cotton"].get("F1", "0"))),
        ))
        lines.append("- abuth: IoU {} / Acc {} / F1 {}".format(
            pct(to_float(r["abuth"].get("IoU", "0"))),
            pct(to_float(r["abuth"].get("Acc", "0"))),
            pct(to_float(r["abuth"].get("F1", "0"))),
        ))
        lines.append("- confusion: cotton->abuth {} | abuth->cotton {}".format(
            pct(r["cotton_to_abuth"]),
            pct(r["abuth_to_cotton"]),
        ))
        lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    exp_dirs = select_experiments(args.work_dir, args.timestamp)
    results = []
    for exp_dir in exp_dirs:
        summary = summarize_experiment(exp_dir)
        if summary is not None:
            results.append(summary)
    results = sorted(results, key=lambda x: x["exp_name"])
    report = build_report(results)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print("Report generated: {}".format(args.output))
    print("Experiments summarized: {}".format(len(results)))


if __name__ == "__main__":
    main()
