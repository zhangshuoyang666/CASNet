import argparse
import os


def read_tsv(path):
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]
    if not lines:
        return rows
    header = lines[0].split("\t")
    for line in lines[1:]:
        values = line.split("\t")
        row = {}
        for idx, key in enumerate(header):
            row[key] = values[idx] if idx < len(values) else ""
        rows.append(row)
    return rows


def to_float(text, default=0.0):
    try:
        return float(text)
    except Exception:
        return default


def group_metric_rows(rows):
    groups = []
    current = []
    for row in rows:
        if row.get("Iter", ""):
            if current:
                groups.append(current)
            current = [row]
        elif current:
            current.append(row)
    if current:
        groups.append(current)
    return groups


def pick_best_snapshot(metrics_rows):
    groups = group_metric_rows(metrics_rows)
    best = None
    best_miou = -1.0
    for grp in groups:
        head = grp[0]
        miou = to_float(head.get("mIoU", ""))
        if miou > best_miou:
            best_miou = miou
            by_class = {r.get("class", ""): r for r in grp}
            best = {
                "iter": head.get("Iter", ""),
                "epoch": head.get("epoch", ""),
                "mIoU": miou,
                "mPA": to_float(head.get("mAcc", "")),
                "mF1": to_float(head.get("mF1", "")),
                "cotton_iou": to_float(by_class.get("cotton", {}).get("IoU", "")),
                "abuth_iou": to_float(by_class.get("abuth", {}).get("IoU", "")),
            }
    return best


def read_confusion_at_iter(confusion_rows, iter_text):
    for row in confusion_rows:
        if row.get("Iter", "") == str(iter_text):
            return (
                to_float(row.get("cotton_to_abuth", "")),
                to_float(row.get("abuth_to_cotton", "")),
            )
    return 0.0, 0.0


def summarize_experiment(work_dir, exp_name):
    exp_dir = os.path.join(work_dir, exp_name)
    metrics_path = os.path.join(exp_dir, "metrics.tsv")
    confusion_path = os.path.join(exp_dir, "confusion.tsv")
    if not os.path.isfile(confusion_path):
        confusion_path = os.path.join(exp_dir, "similarity_confusion.tsv")

    metrics_rows = read_tsv(metrics_path)
    confusion_rows = read_tsv(confusion_path)
    if not metrics_rows:
        return None
    best = pick_best_snapshot(metrics_rows)
    if best is None:
        return None
    c2a, a2c = read_confusion_at_iter(confusion_rows, best["iter"])
    best["cotton_to_abuth"] = c2a
    best["abuth_to_cotton"] = a2c
    best["exp_name"] = exp_name
    return best


def pct(v):
    return "{:.2f}%".format(v * 100.0)


def build_markdown(baseline, saff):
    lines = []
    lines.append("# SAFF vs Baseline (cottonweed)")
    lines.append("")
    lines.append("| Experiment | Best Iter | Best Epoch | best mIoU | best mPA | best mF1 | cotton IoU | abuth IoU | cotton->abuth | abuth->cotton |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for item in [baseline, saff]:
        if item is None:
            continue
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                item["exp_name"],
                item["iter"],
                item["epoch"],
                pct(item["mIoU"]),
                pct(item["mPA"]),
                pct(item["mF1"]),
                pct(item["cotton_iou"]),
                pct(item["abuth_iou"]),
                pct(item["cotton_to_abuth"]),
                pct(item["abuth_to_cotton"]),
            )
        )
    lines.append("")
    if baseline is not None and saff is not None:
        lines.append("## SAFF Gain (SAFF - Baseline)")
        lines.append("- mIoU: {}".format(pct(saff["mIoU"] - baseline["mIoU"])))
        lines.append("- mPA: {}".format(pct(saff["mPA"] - baseline["mPA"])))
        lines.append("- mF1: {}".format(pct(saff["mF1"] - baseline["mF1"])))
        lines.append("- cotton IoU: {}".format(pct(saff["cotton_iou"] - baseline["cotton_iou"])))
        lines.append("- abuth IoU: {}".format(pct(saff["abuth_iou"] - baseline["abuth_iou"])))
        lines.append("- cotton->abuth confusion: {}".format(pct(saff["cotton_to_abuth"] - baseline["cotton_to_abuth"])))
        lines.append("- abuth->cotton confusion: {}".format(pct(saff["abuth_to_cotton"] - baseline["abuth_to_cotton"])))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="./workdirs")
    parser.add_argument("--baseline_exp", type=str, required=True)
    parser.add_argument("--saff_exp", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    baseline = summarize_experiment(args.work_dir, args.baseline_exp)
    saff = summarize_experiment(args.work_dir, args.saff_exp)

    report = build_markdown(baseline, saff)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)

    print("Report generated: {}".format(args.output))
    print("baseline found: {}".format(baseline is not None))
    print("saff found: {}".format(saff is not None))


if __name__ == "__main__":
    main()
