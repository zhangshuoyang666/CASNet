import argparse
import os


def to_float(text, default=0.0):
    try:
        return float(text)
    except Exception:
        return default


def read_tsv(path):
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]
    if not lines:
        return []
    header = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        vals = line.split("\t")
        row = {}
        for i, k in enumerate(header):
            row[k] = vals[i] if i < len(vals) else ""
        rows.append(row)
    return rows


def best_snapshot(metrics_rows):
    groups = []
    cur = []
    for row in metrics_rows:
        if row.get("Iter", ""):
            if cur:
                groups.append(cur)
            cur = [row]
        elif cur:
            cur.append(row)
    if cur:
        groups.append(cur)

    best = None
    best_miou = -1.0
    for grp in groups:
        head = grp[0]
        miou = to_float(head.get("mIoU", ""))
        if miou > best_miou:
            best_miou = miou
            cls = {r.get("class", ""): r for r in grp}
            best = {
                "iter": head.get("Iter", ""),
                "epoch": head.get("epoch", ""),
                "mIoU": miou,
                "mPA": to_float(head.get("mAcc", "")),
                "mF1": to_float(head.get("mF1", "")),
                "cotton_iou": to_float(cls.get("cotton", {}).get("IoU", "")),
                "abuth_iou": to_float(cls.get("abuth", {}).get("IoU", "")),
            }
    return best


def confusion_at_iter(conf_rows, iter_id):
    for row in conf_rows:
        if row.get("Iter", "") == str(iter_id):
            return (
                to_float(row.get("cotton_to_abuth", "")),
                to_float(row.get("abuth_to_cotton", "")),
            )
    return 0.0, 0.0


def summarize_exp(work_dir, exp_name):
    exp_dir = os.path.join(work_dir, exp_name)
    metrics = read_tsv(os.path.join(exp_dir, "metrics.tsv"))
    if not metrics:
        return None
    conf_path = os.path.join(exp_dir, "confusion.tsv")
    if not os.path.isfile(conf_path):
        conf_path = os.path.join(exp_dir, "similarity_confusion.tsv")
    conf = read_tsv(conf_path)
    best = best_snapshot(metrics)
    if best is None:
        return None
    c2a, a2c = confusion_at_iter(conf, best["iter"])
    best["cotton_to_abuth"] = c2a
    best["abuth_to_cotton"] = a2c
    best["exp_name"] = exp_name
    return best


def pct(x):
    return "{:.2f}%".format(x * 100.0)


def build_md(rows):
    lines = []
    lines.append("# SAFF F1/F2 High-Gain Ablation")
    lines.append("")
    lines.append("| Experiment | best mIoU | best mPA | best mF1 | cotton IoU | abuth IoU | cotton->abuth | abuth->cotton | best Iter | best epoch |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    valid = [r for r in rows if r is not None]
    valid = sorted(valid, key=lambda x: x["mIoU"], reverse=True)
    for r in valid:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                r["exp_name"],
                pct(r["mIoU"]),
                pct(r["mPA"]),
                pct(r["mF1"]),
                pct(r["cotton_iou"]),
                pct(r["abuth_iou"]),
                pct(r["cotton_to_abuth"]),
                pct(r["abuth_to_cotton"]),
                r["iter"],
                r["epoch"],
            )
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="./workdirs")
    parser.add_argument("--experiments", type=str, required=True,
                        help="comma-separated experiment names")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    names = [x.strip() for x in args.experiments.split(",") if x.strip()]
    rows = [summarize_exp(args.work_dir, n) for n in names]
    md = build_md(rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(md)
    print("Report generated: {}".format(args.output))
    print("Experiments requested: {}".format(len(names)))
    print("Experiments found: {}".format(sum(r is not None for r in rows)))


if __name__ == "__main__":
    main()
