#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import sys


def to_float(text):
    s = str(text).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_latest_round(metrics_tsv):
    if not os.path.isfile(metrics_tsv):
        raise FileNotFoundError(metrics_tsv)
    with open(metrics_tsv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    current = None
    rounds = []
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
        name = str(row.get("class", "")).strip()
        if not name:
            continue
        current["classes"][name] = {
            "IoU": to_float(row.get("IoU")),
            "Acc": to_float(row.get("Acc")),
        }
    if current is not None:
        rounds.append(current)
    return rounds[-1] if rounds else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_tsv", required=True)
    parser.add_argument("--json_out", default="")
    parser.add_argument(
        "--required_classes",
        default="background,cotton,abuth,canger,machixian,longkui,tianxuanhua,niujincao",
    )
    args = parser.parse_args()
    required = [x.strip() for x in args.required_classes.split(",") if x.strip()]

    latest = parse_latest_round(args.metrics_tsv)
    if latest is None:
        payload = {"ok": False, "reason": "no validation round"}
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        print("HEALTH_CHECK_FAIL")
        sys.exit(2)

    bad = []
    for cname in required:
        metrics = latest["classes"].get(cname)
        if metrics is None:
            bad.append({"class": cname, "issue": "missing"})
            continue
        for key in ("IoU", "Acc"):
            v = metrics.get(key)
            if v is None or math.isnan(v):
                bad.append({"class": cname, "issue": f"{key}_nan"})
            elif v <= 0:
                bad.append({"class": cname, "issue": f"{key}_non_positive", "value": v})

    payload = {
        "ok": len(bad) == 0,
        "iter": latest["iter"],
        "epoch": latest["epoch"],
        "aAcc": latest["aAcc"],
        "mIoU": latest["mIoU"],
        "mAcc": latest["mAcc"],
        "bad_metrics": bad,
    }
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    if payload["ok"]:
        print("HEALTH_CHECK_PASS")
        sys.exit(0)
    print("HEALTH_CHECK_FAIL")
    print(json.dumps(payload, ensure_ascii=False))
    sys.exit(2)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import sys


def to_float(text):
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_latest_round(metrics_tsv):
    if not os.path.isfile(metrics_tsv):
        raise FileNotFoundError(f"metrics.tsv not found: {metrics_tsv}")
    with open(metrics_tsv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        return None

    rounds = []
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
                "mF1": to_float(row.get("mF1")),
                "classes": {},
            }
        if current is None:
            continue
        cls = str(row.get("class", "")).strip()
        if not cls:
            continue
        current["classes"][cls] = {
            "IoU": to_float(row.get("IoU")),
            "Acc": to_float(row.get("Acc")),
            "F1": to_float(row.get("F1")),
        }
    if current is not None:
        rounds.append(current)
    return rounds[-1] if rounds else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_tsv", required=True)
    parser.add_argument(
        "--required_classes",
        default="background,cotton,abuth,canger,machixian,longkui,tianxuanhua,niujincao",
    )
    parser.add_argument("--json_out", default="")
    args = parser.parse_args()

    required = [x.strip() for x in args.required_classes.split(",") if x.strip()]
    latest = parse_latest_round(args.metrics_tsv)
    if latest is None:
        msg = "No validation rounds found in metrics.tsv"
        payload = {"ok": False, "reason": msg}
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        print(msg)
        sys.exit(2)

    bad = []
    for name in required:
        m = latest["classes"].get(name)
        if m is None:
            bad.append({"class": name, "issue": "missing"})
            continue
        for k in ("IoU", "Acc"):
            v = m.get(k)
            if v is None or math.isnan(v):
                bad.append({"class": name, "issue": f"{k}_nan"})
            elif v <= 0:
                bad.append({"class": name, "issue": f"{k}_non_positive", "value": v})

    payload = {
        "ok": len(bad) == 0,
        "iter": latest["iter"],
        "epoch": latest["epoch"],
        "aAcc": latest["aAcc"],
        "mIoU": latest["mIoU"],
        "mAcc": latest["mAcc"],
        "classes_checked": required,
        "bad_metrics": bad,
    }
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    if payload["ok"]:
        print("HEALTH_CHECK_PASS")
        sys.exit(0)
    print("HEALTH_CHECK_FAIL")
    print(json.dumps(payload, ensure_ascii=False))
    sys.exit(2)


if __name__ == "__main__":
    main()
