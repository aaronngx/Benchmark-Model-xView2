#!/usr/bin/env python3
"""
Select the N most uncertain val-split buildings from cnn_probs.csv.

Uncertainty metric: margin = p_top1 - p_top2 (lower = more uncertain).

Outputs building_id lists (tile_id:uid format) sorted by ascending margin
(most uncertain first), for use with run_vlm.py --building_ids_path.

Usage:
  python scripts/select_uncertain_cases.py \\
    --cnn_probs reports/fusion/cnn_probs.csv \\
    --topk 50 100 200
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Select uncertain buildings for VLM")
    parser.add_argument("--cnn_probs", default="reports/fusion/cnn_probs.csv",
                        help="CSV from export_cnn_probs.py")
    parser.add_argument("--topk", type=int, nargs="+", default=[50, 100, 200],
                        help="Sizes of output lists (default: 50 100 200)")
    parser.add_argument("--out_dir", default="reports/vlm",
                        help="Output directory for building ID files")
    parser.add_argument("--metric", choices=["margin", "entropy"], default="margin",
                        help="Uncertainty metric: margin (default) or entropy")
    args = parser.parse_args()

    # Load and sort by uncertainty
    rows = []
    with open(args.cnn_probs, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    print(f"Loaded {len(rows)} val-split buildings from {args.cnn_probs}")

    if args.metric == "margin":
        # ascending margin = most uncertain first
        rows.sort(key=lambda r: float(r["margin_top1_top2"]))
        print("Sorting by margin (ascending = most uncertain first)")
    else:
        # descending entropy = most uncertain first
        rows.sort(key=lambda r: float(r["entropy"]), reverse=True)
        print("Sorting by entropy (descending = most uncertain first)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for k in sorted(args.topk):
        subset = rows[:k]
        metric_name = args.metric
        out_path = out_dir / f"uncertain_bottom{k}_ids.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            for row in subset:
                f.write(row["building_id"] + "\n")
        print(f"  Wrote {len(subset)} IDs to {out_path}")

        # Print breakdown of GT labels in the uncertain subset
        from collections import Counter
        label_counts = Counter(row["gt_label"] for row in subset)
        print(f"    GT label breakdown: {dict(label_counts)}")
        margin_vals = [float(r["margin_top1_top2"]) for r in subset]
        import statistics
        print(f"    Margin range: [{min(margin_vals):.3f}, {max(margin_vals):.3f}]  "
              f"mean={statistics.mean(margin_vals):.3f}")


if __name__ == "__main__":
    main()
