#!/usr/bin/env python3
"""
Aggregate 5-fold CV results from val_metrics.jsonl files.

For each fold run:
  - Reads runs/cv5_fold{f}/val_metrics.jsonl
  - Picks the best epoch (max macro_f1)
  - Extracts: macro_f1, f1_minor, f1_major,
              fp_per_1000_no_minor, fp_per_1000_no_major,
              pred_minor, pred_major

Writes:
  reports/cv5/best_epoch_per_fold.csv
  reports/cv5/mean_std_across_folds.csv

Usage:
  python scripts/summarize_cv.py
  python scripts/summarize_cv.py --runs_dir runs --pattern cv5_fold --out_dir reports/cv5
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


METRICS = [
    "macro_f1",
    "f1_minor",
    "f1_major",
    "fp_per_1000_no_minor",
    "fp_per_1000_no_major",
    "pred_minor",
    "pred_major",
]


def load_best(jsonl_path: Path) -> dict:
    """Return the row with the highest macro_f1 from a val_metrics.jsonl."""
    if not jsonl_path.exists():
        return {}
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return {}
    return max(rows, key=lambda r: r.get("macro_f1", -1.0))


def mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(values) / n
    if n == 1:
        return m, float("nan")
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return m, math.sqrt(var)


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize 5-fold CV results")
    p.add_argument("--runs_dir", default="runs",
                   help="Directory containing per-fold run subdirectories")
    p.add_argument("--pattern",  default="cv5_fold",
                   help="Run-id prefix, e.g. 'cv5_fold' → cv5_fold0..cv5_fold4")
    p.add_argument("--k",        type=int, default=5, help="Number of folds")
    p.add_argument("--out_dir",  default="reports/cv5",
                   help="Directory to write CSV outputs")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_fold_rows = []
    missing_folds = []

    for fold in range(args.k):
        run_id   = f"{args.pattern}{fold}"
        jsonl_p  = runs_dir / run_id / "val_metrics.jsonl"
        best_row = load_best(jsonl_p)

        if not best_row:
            print(f"WARNING: no data for fold {fold} ({jsonl_p})", file=sys.stderr)
            missing_folds.append(fold)
            continue

        row = {"fold": fold, "run_id": run_id, "best_epoch": best_row.get("epoch", "?")}
        for m in METRICS:
            row[m] = best_row.get(m, float("nan"))
        per_fold_rows.append(row)
        print(f"Fold {fold} | epoch={row['best_epoch']:>3} | "
              f"macro_f1={row['macro_f1']:.4f} | "
              f"f1_minor={row['f1_minor']:.4f} | "
              f"f1_major={row['f1_major']:.4f} | "
              f"fp/1k-minor={row['fp_per_1000_no_minor']:.1f} | "
              f"fp/1k-major={row['fp_per_1000_no_major']:.1f}")

    if missing_folds:
        print(f"\n{len(missing_folds)} fold(s) missing: {missing_folds}", file=sys.stderr)
        if len(per_fold_rows) == 0:
            print("No folds found — cannot compute summary.", file=sys.stderr)
            sys.exit(1)

    # Write best_epoch_per_fold.csv
    fold_csv = out_dir / "best_epoch_per_fold.csv"
    fieldnames = ["fold", "run_id", "best_epoch"] + METRICS
    with open(fold_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_fold_rows)
    print(f"\nSaved {fold_csv}")

    # Compute mean ± std for each metric
    summary_rows = []
    print("\n--- Mean ± Std across folds ---")
    for m in METRICS:
        vals = [r[m] for r in per_fold_rows if not math.isnan(float(r[m]))]
        mu, sd = mean_std([float(v) for v in vals])
        sd_str = f"{sd:.4f}" if not math.isnan(sd) else "n/a"
        print(f"  {m:30s}: {mu:.4f} ± {sd_str}  (n={len(vals)})")
        summary_rows.append({
            "metric": m,
            "mean":   round(mu, 4) if not math.isnan(mu) else "",
            "std":    round(sd, 4) if not math.isnan(sd) else "n/a",
            "n_folds": len(vals),
        })

    # Write mean_std_across_folds.csv
    summary_csv = out_dir / "mean_std_across_folds.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "mean", "std", "n_folds"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSaved {summary_csv}")


if __name__ == "__main__":
    main()
