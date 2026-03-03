#!/usr/bin/env python3
"""
Aggregate cascade outer-test metrics from 5-fold nested CV.

Reads:  runs/cv5_fold{f}/test_metrics_cascade.json  (for f in 0..k-1)
Writes: reports/cv5/cascade_test_summary.csv
        reports/cv5/cascade_test_per_fold.csv

Usage:
  python scripts/summarize_cascade_cv.py
  python scripts/summarize_cascade_cv.py --runs_dir runs --out_dir reports/cv5
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


PER_FOLD_FIELDS = [
    "fold",
    "macro_f1",
    "f1_minor", "f1_major", "f1_dest",
    "FN_damage", "FN_severe",
    "recall_damage", "recall_severe",
    "FP_dmg_per_1k_no", "FP_sev_per_1k_nonsevere",
    "n_minor_test", "n_major_test",
    "tau_damage", "tau_severe",
    "pooled_n_calib",
]

SUMMARY_METRICS = [
    "macro_f1",
    "f1_minor", "f1_major", "f1_dest",
    "FN_damage", "FN_severe",
    "FP_dmg_per_1k_no", "FP_sev_per_1k_nonsevere",
]


def load_fold(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def mean_std(vals: list[float]) -> tuple[float, float]:
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(vals) / n
    if n == 1:
        return m, float("nan")
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return m, math.sqrt(var)


def extract_row(fold: int, d: dict) -> dict:
    pc = d.get("per_class", {})
    thr = d.get("thresholds_used") or d.get("thresholds") or {}
    # Pooled calib size: check thresholds_crossfit.json (optional, not in this file)
    return {
        "fold":                   fold,
        "macro_f1":               d.get("macro_f1"),
        "f1_minor":               pc.get("minor-damage", {}).get("f1"),
        "f1_major":               pc.get("major-damage", {}).get("f1"),
        "f1_dest":                pc.get("destroyed",    {}).get("f1"),
        "FN_damage":              d.get("FN_damage"),
        "FN_severe":              d.get("FN_severe"),
        "recall_damage":          d.get("recall_damage"),
        "recall_severe":          d.get("recall_severe"),
        "FP_dmg_per_1k_no":       d.get("FP_damage_per_1000_no"),
        "FP_sev_per_1k_nonsevere": d.get("FP_severe_per_1000_nonsevere"),
        "n_minor_test":           d.get("n_minor_test"),
        "n_major_test":           d.get("n_major_test"),
        "tau_damage":             thr.get("tau_damage"),
        "tau_severe":             thr.get("tau_severe"),
        "pooled_n_calib":         None,  # filled below if thresholds_crossfit.json exists
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize cascade nested-CV outer-test results")
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--pattern",  default="cv5_fold",
                   help="Run-id prefix (default: cv5_fold → cv5_fold0..cv5_fold4)")
    p.add_argument("--k",        type=int, default=5)
    p.add_argument("--out_dir",  default="reports/cv5")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for fold in range(args.k):
        run_id = f"{args.pattern}{fold}"
        metrics_path = runs_dir / run_id / "test_metrics_cascade.json"
        xfit_path    = runs_dir / run_id / "thresholds_crossfit.json"

        d = load_fold(metrics_path)
        if d is None:
            print(f"WARNING: {metrics_path} not found — fold {fold} skipped")
            continue

        row = extract_row(fold, d)

        # Enrich with pooled-calib size from thresholds_crossfit.json if present
        if xfit_path.exists():
            xfit = json.loads(xfit_path.read_text())
            row["pooled_n_calib"] = xfit.get("n_pooled_samples")

        rows.append(row)

        fn_ok = row["FN_damage"] == 0 and row["FN_severe"] == 0
        fn_tag = "OK" if fn_ok else f"FN_dmg={row['FN_damage']} FN_sev={row['FN_severe']} FAIL"
        print(
            f"Fold {fold} | macro_f1={row['macro_f1']:.4f} | "
            f"f1_minor={row['f1_minor']:.4f} f1_major={row['f1_major']:.4f} "
            f"f1_dest={row['f1_dest']:.4f} | "
            f"FP_dmg={row['FP_dmg_per_1k_no']:.1f}/1k "
            f"FP_sev={row['FP_sev_per_1k_nonsevere']:.1f}/1k | "
            f"tau_d={row['tau_damage']:.5f} tau_s={row['tau_severe']:.5f} | "
            f"NeverMiss: {fn_tag}"
        )

    if not rows:
        print("No cascade test metrics found.")
        return

    # Per-fold CSV
    per_fold_csv = out_dir / "cascade_test_per_fold.csv"
    with open(per_fold_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PER_FOLD_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {per_fold_csv}")

    # Summary mean ± std
    print("\n--- Mean ± Std across folds ---")
    summary_rows = []
    for m in SUMMARY_METRICS:
        vals = [float(r[m]) for r in rows if r.get(m) is not None]
        mu, sd = mean_std(vals)
        sd_str = f"{sd:.4f}" if not math.isnan(sd) else "n/a"
        print(f"  {m:30s}: {mu:.4f} ± {sd_str}  (n={len(vals)})")
        summary_rows.append({
            "metric": m,
            "mean":   round(mu, 4) if not math.isnan(mu) else "",
            "std":    sd_str,
            "n_folds": len(vals),
        })

    summary_csv = out_dir / "cascade_test_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "mean", "std", "n_folds"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved {summary_csv}")

    # Never-miss audit
    fn_violations = [r["fold"] for r in rows
                     if r.get("FN_damage") or r.get("FN_severe")]
    if fn_violations:
        print(f"\n!! NEVER-MISS VIOLATION in folds: {fn_violations}")
    else:
        print(f"\nOK: FN_damage=0 and FN_severe=0 across all {len(rows)} folds.")


if __name__ == "__main__":
    main()
