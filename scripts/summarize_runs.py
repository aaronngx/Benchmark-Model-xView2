#!/usr/bin/env python3
"""
Summarize val_metrics.jsonl outputs from benchmark runs.

Outputs:
  reports/tables/best_epoch_per_run.csv   — one row per run_id at best macro_F1 epoch
  reports/tables/mean_std_by_method.csv   — mean ± std grouped by method (3-seed groups)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Config: which runs to include and how to group them into methods
# ---------------------------------------------------------------------------
RUNS_DIR = Path("runs")
OUT_DIR  = Path("reports/tables")

# All expected run_ids (seeded benchmark set)
RUN_IDS = [
    "baseline_s1", "baseline_s2", "baseline_s3",
    "sampler_capped_s1", "sampler_capped_s2", "sampler_capped_s3",
    "sampler_noweights_s1", "sampler_noweights_s2", "sampler_noweights_s3",
]

# Method groups for mean/std table
METHOD_GROUPS: dict[str, list[str]] = {
    "baseline":          ["baseline_s1", "baseline_s2", "baseline_s3"],
    "sampler_capped":    ["sampler_capped_s1", "sampler_capped_s2", "sampler_capped_s3"],
    "sampler_noweights": ["sampler_noweights_s1", "sampler_noweights_s2", "sampler_noweights_s3"],
}

METRICS = [
    "macro_f1",
    "f1_minor", "f1_major",
    "fp_per_1000_no_minor", "fp_per_1000_no_major",
    "pred_minor", "pred_major",
]


def load_best_epoch(run_id: str) -> dict:
    """Load val_metrics.jsonl for run_id and return the row with max macro_f1."""
    path = RUNS_DIR / run_id / "val_metrics.jsonl"
    if not path.exists():
        print(f"ERROR: missing {path}", file=sys.stderr)
        sys.exit(1)

    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        print(f"ERROR: {path} is empty", file=sys.stderr)
        sys.exit(1)

    best = max(rows, key=lambda r: r["macro_f1"])
    return {"run_id": run_id, **{m: best.get(m, float("nan")) for m in METRICS},
            "best_epoch": best["epoch"]}


def write_csv(rows: list[dict], path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(c, "")) for c in columns) + "\n")
    print(f"Wrote {path}")


def main() -> None:
    # -----------------------------------------------------------------------
    # 1) best_epoch_per_run.csv
    # -----------------------------------------------------------------------
    best_rows = [load_best_epoch(rid) for rid in RUN_IDS]

    cols_best = ["run_id", "best_epoch"] + METRICS
    write_csv(best_rows, OUT_DIR / "best_epoch_per_run.csv", cols_best)

    # Print a quick table to stdout
    print("\n--- Best epoch per run ---")
    hdr = f"{'run_id':<26} {'ep':>3}  {'mF1':>6}  {'F1min':>6}  {'F1maj':>6}  "
    hdr += f"{'FP/1k_min':>9}  {'FP/1k_maj':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in best_rows:
        print(f"{r['run_id']:<26} {r['best_epoch']:>3}  "
              f"{r['macro_f1']:>6.4f}  {r['f1_minor']:>6.4f}  {r['f1_major']:>6.4f}  "
              f"{r['fp_per_1000_no_minor']:>9.1f}  {r['fp_per_1000_no_major']:>9.1f}")

    # -----------------------------------------------------------------------
    # 2) mean_std_by_method.csv
    # -----------------------------------------------------------------------
    lookup = {r["run_id"]: r for r in best_rows}

    method_rows = []
    for method, run_ids in METHOD_GROUPS.items():
        available = [rid for rid in run_ids if rid in lookup]
        if len(available) < 2:
            print(f"  Skipping {method}: only {len(available)} seed(s) available")
            continue
        vals: dict[str, list[float]] = {m: [] for m in METRICS}
        for rid in available:
            for m in METRICS:
                vals[m].append(float(lookup[rid].get(m, float("nan"))))
        row: dict = {"method": method, "n_seeds": len(available)}
        for m in METRICS:
            arr = np.array(vals[m])
            row[f"{m}_mean"] = round(float(np.nanmean(arr)), 4)
            row[f"{m}_std"]  = round(float(np.nanstd(arr, ddof=1)), 4)
        method_rows.append(row)

    cols_method = ["method", "n_seeds"]
    for m in METRICS:
        cols_method += [f"{m}_mean", f"{m}_std"]
    write_csv(method_rows, OUT_DIR / "mean_std_by_method.csv", cols_method)

    # Print mean±std table
    print("\n--- Mean ± std by method ---")
    for r in method_rows:
        print(f"\n{r['method']} (n={r['n_seeds']})")
        for m in METRICS:
            print(f"  {m:<28} {r[f'{m}_mean']:>7.4f} ± {r[f'{m}_std']:.4f}")


if __name__ == "__main__":
    main()
