#!/usr/bin/env python3
"""
Tile-cluster bootstrap confidence intervals.

Groups predictions by tile_id, resamples tiles with replacement,
and computes minor/major precision/recall/F1 + FP/1k-no distributions.
Reports median + 95% CI across 1000 bootstrap iterations.

Usage:
  python scripts/bootstrap_tile_ci.py \
    --preds_csv reports/lowo/preds_lowo_train_socal_test_santarosa.csv \
    --run_id lowo_train_socal_test_santarosa \
    --out_json reports/lowo/bootstrap_lowo_train_socal_test_santarosa.json

  # Run both LOWO runs and write combined summary:
  python scripts/bootstrap_tile_ci.py --all_lowo
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


# ---- metric helpers --------------------------------------------------------

def prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def compute_metrics(buildings: list[dict]) -> dict | None:
    """
    Given a list of {y_true, y_pred} dicts, compute a 4×4 confusion matrix
    and return minor/major precision/recall/F1 + FP/1k-no.
    Returns None if N_no == 0 (skip iteration).
    """
    NO, MINOR, MAJOR, DEST = 0, 1, 2, 3
    cm = [[0] * 4 for _ in range(4)]
    for b in buildings:
        t, p = b["y_true"], b["y_pred"]
        cm[t][p] += 1

    N_no = sum(cm[NO])
    if N_no == 0:
        return None

    # minor
    tp_min = cm[MINOR][MINOR]
    fp_min = sum(cm[r][MINOR] for r in range(4)) - tp_min
    fn_min = sum(cm[MINOR])   - tp_min
    pr_min, re_min, f1_min = prf1(tp_min, fp_min, fn_min)
    fp_per_1k_no_min = 1000.0 * cm[NO][MINOR] / N_no

    # major
    tp_maj = cm[MAJOR][MAJOR]
    fp_maj = sum(cm[r][MAJOR] for r in range(4)) - tp_maj
    fn_maj = sum(cm[MAJOR])   - tp_maj
    pr_maj, re_maj, f1_maj = prf1(tp_maj, fp_maj, fn_maj)
    fp_per_1k_no_maj = 1000.0 * cm[NO][MAJOR] / N_no

    return {
        "minor": {
            "precision":     pr_min,
            "recall":        re_min,
            "f1":            f1_min,
            "fp_per_1k_no":  fp_per_1k_no_min,
        },
        "major": {
            "precision":     pr_maj,
            "recall":        re_maj,
            "f1":            f1_maj,
            "fp_per_1k_no":  fp_per_1k_no_maj,
        },
    }


# ---- bootstrap core --------------------------------------------------------

def run_bootstrap(preds_csv: Path, run_id: str, out_json: Path,
                  n_boot: int = 1000, seed: int = 123) -> dict:
    # Load CSV
    tiles: dict[str, list[dict]] = defaultdict(list)
    with open(preds_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tiles[row["tile_id"]].append({
                "y_true": int(row["y_true"]),
                "y_pred": int(row["y_pred"]),
            })

    tile_ids = list(tiles.keys())
    n_tiles  = len(tile_ids)
    print(f"\n[{run_id}] {n_tiles} tiles, "
          f"{sum(len(v) for v in tiles.values())} buildings")

    # Observed (full sample) metrics
    all_buildings = [b for bldgs in tiles.values() for b in bldgs]
    obs = compute_metrics(all_buildings)
    if obs:
        print(f"  Observed: minor F1={obs['minor']['f1']:.3f}  "
              f"major F1={obs['major']['f1']:.3f}  "
              f"FP/1k-no minor={obs['minor']['fp_per_1k_no']:.1f}  "
              f"FP/1k-no major={obs['major']['fp_per_1k_no']:.1f}")

    # Bootstrap
    rng = random.Random(seed)
    CLASS_KEYS   = ["minor", "major"]
    METRIC_KEYS  = ["precision", "recall", "f1", "fp_per_1k_no"]
    dist: dict[str, dict[str, list[float]]] = {
        cls: {m: [] for m in METRIC_KEYS} for cls in CLASS_KEYS
    }
    skipped = 0

    for _ in range(n_boot):
        sampled_tiles = rng.choices(tile_ids, k=n_tiles)
        boot_buildings = [b for t in sampled_tiles for b in tiles[t]]
        result = compute_metrics(boot_buildings)
        if result is None:
            skipped += 1
            continue
        for cls in CLASS_KEYS:
            for m in METRIC_KEYS:
                dist[cls][m].append(result[cls][m])

    if skipped:
        print(f"  Skipped {skipped}/{n_boot} iterations (N_no==0)")

    # Summarise
    def summarise(values: list[float]) -> dict:
        s = sorted(values)
        n = len(s)
        if n == 0:
            return {"median": None, "ci95": [None, None]}
        med  = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
        lo   = s[int(0.025 * n)]
        hi   = s[min(int(0.975 * n), n - 1)]
        return {"median": round(med, 4), "ci95": [round(lo, 4), round(hi, 4)]}

    metrics_out: dict = {}
    for cls in CLASS_KEYS:
        metrics_out[cls] = {}
        for m in METRIC_KEYS:
            s = summarise(dist[cls][m])
            metrics_out[cls][m] = s
            print(f"  {cls:5s} {m:15s}: median={s['median']}  "
                  f"95% CI [{s['ci95'][0]}, {s['ci95'][1]}]  "
                  f"(n={len(dist[cls][m])})")

    result_json = {
        "run_id":      run_id,
        "n_boot":      n_boot,
        "tile_count":  n_tiles,
        "seed":        seed,
        "skipped":     skipped,
        "metrics":     metrics_out,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2)
    print(f"  Saved {out_json}")

    return result_json


# ---- summary CSV -----------------------------------------------------------

def write_summary_csv(results: list[dict], out_csv: Path) -> None:
    CLASS_KEYS  = ["minor", "major"]
    METRIC_KEYS = ["precision", "recall", "f1", "fp_per_1k_no"]
    fieldnames  = ["run_id"]
    for cls in CLASS_KEYS:
        for m in METRIC_KEYS:
            fieldnames += [f"{cls}_{m}_median",
                           f"{cls}_{m}_ci_lo",
                           f"{cls}_{m}_ci_hi"]

    rows = []
    for r in results:
        row = {"run_id": r["run_id"]}
        for cls in CLASS_KEYS:
            for m in METRIC_KEYS:
                s = r["metrics"][cls][m]
                row[f"{cls}_{m}_median"] = s["median"]
                row[f"{cls}_{m}_ci_lo"]  = s["ci95"][0]
                row[f"{cls}_{m}_ci_hi"]  = s["ci95"][1]
        rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved summary -> {out_csv}")


# ---- CLI -------------------------------------------------------------------

LOWO_RUNS = [
    {
        "run_id":    "lowo_train_socal_test_santarosa",
        "preds_csv": "reports/lowo/preds_lowo_train_socal_test_santarosa.csv",
        "out_json":  "reports/lowo/bootstrap_lowo_train_socal_test_santarosa.json",
    },
    {
        "run_id":    "lowo_train_santarosa_test_socal",
        "preds_csv": "reports/lowo/preds_lowo_train_santarosa_test_socal.csv",
        "out_json":  "reports/lowo/bootstrap_lowo_train_santarosa_test_socal.json",
    },
]


def main() -> None:
    p = argparse.ArgumentParser(description="Tile-cluster bootstrap CIs")
    p.add_argument("--preds_csv", type=str, default=None,
                   help="Path to predictions CSV")
    p.add_argument("--run_id",    type=str, default=None,
                   help="Run identifier (used in output filename)")
    p.add_argument("--out_json",  type=str, default=None,
                   help="Output JSON path")
    p.add_argument("--n_boot",    type=int, default=1000)
    p.add_argument("--seed",      type=int, default=123)
    p.add_argument("--all_lowo",  action="store_true",
                   help="Run bootstrap for both LOWO runs and write summary CSV")
    p.add_argument("--summary_csv", default="reports/lowo/bootstrap_summary.csv")
    args = p.parse_args()

    if args.all_lowo:
        results = []
        for run in LOWO_RUNS:
            preds = Path(run["preds_csv"])
            if not preds.exists():
                print(f"WARNING: {preds} not found — skipping", file=sys.stderr)
                continue
            r = run_bootstrap(preds, run["run_id"], Path(run["out_json"]),
                              args.n_boot, args.seed)
            results.append(r)
        if results:
            write_summary_csv(results, Path(args.summary_csv))
        return

    # Single-run mode
    if not args.preds_csv or not args.run_id:
        p.error("--preds_csv and --run_id are required (or use --all_lowo)")

    out_json = Path(args.out_json) if args.out_json else \
               Path("reports/lowo") / f"bootstrap_{args.run_id}.json"

    result = run_bootstrap(Path(args.preds_csv), args.run_id, out_json,
                           args.n_boot, args.seed)

    # Also append/create a summary CSV with this one run
    write_summary_csv([result], Path(args.summary_csv))


if __name__ == "__main__":
    main()
