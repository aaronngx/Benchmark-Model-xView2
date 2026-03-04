#!/usr/bin/env python3
"""
Evaluate VLM predictions against ground truth.

Reads one or more predictions CSVs (from run_vlm.py) and prints:
  - Per-class precision, recall, F1
  - Macro F1 (compare to CNN baseline 0.515)
  - Confusion matrix
  - Latency and token stats

Usage:
  python scripts/eval_vlm.py --predictions reports/vlm/predictions_openai_gpt_4o_mini.csv
  python scripts/eval_vlm.py --predictions reports/vlm/pred_a.csv reports/vlm/pred_b.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

CLASSES   = ["no-damage", "minor-damage", "major-damage", "destroyed"]
CNN_MACRO = 0.515   # CNN sampler_noweights 3-seed average (benchmark baseline)


def load_predictions(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_metrics(rows: list[dict]) -> dict:
    gt   = [r["gt_label"]   for r in rows]
    pred = [r["pred_label"] for r in rows]

    per_class = {}
    for cls in CLASSES:
        tp = sum(1 for g, p in zip(gt, pred) if g == cls and p == cls)
        fp = sum(1 for g, p in zip(gt, pred) if g != cls and p == cls)
        fn = sum(1 for g, p in zip(gt, pred) if g == cls and p != cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[cls] = {"tp": tp, "fp": fp, "fn": fn,
                          "prec": prec, "rec": rec, "f1": f1,
                          "n_gt": sum(1 for g in gt if g == cls)}

    macro_f1 = np.mean([per_class[c]["f1"] for c in CLASSES])
    accuracy  = sum(1 for g, p in zip(gt, pred) if g == p) / max(len(gt), 1)

    # Confusion matrix
    idx = {c: i for i, c in enumerate(CLASSES)}
    cm  = np.zeros((4, 4), dtype=int)
    for g, p in zip(gt, pred):
        if g in idx and p in idx:
            cm[idx[g]][idx[p]] += 1

    # Latency / token stats — VLM rows only (exclude filtered rows)
    vlm_rows  = [r for r in rows if not r.get("filter_applied")]
    latencies = [float(r.get("latency_ms", 0)) for r in vlm_rows if r.get("latency_ms")]
    tokens    = [int(r.get("tokens_used", 0))   for r in vlm_rows if r.get("tokens_used")]

    # Filter counts
    n_filtered_cnn = sum(1 for r in rows if r.get("filter_applied") == "cnn")
    n_filtered_ens = sum(1 for r in rows if r.get("filter_applied") == "ensemble")

    return {
        "n":             len(rows),
        "n_vlm":         len(vlm_rows),
        "n_filtered_cnn": n_filtered_cnn,
        "n_filtered_ens": n_filtered_ens,
        "accuracy":      accuracy,
        "macro_f1":      float(macro_f1),
        "per_class":     per_class,
        "confusion":     cm,
        "latency_mean":  np.mean(latencies) if latencies else 0,
        "latency_p95":   float(np.percentile(latencies, 95)) if latencies else 0,
        "tokens_mean":   np.mean(tokens) if tokens else 0,
        "parse_errors":  sum(1 for r in vlm_rows if r.get("parse_error")),
    }


def print_report(path: str, m: dict, model: str) -> None:
    print(f"\n{'='*65}")
    print(f"Model: {model}")
    print(f"File:  {path}")
    print(f"{'='*65}")
    print(f"  Buildings total:      {m['n']}")
    print(f"  VLM calls made:       {m['n_vlm']}")
    if m["n_filtered_cnn"] or m["n_filtered_ens"]:
        print(f"  Filtered by CNN:      {m['n_filtered_cnn']}")
        print(f"  Filtered by ensemble: {m['n_filtered_ens']}")
    print(f"  Accuracy:             {m['accuracy']:.1%}")
    print(f"  Macro F1:             {m['macro_f1']:.3f}  "
          f"({'above' if m['macro_f1'] > CNN_MACRO else 'below'} CNN baseline {CNN_MACRO})")
    print(f"  Parse errors:         {m['parse_errors']}")
    print(f"  Avg latency (VLM):    {m['latency_mean']:.0f} ms  "
          f"(p95={m['latency_p95']:.0f} ms)")
    print(f"  Avg tokens/VLM call:  {m['tokens_mean']:.0f}")

    print(f"\n  {'Class':<15} {'N':>5}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
    print(f"  {'-'*45}")
    for cls in CLASSES:
        pc = m["per_class"][cls]
        print(f"  {cls:<15} {pc['n_gt']:>5}  {pc['prec']:>6.3f}  "
              f"{pc['rec']:>6.3f}  {pc['f1']:>6.3f}")

    print(f"\n  Confusion matrix (rows=GT, cols=Pred):")
    header = "  " + " " * 16 + "  ".join(f"{c[:8]:>8}" for c in CLASSES)
    print(header)
    cm = m["confusion"]
    for i, cls in enumerate(CLASSES):
        row_str = "  " + f"{cls:<16}" + "  ".join(f"{cm[i][j]:>8}" for j in range(4))
        print(row_str)


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate VLM predictions vs ground truth")
    p.add_argument("--predictions", nargs="+", required=True,
                   help="One or more predictions CSV files from run_vlm.py")
    args = p.parse_args()

    for path in args.predictions:
        rows = load_predictions(path)
        if not rows:
            print(f"  WARN: no rows in {path}")
            continue
        model = rows[0].get("model", "unknown")
        m = compute_metrics(rows)
        print_report(path, m, model)

    if len(args.predictions) > 1:
        # Side-by-side macro F1 comparison
        print(f"\n{'='*65}")
        print("Model comparison (macro F1):")
        results = []
        for path in args.predictions:
            rows = load_predictions(path)
            if rows:
                model = rows[0].get("model", path)
                m = compute_metrics(rows)
                results.append((model, m["macro_f1"], m["tokens_mean"]))
        results.sort(key=lambda x: -x[1])
        print(f"  {'Model':<45} {'Macro F1':>8}  {'Avg tokens':>10}")
        print(f"  {'-'*65}")
        for model, f1, tok in results:
            marker = " <-- best" if f1 == results[0][1] else ""
            print(f"  {model:<45} {f1:>8.3f}  {tok:>10.0f}{marker}")
        print(f"  CNN baseline (sampler_noweights):             {CNN_MACRO:>8.3f}")


if __name__ == "__main__":
    main()
