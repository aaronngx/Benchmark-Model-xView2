#!/usr/bin/env python3
"""
Soft fusion of 4-class CNN probabilities with minor/major binary ensemble scores.

Reads:
  --cnn_probs    reports/fusion/cnn_probs.csv
  --minor_scores models/binary_ensemble/minor_cv/val_scores.csv
  --major_scores models/binary_ensemble/major_cv/val_scores.csv

Join rule: inner-join on (tile_id, uid). Only rows present in cnn_probs.csv
(the val split) are evaluated. The ensemble CSVs cover all 8316 buildings.

Three fusion modes:
  A — additive probability injection:
        p_fused[1] += alpha * s_minor_calib
        p_fused[2] += beta  * s_major_calib
        renormalize, then argmax

  B — logit-space boost:
        logp[1] += alpha * log(s_minor / (1 - s_minor))
        logp[2] += beta  * log(s_major / (1 - s_major))
        softmax, then argmax

  C — confidence-gated override (CNN stays in control unless uncertain):
        if confidence_max < conf_threshold:
            if s_major_calib >= major_gate: pred = 2
            elif s_minor_calib >= minor_gate: pred = 1
            else: pred = argmax(p)
        else: pred = argmax(p)

Usage:
  python scripts/fuse_cnn_ensemble.py \\
    --cnn_probs    reports/fusion/cnn_probs.csv \\
    --minor_scores models/binary_ensemble/minor_cv/val_scores.csv \\
    --major_scores models/binary_ensemble/major_cv/val_scores.csv \\
    --sweep
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(preds: np.ndarray, labels: np.ndarray):
    """Returns (f1s[4], macro_f1, precs[4], recs[4])."""
    from collections import defaultdict
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for pr, gt in zip(preds, labels):
        if pr == gt: tp[gt] += 1
        else: fp[pr] += 1; fn[gt] += 1
    f1s, precs, recs = [], [], []
    for c in range(4):
        pr = tp[c]/(tp[c]+fp[c]) if tp[c]+fp[c] > 0 else 0.0
        rc = tp[c]/(tp[c]+fn[c]) if tp[c]+fn[c] > 0 else 0.0
        f1 = 2*pr*rc/(pr+rc) if pr+rc > 0 else 0.0
        f1s.append(f1); precs.append(pr); recs.append(rc)
    return f1s, float(np.mean(f1s)), precs, recs


def _triage_score(f1s):
    return f1s[1] + f1s[2]  # f1_minor + f1_major


# ---------------------------------------------------------------------------
# Fusion functions
# ---------------------------------------------------------------------------

def _fuse_mode_a(p: np.ndarray, s_minor: np.ndarray, s_major: np.ndarray,
                 alpha: float, beta: float) -> np.ndarray:
    """Additive probability injection."""
    p_fused = p.copy()
    p_fused[:, 1] += alpha * s_minor
    p_fused[:, 2] += beta  * s_major
    p_fused = np.clip(p_fused, 0.0, None)
    row_sums = p_fused.sum(axis=1, keepdims=True)
    p_fused /= np.where(row_sums > 0, row_sums, 1.0)
    return p_fused.argmax(axis=1)


def _fuse_mode_b(p: np.ndarray, s_minor: np.ndarray, s_major: np.ndarray,
                 alpha: float, beta: float) -> np.ndarray:
    """Logit-space boost."""
    eps = 1e-8
    logp = np.log(np.clip(p, eps, 1.0))
    logp[:, 1] += alpha * np.log(s_minor / (1.0 - s_minor + eps) + eps)
    logp[:, 2] += beta  * np.log(s_major / (1.0 - s_major + eps) + eps)
    # softmax
    logp -= logp.max(axis=1, keepdims=True)
    p_fused = np.exp(logp)
    p_fused /= p_fused.sum(axis=1, keepdims=True)
    return p_fused.argmax(axis=1)


def _fuse_mode_c(p: np.ndarray, s_minor: np.ndarray, s_major: np.ndarray,
                 confidence: np.ndarray, conf_threshold: float,
                 minor_gate: float, major_gate: float) -> np.ndarray:
    """Confidence-gated override."""
    preds = p.argmax(axis=1).copy()
    uncertain = confidence < conf_threshold
    for i in np.where(uncertain)[0]:
        if s_major[i] >= major_gate:
            preds[i] = 2
        elif s_minor[i] >= minor_gate:
            preds[i] = 1
        # else keep CNN argmax
    return preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fuse CNN probs with ensemble scores")
    parser.add_argument("--cnn_probs",    required=True)
    parser.add_argument("--minor_scores", required=True)
    parser.add_argument("--major_scores", required=True)
    parser.add_argument("--sweep", action="store_true",
                        help="Grid-sweep all fusion modes and report best configs")
    parser.add_argument("--mode", choices=["A", "B", "C"], default=None,
                        help="Single mode to run (ignored if --sweep)")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta",  type=float, default=1.0)
    parser.add_argument("--conf_threshold", type=float, default=0.7)
    parser.add_argument("--minor_gate",     type=float, default=0.1)
    parser.add_argument("--major_gate",     type=float, default=0.1)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load CNN probs (val split only)
    # ------------------------------------------------------------------
    cnn_rows: dict[tuple[str, str], dict] = {}
    with open(args.cnn_probs, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["tile_id"], row["uid"])
            cnn_rows[key] = row
    print(f"CNN probs loaded: {len(cnn_rows)} rows (val split)")

    # ------------------------------------------------------------------
    # Load ensemble scores (all 8316 buildings)
    # ------------------------------------------------------------------
    def _load_scores(path: str) -> dict[tuple[str, str], float]:
        scores = {}
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row["tile_id"], row["uid"])
                scores[key] = float(row["calib_score"])
        return scores

    minor_scores = _load_scores(args.minor_scores)
    major_scores = _load_scores(args.major_scores)
    print(f"Minor ensemble: {len(minor_scores)} rows")
    print(f"Major ensemble: {len(major_scores)} rows")

    # ------------------------------------------------------------------
    # Inner join: keep only val buildings that have ensemble scores
    # ------------------------------------------------------------------
    matched_keys = [k for k in cnn_rows if k in minor_scores and k in major_scores]
    matched_keys.sort()  # deterministic order
    print(f"Matched rows (inner join): {len(matched_keys)}")

    if len(matched_keys) == 0:
        print("ERROR: No matching rows. Check that tile_id/uid columns match.", file=sys.stderr)
        sys.exit(1)

    # Build arrays
    p         = np.array([[float(cnn_rows[k]["p_nodmg"]),
                            float(cnn_rows[k]["p_minor"]),
                            float(cnn_rows[k]["p_major"]),
                            float(cnn_rows[k]["p_dest"])] for k in matched_keys])
    s_minor   = np.array([minor_scores[k] for k in matched_keys])
    s_major   = np.array([major_scores[k] for k in matched_keys])
    labels    = np.array([int(cnn_rows[k]["gt_label_idx"]) for k in matched_keys])
    confidence= np.array([float(cnn_rows[k]["confidence_max"]) for k in matched_keys])

    # ------------------------------------------------------------------
    # Baseline: CNN-only argmax
    # ------------------------------------------------------------------
    baseline_preds = p.argmax(axis=1)
    f1s_base, mf1_base, _, _ = _compute_metrics(baseline_preds, labels)
    print(f"\n[CNN Baseline]")
    print(f"  macro_F1={mf1_base:.4f}  "
          f"f1_minor={f1s_base[1]:.3f}  f1_major={f1s_base[2]:.3f}  "
          f"f1_dest={f1s_base[3]:.3f}")
    print(f"  triage_score={_triage_score(f1s_base):.3f}")

    if not args.sweep:
        # Single eval
        mode = args.mode or "A"
        if mode == "A":
            preds = _fuse_mode_a(p, s_minor, s_major, args.alpha, args.beta)
        elif mode == "B":
            preds = _fuse_mode_b(p, s_minor, s_major, args.alpha, args.beta)
        else:
            preds = _fuse_mode_c(p, s_minor, s_major, confidence,
                                 args.conf_threshold, args.minor_gate, args.major_gate)
        f1s, mf1, _, _ = _compute_metrics(preds, labels)
        print(f"\n[Mode {mode}]")
        print(f"  macro_F1={mf1:.4f}  f1_minor={f1s[1]:.3f}  f1_major={f1s[2]:.3f}  "
              f"f1_dest={f1s[3]:.3f}  triage={_triage_score(f1s):.3f}")
        return

    # ------------------------------------------------------------------
    # Sweep
    # ------------------------------------------------------------------
    ALPHA_BETA_GRID = [0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    CONF_GRID       = [0.5, 0.6, 0.7, 0.8, 0.9]
    GATE_GRID       = [0.01, 0.05, 0.1, 0.2, 0.5]

    results: list[dict] = []

    print("\n[Mode A sweep — additive probability injection]")
    for alpha in ALPHA_BETA_GRID:
        for beta in ALPHA_BETA_GRID:
            preds = _fuse_mode_a(p, s_minor, s_major, alpha, beta)
            f1s, mf1, _, _ = _compute_metrics(preds, labels)
            results.append(dict(
                mode="A", alpha=alpha, beta=beta,
                conf_threshold=None, minor_gate=None, major_gate=None,
                macro_f1=mf1, f1_minor=f1s[1], f1_major=f1s[2], f1_dest=f1s[3],
                triage=_triage_score(f1s),
            ))

    print("[Mode B sweep — logit-space boost]")
    for alpha in ALPHA_BETA_GRID:
        for beta in ALPHA_BETA_GRID:
            preds = _fuse_mode_b(p, s_minor, s_major, alpha, beta)
            f1s, mf1, _, _ = _compute_metrics(preds, labels)
            results.append(dict(
                mode="B", alpha=alpha, beta=beta,
                conf_threshold=None, minor_gate=None, major_gate=None,
                macro_f1=mf1, f1_minor=f1s[1], f1_major=f1s[2], f1_dest=f1s[3],
                triage=_triage_score(f1s),
            ))

    print("[Mode C sweep — confidence-gated override]")
    for conf_thr in CONF_GRID:
        for minor_g in GATE_GRID:
            for major_g in GATE_GRID:
                preds = _fuse_mode_c(p, s_minor, s_major, confidence,
                                     conf_thr, minor_g, major_g)
                f1s, mf1, _, _ = _compute_metrics(preds, labels)
                results.append(dict(
                    mode="C", alpha=None, beta=None,
                    conf_threshold=conf_thr, minor_gate=minor_g, major_gate=major_g,
                    macro_f1=mf1, f1_minor=f1s[1], f1_major=f1s[2], f1_dest=f1s[3],
                    triage=_triage_score(f1s),
                ))

    print(f"Total configs evaluated: {len(results)}")

    # ------------------------------------------------------------------
    # Report: best by macro_F1 and best by triage_score
    # ------------------------------------------------------------------
    def _fmt(r: dict) -> str:
        if r["mode"] in ("A", "B"):
            params = f"alpha={r['alpha']}  beta={r['beta']}"
        else:
            params = (f"conf_thr={r['conf_threshold']}  "
                      f"minor_gate={r['minor_gate']}  major_gate={r['major_gate']}")
        return (f"mode={r['mode']}  {params}  "
                f"macro_F1={r['macro_f1']:.4f}  "
                f"f1_minor={r['f1_minor']:.3f}  f1_major={r['f1_major']:.3f}  "
                f"f1_dest={r['f1_dest']:.3f}  triage={r['triage']:.3f}")

    best_macro  = max(results, key=lambda r: r["macro_f1"])
    best_triage = max(results, key=lambda r: r["triage"])

    print(f"\n[Best by macro_F1]")
    print(f"  {_fmt(best_macro)}")
    delta_macro = best_macro["macro_f1"] - mf1_base
    print(f"  Delta vs CNN baseline: {delta_macro:+.4f}")

    print(f"\n[Best by triage_score (f1_minor + f1_major)]")
    print(f"  {_fmt(best_triage)}")
    delta_triage = best_triage["triage"] - _triage_score(f1s_base)
    print(f"  Delta triage vs CNN baseline: {delta_triage:+.3f}")

    # Verdict
    print("\n[Verdict]")
    if best_macro["macro_f1"] >= 0.515 and (
        best_macro["f1_minor"] > f1s_base[1] or best_macro["f1_major"] > f1s_base[2]
    ):
        print("  SUCCESS: macro_F1 >= 0.515 and at least one rare class improved.")
    elif best_macro["macro_f1"] >= 0.505 and best_macro["f1_major"] > f1s_base[2]:
        print("  ACCEPTABLE: macro_F1 >= 0.505 and f1_major improved.")
    else:
        print("  FAILURE: No config beats baseline. Negative result — fusion does not help.")

    # Save results CSV
    out_path = Path("reports/fusion/sweep_results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["mode", "alpha", "beta", "conf_threshold", "minor_gate",
                      "major_gate", "macro_f1", "f1_minor", "f1_major", "f1_dest", "triage"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSweep results saved to {out_path}")


if __name__ == "__main__":
    main()
