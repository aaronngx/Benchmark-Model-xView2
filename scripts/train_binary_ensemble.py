#!/usr/bin/env python3
"""
Train one-vs-rest binary ensemble classifiers for minor and/or major damage.

Improvements over v1:
  --standardize     : StandardScaler on embeddings before LR training
  --hnm_rounds INT  : hard-negative mining (refit on high-scoring FPs)
  --ratio INT       : negatives per positive per member (try 1 or 3)
  --cv_folds_path   : tile-grouped K-fold CV mode (pools all positives; use for major)

Metrics reported:
  Threshold sweep (precision / recall / F1)
  Top-K recall  (K=50, 100): of N positives in val, how many are in the top-K scored?
  FP/1k recall  at FP/1k-no = 5, 10, 20: recall at operating points by false-positive rate

Usage (single-split, standard):
  python scripts/train_binary_ensemble.py --cls both --n_models 50 --ratio 1 --standardize

Usage (CV mode for major to pool all val positives):
  python scripts/train_binary_ensemble.py --cls 2 --cv_folds_path data/processed/cv_folds_k5_seed42.json
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

CLASS_NAMES  = {1: "minor", 2: "major"}
BASELINE_F1  = {1: 0.040, 2: 0.103}   # mae80 s1/s2/s3 mean F1
TOP_KS       = [50, 100]
FP1K_TARGETS = [5, 10, 20]


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def f1_score_binary(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def compute_ranking_metrics(
    val_scores: np.ndarray,   # (N_val,) ensemble mean score
    y_val_bin: np.ndarray,    # (N_val,) binary: 1=positive class
    y_val_full: np.ndarray,   # (N_val,) 4-class labels (to identify no-damage=0)
) -> dict:
    """Top-K recall and FP/1k-no-damage recall."""
    n_pos   = int(y_val_bin.sum())
    n_nodmg = int((y_val_full == 0).sum())

    # Sort descending by score
    order = np.argsort(-val_scores)
    y_sorted     = y_val_bin[order]
    nodmg_sorted = (y_val_full[order] == 0)

    results: dict = {"n_pos_val": n_pos, "n_nodmg_val": n_nodmg}

    # --- Top-K recall
    top_k_results = {}
    cum_pos = 0
    for k in sorted(TOP_KS):
        if k > len(y_sorted):
            top_k_results[k] = None
            continue
        tp_at_k = int(y_sorted[:k].sum())
        rec = tp_at_k / n_pos if n_pos > 0 else 0.0
        top_k_results[k] = {"tp": tp_at_k, "recall": round(rec, 4)}
    results["top_k"] = top_k_results

    # --- FP/1k recall (walk sorted list)
    cumtp  = np.cumsum(y_sorted)           # TP at each rank
    cumfp  = np.cumsum(nodmg_sorted)       # FP (no-damage) at each rank
    fp1k   = cumfp / n_nodmg * 1000 if n_nodmg > 0 else np.zeros_like(cumfp, dtype=float)

    fp1k_results = {}
    for target in FP1K_TARGETS:
        # First rank where FP/1k >= target
        idx = np.searchsorted(fp1k, target, side="left")
        if idx >= len(cumtp):
            tp_here = int(cumtp[-1])
            fp_here = int(cumfp[-1])
        else:
            tp_here = int(cumtp[idx])
            fp_here = int(cumfp[idx])
        rec = tp_here / n_pos if n_pos > 0 else 0.0
        fp1k_results[target] = {
            "recall": round(rec, 4),
            "tp": tp_here,
            "fp": fp_here,
            "actual_fp1k": round(float(fp1k[min(idx, len(fp1k)-1)]), 1),
        }
    results["fp1k"] = fp1k_results
    return results


def print_ranking_metrics(metrics: dict, cls_name: str) -> None:
    print(f"\n  --- Ranking metrics ({cls_name}) ---")
    print(f"  Val positives: {metrics['n_pos_val']}  |  "
          f"No-damage buildings: {metrics['n_nodmg_val']}")

    print(f"\n  Top-K recall:")
    for k, v in sorted(metrics["top_k"].items()):
        if v is None:
            print(f"    Top-{k:<4}: N/A (fewer than {k} val buildings)")
        else:
            print(f"    Top-{k:<4}: {v['tp']} TP -> recall={v['recall']:.3f}")

    print(f"\n  Recall at FP/1k-no-damage target:")
    for target, v in sorted(metrics["fp1k"].items()):
        print(f"    FP/1k={target:>3}: recall={v['recall']:.3f}  "
              f"(TP={v['tp']}, FP={v['fp']}, actual FP/1k={v['actual_fp1k']})")


# ---------------------------------------------------------------------------
# Ensemble training helpers
# ---------------------------------------------------------------------------

def train_members(
    Z_train: np.ndarray,
    y_train_bin: np.ndarray,
    pos_idx: np.ndarray,
    neg_idx: np.ndarray,
    n_models: int,
    ratio: int,
    C: float,
    seed: int,
    hnm_rounds: int = 0,
    hnm_threshold: float = 0.3,
) -> list:
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    n_pos = len(pos_idx)
    n_neg_per = max(1, ratio * n_pos)

    current_neg_pool = neg_idx.copy()

    for rnd in range(max(1, hnm_rounds + 1)):
        if rnd > 0:
            # Hard negative mining: score all negatives, keep high-scoring ones
            scores_neg = np.mean(
                np.stack([m.predict_proba(Z_train[current_neg_pool])[:, 1]
                          for m in models], axis=0),
                axis=0,
            )
            hard = current_neg_pool[scores_neg >= hnm_threshold]
            if len(hard) >= n_neg_per:
                current_neg_pool = hard
                print(f"  [HNM round {rnd}] Hard negatives: {len(hard)}")
            else:
                print(f"  [HNM round {rnd}] Only {len(hard)} hard negatives "
                      f"(< {n_neg_per} needed) — keeping full pool.")

        models = []
        for i in range(n_models):
            boot_pos = rng.choice(pos_idx, size=n_pos, replace=True)
            n_sample = min(n_neg_per, len(current_neg_pool))
            sub_neg  = rng.choice(current_neg_pool, size=n_sample, replace=False)
            idx = np.concatenate([boot_pos, sub_neg])
            X_i = Z_train[idx]
            y_i = y_train_bin[idx]
            lr = LogisticRegression(C=C, max_iter=1000, solver="lbfgs",
                                    random_state=int(seed + i + rnd * 1000))
            lr.fit(X_i, y_i)
            models.append(lr)

        if (rnd == 0 and hnm_rounds == 0) or rnd == hnm_rounds:
            break   # last round

    return models


def ensemble_predict(models: list, Z: np.ndarray) -> np.ndarray:
    return np.mean(
        np.stack([m.predict_proba(Z)[:, 1] for m in models], axis=0),
        axis=0,
    )


def run_threshold_sweep(val_scores, y_val_bin) -> tuple[list[dict], float, float]:
    thresholds = np.arange(0.05, 0.96, 0.05)
    rows, best_f1, best_thresh = [], 0.0, 0.5
    print(f"\n  {'Thresh':>7}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'TP':>4}  {'FP':>4}")
    for thr in thresholds:
        y_pred = (val_scores >= thr).astype(np.int32)
        prec, rec, f1 = f1_score_binary(y_val_bin, y_pred)
        tp = int(((y_pred == 1) & (y_val_bin == 1)).sum())
        fp = int(((y_pred == 1) & (y_val_bin == 0)).sum())
        rows.append({"threshold": round(float(thr), 2), "precision": round(prec, 4),
                     "recall": round(rec, 4), "f1": round(f1, 4), "tp": tp, "fp": fp})
        print(f"  {thr:>7.2f}  {prec:>6.3f}  {rec:>6.3f}  {f1:>6.3f}  {tp:>4d}  {fp:>4d}")
        if f1 > best_f1:
            best_f1, best_thresh = f1, float(thr)
    return rows, best_f1, best_thresh


def save_artifacts(
    out_cls: Path,
    models: list,
    calib,
    uid_val: np.ndarray,
    tile_id_val: np.ndarray,
    y_val_bin: np.ndarray,
    val_scores: np.ndarray,
    calib_scores: np.ndarray,
    sweep_rows: list[dict],
    summary: dict,
) -> None:
    out_cls.mkdir(parents=True, exist_ok=True)
    with open(out_cls / "ensemble.pkl", "wb") as f:
        pickle.dump(models, f)
    with open(out_cls / "calibrator.pkl", "wb") as f:
        pickle.dump(calib, f)
    with open(out_cls / "val_scores.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["uid", "tile_id", "y_true",
                                               "raw_score", "calib_score"])
        writer.writeheader()
        for u, t, yt, rs, cs in zip(uid_val, tile_id_val, y_val_bin,
                                    val_scores, calib_scores):
            writer.writerow({"uid": u, "tile_id": t, "y_true": int(yt),
                             "raw_score": round(float(rs), 6),
                             "calib_score": round(float(cs), 6)})
    with open(out_cls / "threshold_sweep.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "precision", "recall",
                                               "f1", "tp", "fp"])
        writer.writeheader()
        writer.writerows(sweep_rows)
    with open(out_cls / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# Single-split run (original mode)
# ---------------------------------------------------------------------------

def run_ensemble(
    cls: int,
    Z_train: np.ndarray, y_train: np.ndarray,
    Z_val:   np.ndarray, y_val:   np.ndarray,
    uid_val: np.ndarray, tile_id_val: np.ndarray,
    n_models: int, ratio: int, C: float, seed: int,
    hnm_rounds: int, hnm_threshold: float,
    out_dir: Path,
) -> dict:
    from sklearn.linear_model import LogisticRegression

    cls_name = CLASS_NAMES[cls]
    out_cls  = out_dir / cls_name

    y_train_bin = (y_train == cls).astype(np.int32)
    y_val_bin   = (y_val   == cls).astype(np.int32)
    pos_idx = np.where(y_train_bin == 1)[0]
    neg_idx = np.where(y_train_bin == 0)[0]
    n_pos = len(pos_idx)

    if n_pos == 0:
        print(f"  [{cls_name}] No positive examples in training split — skipping.")
        return {}

    print(f"\n{'='*60}")
    print(f"  Class: {cls_name}  train pos={n_pos}  neg_pool={len(neg_idx)}")
    print(f"  {n_models} models  ratio={ratio}  C={C}  hnm_rounds={hnm_rounds}")
    print(f"{'='*60}")

    models = train_members(Z_train, y_train_bin, pos_idx, neg_idx,
                           n_models, ratio, C, seed, hnm_rounds, hnm_threshold)

    val_scores = ensemble_predict(models, Z_val)

    # Platt calibration on val
    calib = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    calib.fit(val_scores.reshape(-1, 1), y_val_bin)
    calib_scores = calib.predict_proba(val_scores.reshape(-1, 1))[:, 1]

    print(f"\n  Threshold sweep (val positives={y_val_bin.sum()}, total val={len(y_val_bin)}):")
    sweep_rows, best_f1, best_thresh = run_threshold_sweep(val_scores, y_val_bin)

    ranking = compute_ranking_metrics(val_scores, y_val_bin, y_val)
    print_ranking_metrics(ranking, cls_name)

    print(f"\n  Best F1={best_f1:.4f} @ threshold={best_thresh:.2f}")
    delta = best_f1 - BASELINE_F1[cls]
    print(f"  4-class baseline F1={BASELINE_F1[cls]:.3f}  delta={'+' if delta>=0 else ''}{delta:.3f}")

    summary = {
        "mode": "single_split",
        "cls": cls, "cls_name": cls_name,
        "n_models": n_models, "ratio": ratio, "C": C,
        "hnm_rounds": hnm_rounds,
        "n_pos_train": int(n_pos), "n_neg_train": int(len(neg_idx)),
        "n_pos_val": int(y_val_bin.sum()),
        "best_f1": round(best_f1, 4),
        "best_threshold": round(best_thresh, 2),
        "baseline_f1_4class": BASELINE_F1[cls],
        "delta_vs_baseline": round(best_f1 - BASELINE_F1[cls], 4),
        "ranking": ranking,
    }
    save_artifacts(out_cls, models, calib, uid_val, tile_id_val, y_val_bin,
                   val_scores, calib_scores, sweep_rows, summary)
    print(f"\n  Saved -> {out_cls}/")
    return summary


# ---------------------------------------------------------------------------
# Cross-validation mode (pools all positives — use for major with only 6 val)
# ---------------------------------------------------------------------------

def run_ensemble_cv(
    cls: int,
    Z_all: np.ndarray, y_all: np.ndarray,
    tile_id_all: np.ndarray, uid_all: np.ndarray,
    tile_to_fold: dict[str, int],
    n_folds: int,
    n_models: int, ratio: int, C: float, seed: int,
    hnm_rounds: int, hnm_threshold: float,
    out_dir: Path,
) -> dict:
    cls_name = CLASS_NAMES[cls]
    out_cls  = out_dir / f"{cls_name}_cv"
    out_cls.mkdir(parents=True, exist_ok=True)

    # Assign fold to each building (-1 if tile not in fold map — excluded)
    fold_assign = np.array([tile_to_fold.get(t, -1) for t in tile_id_all], dtype=np.int32)
    known_mask  = fold_assign >= 0

    print(f"\n{'='*60}")
    print(f"  Class: {cls_name} — CV MODE ({n_folds} folds, pooled val)")
    total_pos = int((y_all[known_mask] == cls).sum())
    print(f"  Total {cls_name} buildings in fold map: {total_pos}")
    print(f"  {n_models} models/fold  ratio={ratio}  C={C}  hnm_rounds={hnm_rounds}")
    print(f"{'='*60}")

    all_uid_val, all_tile_val, all_y_bin, all_scores = [], [], [], []

    for fold in range(n_folds):
        val_mask   = known_mask & (fold_assign == fold)
        train_mask = known_mask & (fold_assign != fold)

        Z_tr = Z_all[train_mask];  y_tr = y_all[train_mask]
        Z_va = Z_all[val_mask];    y_va = y_all[val_mask]

        y_tr_bin = (y_tr == cls).astype(np.int32)
        y_va_bin = (y_va == cls).astype(np.int32)

        pos_idx = np.where(y_tr_bin == 1)[0]
        neg_idx = np.where(y_tr_bin == 0)[0]
        n_pos = len(pos_idx)

        print(f"\n  Fold {fold}: train pos={n_pos}  val pos={int(y_va_bin.sum())}  "
              f"val total={len(y_va_bin)}")

        if n_pos == 0:
            # No positives in train — score val as 0
            fold_scores = np.zeros(len(y_va_bin), dtype=np.float32)
        else:
            models = train_members(Z_tr, y_tr_bin, pos_idx, neg_idx,
                                   n_models, ratio, C, seed + fold * 10000,
                                   hnm_rounds, hnm_threshold)
            fold_scores = ensemble_predict(models, Z_va)

        all_uid_val.append(uid_all[val_mask])
        all_tile_val.append(tile_id_all[val_mask])
        all_y_bin.append(y_va_bin)
        all_scores.append(fold_scores)

    # Pool across all folds
    uid_val    = np.concatenate(all_uid_val)
    tile_val   = np.concatenate(all_tile_val)
    y_val_bin  = np.concatenate(all_y_bin)
    val_scores = np.concatenate(all_scores)

    # Need 4-class labels for FP/1k (no-damage identification)
    # Rebuild from cls label: 0=no-damage in y_all that we collected
    # We stored y_va which contains 4-class labels — rebuild:
    y_val_full = np.concatenate([
        y_all[known_mask & (fold_assign == f)] for f in range(n_folds)
    ])

    print(f"\n  --- Pooled val: {len(y_val_bin)} buildings, {int(y_val_bin.sum())} positives ---")

    # Platt calibration on pooled val
    from sklearn.linear_model import LogisticRegression
    calib = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    calib.fit(val_scores.reshape(-1, 1), y_val_bin)
    calib_scores = calib.predict_proba(val_scores.reshape(-1, 1))[:, 1]

    print(f"\n  Threshold sweep (pooled val, positives={y_val_bin.sum()}):")
    sweep_rows, best_f1, best_thresh = run_threshold_sweep(val_scores, y_val_bin)

    ranking = compute_ranking_metrics(val_scores, y_val_bin, y_val_full)
    print_ranking_metrics(ranking, cls_name)

    print(f"\n  Best F1={best_f1:.4f} @ threshold={best_thresh:.2f}")
    delta = best_f1 - BASELINE_F1[cls]
    print(f"  4-class baseline F1={BASELINE_F1[cls]:.3f}  delta={'+' if delta>=0 else ''}{delta:.3f}")

    summary = {
        "mode": "cv",
        "cls": cls, "cls_name": cls_name,
        "n_folds": n_folds,
        "n_models_per_fold": n_models, "ratio": ratio, "C": C,
        "hnm_rounds": hnm_rounds,
        "n_pos_total": total_pos,
        "best_f1": round(best_f1, 4),
        "best_threshold": round(best_thresh, 2),
        "baseline_f1_4class": BASELINE_F1[cls],
        "delta_vs_baseline": round(best_f1 - BASELINE_F1[cls], 4),
        "ranking": ranking,
    }
    save_artifacts(out_cls, [], calib, uid_val, tile_val, y_val_bin,
                   val_scores, calib_scores, sweep_rows, summary)
    print(f"\n  Saved -> {out_cls}/")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Train one-vs-rest binary ensemble for minor/major damage")
    p.add_argument("--embeddings_npz", default="data/processed/embeddings_mae80_s1.npz")
    p.add_argument("--cls", default="both", choices=["1", "2", "both"])
    p.add_argument("--n_models",  type=int,   default=50)
    p.add_argument("--ratio",     type=int,   default=1,
                   help="Negatives per positive per ensemble member (default 1)")
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--C",         type=float, default=0.1,
                   help="LR regularization strength (default 0.1)")
    p.add_argument("--standardize", action="store_true",
                   help="Apply StandardScaler to embeddings before training")
    p.add_argument("--hnm_rounds", type=int, default=0,
                   help="Hard negative mining rounds (default 0 = off)")
    p.add_argument("--hnm_threshold", type=float, default=0.3,
                   help="Score threshold for hard negatives (default 0.3)")
    p.add_argument("--cv_folds_path", type=str, default=None,
                   help="Path to cv_folds JSON; triggers CV mode (pools all positives)")
    p.add_argument("--out_dir",   default="models/binary_ensemble")
    args = p.parse_args()

    try:
        from sklearn.preprocessing import StandardScaler  # noqa: F401
    except ImportError:
        print("scikit-learn not installed. Run: pip install scikit-learn", file=sys.stderr)
        sys.exit(1)

    # ---------------------------------------------------------------- load
    npz = np.load(args.embeddings_npz, allow_pickle=True)
    Z         = npz["Z"].astype(np.float32)
    label_idx = npz["label_idx"].astype(np.int32)
    split     = npz["split"]
    tile_id   = npz["tile_id"]
    uid       = npz["uid"]
    print(f"Loaded: {Z.shape}  from {args.embeddings_npz}")

    classes  = [1, 2] if args.cls == "both" else [int(args.cls)]
    out_dir  = Path(args.out_dir)
    summaries = {}

    # -------------------------------------------------------- CV mode
    if args.cv_folds_path:
        cv_data = json.loads(Path(args.cv_folds_path).read_text())
        tile_to_fold = cv_data["tile_to_fold"]
        n_folds      = cv_data["k"]

        Z_use = Z.copy()
        if args.standardize:
            from sklearn.preprocessing import StandardScaler
            # Fit scaler on full dataset (no leakage risk — only embeddings, no labels)
            scaler = StandardScaler()
            Z_use = scaler.fit_transform(Z_use).astype(np.float32)
            print("StandardScaler applied (fit on all embeddings).")

        for cls in classes:
            s = run_ensemble_cv(
                cls, Z_use, label_idx, tile_id, uid, tile_to_fold, n_folds,
                args.n_models, args.ratio, args.C, args.seed,
                args.hnm_rounds, args.hnm_threshold, out_dir,
            )
            summaries[CLASS_NAMES[cls]] = s

    else:
        # ------------------------------------------- single-split mode
        train_mask = (split == "train")
        val_mask   = (split == "val")
        Z_tr, y_tr = Z[train_mask], label_idx[train_mask]
        Z_va, y_va = Z[val_mask],   label_idx[val_mask]
        uid_va     = uid[val_mask]
        tid_va     = tile_id[val_mask]

        print(f"Train: {train_mask.sum()} | Val: {val_mask.sum()}")

        if args.standardize:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            Z_tr = scaler.fit_transform(Z_tr).astype(np.float32)
            Z_va = scaler.transform(Z_va).astype(np.float32)
            print("StandardScaler applied (fit on train, transform val).")

        for cls in classes:
            s = run_ensemble(
                cls, Z_tr, y_tr, Z_va, y_va, uid_va, tid_va,
                args.n_models, args.ratio, args.C, args.seed,
                args.hnm_rounds, args.hnm_threshold, out_dir,
            )
            summaries[CLASS_NAMES[cls]] = s

    # ---------------------------------------------------------------- table
    print(f"\n{'='*60}")
    print(f"  FINAL COMPARISON")
    print(f"  {'Class':<10} {'Baseline':>9} {'Ensemble':>9} {'Delta':>7} {'Thresh':>7}")
    print(f"  {'-'*46}")
    for cls in classes:
        name = CLASS_NAMES[cls]
        s = summaries.get(name, {})
        if s:
            d = s["delta_vs_baseline"]
            print(f"  {name:<10} {s['baseline_f1_4class']:>9.3f} {s['best_f1']:>9.3f} "
                  f"{('+' if d>=0 else '')+str(round(d,3)):>7} {s['best_threshold']:>7.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
