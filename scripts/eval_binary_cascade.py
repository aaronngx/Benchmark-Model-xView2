#!/usr/bin/env python3
"""
Cascade inference + 4-class evaluation for the 3-stage binary cascade classifier.

Cascade logic:
  1. Stage 1 model → P(any-damage)
     if P(any-damage) <= s1_threshold: predict no-damage (0), done
  2. Stage 2 model → P(destroyed)
     if P(destroyed) >= s2_threshold: predict destroyed (3), done
       (soft routing: raise s2_threshold to 0.7–0.95 to pass ambiguous buildings to S3)
  3. Stage 3 model → P(major)
     if P(major) >= s3_threshold: predict major (2)
     else: predict minor (1)

Uses the same 80/20 tile-grouped val split as train_damage.py (seed=42 default).

Usage:
  python scripts/eval_binary_cascade.py \\
      --s1_ckpt models/binary_cascade/stage1/best.pt \\
      --s2_ckpt models/binary_cascade/stage2/best.pt \\
      --s3_ckpt models/binary_cascade/stage3/best.pt \\
      --sweep_thresholds

  # Soft routing sweep (optimize triage recall = f1_minor + f1_major):
  python scripts/eval_binary_cascade.py \\
      --s1_ckpt ... --s2_ckpt ... --s3_ckpt ... \\
      --sweep_thresholds --sweep_objective both

  # Fixed thresholds:
  python scripts/eval_binary_cascade.py \\
      --s1_ckpt ... --s2_ckpt ... --s3_ckpt ... \\
      --s1_threshold 0.4 --s2_threshold 0.8 --s3_threshold 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(ckpt_path: str, num_classes: int, device):
    import torch
    from disaster_bench.models.damage.classifiers import build_classifier
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_type = ckpt.get("model_type", "six_channel")
    model = build_classifier(model_type, num_classes=num_classes).to(device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def _infer_probs(model, loader, device) -> np.ndarray:
    """Run inference and return softmax probabilities (N, num_classes)."""
    import torch
    all_probs = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def _compute_metrics(preds, labels, num_classes=4):
    """Return (f1s, macro_f1, precs, recs) per class."""
    from collections import defaultdict
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for p, t in zip(preds, labels):
        if p == t:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    f1s, precs, recs = [], [], []
    for c in range(num_classes):
        pr = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        rc = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0
        f1s.append(f1); precs.append(pr); recs.append(rc)
    return f1s, float(np.mean(f1s)), precs, recs


def _cascade_predict(p_s1, p_s2, p_s3, s1_thr, s2_thr, s3_thr) -> np.ndarray:
    """
    Apply 3-stage cascade to produce 4-class predictions.

    p_s1: (N,) — P(any-damage) from Stage 1
    p_s2: (N,) — P(destroyed) from Stage 2
    p_s3: (N,) — P(major) from Stage 3
    """
    N = len(p_s1)
    preds = np.zeros(N, dtype=np.int64)
    for i in range(N):
        if p_s1[i] <= s1_thr:
            preds[i] = 0   # no-damage
        elif p_s2[i] >= s2_thr:
            preds[i] = 3   # destroyed
        elif p_s3[i] >= s3_thr:
            preds[i] = 2   # major
        else:
            preds[i] = 1   # minor
    return preds


def _error_propagation(labels, p_s1, p_s2, s1_thr, s2_thr):
    """Report how many true minor/major buildings are lost before Stage 3."""
    true_minor = np.where(labels == 1)[0]
    true_major = np.where(labels == 2)[0]

    # Lost at Stage 1: true minor/major predicted as no-damage
    lost_s1_minor = int((p_s1[true_minor] <= s1_thr).sum())
    lost_s1_major = int((p_s1[true_major] <= s1_thr).sum())

    # Remaining minor/major after Stage 1
    reach_s2_minor = true_minor[p_s1[true_minor] > s1_thr]
    reach_s2_major = true_major[p_s1[true_major] > s1_thr]

    # Lost at Stage 2: true minor/major predicted as destroyed
    lost_s2_minor = int((p_s2[reach_s2_minor] >= s2_thr).sum())
    lost_s2_major = int((p_s2[reach_s2_major] >= s2_thr).sum())

    reach_s3_minor = len(reach_s2_minor) - lost_s2_minor
    reach_s3_major = len(reach_s2_major) - lost_s2_major

    print("\n[Error Propagation]")
    print(f"  True minor: {len(true_minor)}  |  "
          f"Lost@S1: {lost_s1_minor}  Lost@S2: {lost_s2_minor}  "
          f"Reach S3: {reach_s3_minor}  "
          f"({100*reach_s3_minor/max(len(true_minor),1):.1f}%)")
    print(f"  True major: {len(true_major)}  |  "
          f"Lost@S1: {lost_s1_major}  Lost@S2: {lost_s2_major}  "
          f"Reach S3: {reach_s3_major}  "
          f"({100*reach_s3_major/max(len(true_major),1):.1f}%)")
    total_lost = lost_s1_minor + lost_s2_minor + lost_s1_major + lost_s2_major
    total_partial = len(true_minor) + len(true_major)
    pct_lost = 100 * total_lost / max(total_partial, 1)
    print(f"  Total lost before S3: {total_lost}/{total_partial} ({pct_lost:.1f}%)")
    return reach_s3_minor, reach_s3_major


def _print_results(label, preds, labels, s1_thr, s2_thr, s3_thr):
    f1s, macro_f1, precs, recs = _compute_metrics(preds, labels, num_classes=4)
    cls = ["no", "minor", "major", "dest"]
    print(f"\n[{label}]  s1={s1_thr:.2f}  s2={s2_thr:.2f}  s3={s3_thr:.2f}")
    print(f"  macro_F1={macro_f1:.4f}")
    print("  " + "  ".join(
        f"{c}: P={precs[i]:.3f} R={recs[i]:.3f} F1={f1s[i]:.3f}"
        for i, c in enumerate(cls)
    ))
    return macro_f1, f1s


def main() -> None:
    p = argparse.ArgumentParser(description="3-stage binary cascade evaluation")
    p.add_argument("--s1_ckpt",     required=True,
                   help="Stage 1 checkpoint (no-damage vs any-damage)")
    p.add_argument("--s2_ckpt",     required=True,
                   help="Stage 2 checkpoint (partial vs destroyed)")
    p.add_argument("--s3_ckpt",     required=True,
                   help="Stage 3 checkpoint (minor vs major)")
    p.add_argument("--s1_threshold", type=float, default=0.5,
                   help="Stage 1 threshold: P(any-damage) > t -> damaged  (default 0.5)")
    p.add_argument("--s2_threshold", type=float, default=0.5,
                   help="Stage 2 threshold: P(destroyed) >= t -> destroyed  (default 0.5)")
    p.add_argument("--s3_threshold", type=float, default=0.5,
                   help="Stage 3 threshold: P(major) >= t -> major  (default 0.5)")
    p.add_argument("--sweep_thresholds", action="store_true",
                   help="Grid-search all 3 thresholds and report best config")
    p.add_argument("--sweep_objective", default="macro_f1",
                   choices=["macro_f1", "triage", "both"],
                   help="Sweep objective: macro_f1 | triage (f1_minor+f1_major) | both (default: macro_f1)")
    p.add_argument("--index_csv",  default="data/processed/index.csv")
    p.add_argument("--crops_dir",  default="data/processed/crops_oracle")
    p.add_argument("--seed",       type=int, default=42,
                   help="Seed for 80/20 split (must match training seed; default 42)")
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--batch",      type=int, default=64)
    p.add_argument("--device",     default=None)
    args = p.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        print("torch not installed.", file=sys.stderr); sys.exit(1)

    from disaster_bench.data.dataset import (
        build_crop_records, train_val_split, CropDataset,
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---------------------------------------------------------------------------
    # Build full val set (80/20 split, same as train_damage.py default)
    # ---------------------------------------------------------------------------
    print("Building crop records...")
    records = build_crop_records(args.index_csv, args.crops_dir)
    print(f"  Total buildings: {len(records)}")

    _, val_recs = train_val_split(records, val_fraction=args.val_fraction, seed=args.seed)
    print(f"  Val buildings: {len(val_recs)}")
    from collections import Counter
    _vd = Counter(r["label"] for r in val_recs)
    for cls, cnt in sorted(_vd.items()):
        print(f"    {cls}: {cnt}")

    def _make_loader(recs, size):
        ds = CropDataset(recs, size=size, augment=False, preload=True)
        return DataLoader(ds, batch_size=args.batch, shuffle=False,
                          collate_fn=lambda batch: (
                              torch.from_numpy(np.stack([b[0] for b in batch])).float(),
                              torch.tensor([b[1] for b in batch], dtype=torch.long),
                          ), num_workers=0)

    # ---------------------------------------------------------------------------
    # Load models and run inference on full val set
    # ---------------------------------------------------------------------------
    print(f"\nLoading Stage 1: {args.s1_ckpt}")
    m1 = _load_model(args.s1_ckpt, num_classes=2, device=device)
    print(f"Loading Stage 2: {args.s2_ckpt}")
    m2 = _load_model(args.s2_ckpt, num_classes=2, device=device)
    print(f"Loading Stage 3: {args.s3_ckpt}")
    m3 = _load_model(args.s3_ckpt, num_classes=2, device=device)

    # Stage 1 uses size=128; Stage 3 uses size=256.
    # For S2 and S3, we run them on all val buildings (cascade filters at eval time).
    print("\nRunning Stage 1 inference (size=128)...")
    s1_probs  = _infer_probs(m1, _make_loader(val_recs, 128), device)
    p_anydmg  = s1_probs[:, 1]   # P(any-damage)

    print("Running Stage 2 inference (size=128)...")
    s2_probs  = _infer_probs(m2, _make_loader(val_recs, 128), device)
    p_dest    = s2_probs[:, 1]   # P(destroyed)

    print("Running Stage 3 inference (size=256)...")
    s3_probs  = _infer_probs(m3, _make_loader(val_recs, 256), device)
    p_major   = s3_probs[:, 1]   # P(major)

    labels = np.array([r["label_idx"] for r in val_recs])

    # ---------------------------------------------------------------------------
    # Per-stage binary metrics (sanity check)
    # ---------------------------------------------------------------------------
    print("\n[Stage 1 binary metrics]")
    s1_bin_labels = np.where(labels > 0, 1, 0)
    s1_preds_bin  = (p_anydmg > 0.5).astype(int)
    s1_f1s, s1_mf1, s1_pr, s1_rc = _compute_metrics(s1_preds_bin, s1_bin_labels, num_classes=2)
    print(f"  macro_F1={s1_mf1:.4f}  "
          f"nodmg: P={s1_pr[0]:.3f} R={s1_rc[0]:.3f} F1={s1_f1s[0]:.3f}  "
          f"anydmg: P={s1_pr[1]:.3f} R={s1_rc[1]:.3f} F1={s1_f1s[1]:.3f}")
    print(f"  recall(any-damage)={s1_rc[1]:.4f}  "
          f"[target: > 0.90]")

    print("\n[Stage 2 binary metrics — on partial+dest val buildings]")
    s2_mask       = labels >= 1                          # skip no-damage
    s2_bin_labels = np.where(labels[s2_mask] == 3, 1, 0)  # destroyed=1, partial=0
    s2_preds_bin  = (p_dest[s2_mask] >= 0.5).astype(int)
    s2_f1s, s2_mf1, s2_pr, s2_rc = _compute_metrics(s2_preds_bin, s2_bin_labels, num_classes=2)
    print(f"  macro_F1={s2_mf1:.4f}  "
          f"partial: P={s2_pr[0]:.3f} R={s2_rc[0]:.3f} F1={s2_f1s[0]:.3f}  "
          f"dest: P={s2_pr[1]:.3f} R={s2_rc[1]:.3f} F1={s2_f1s[1]:.3f}")
    print(f"  recall(partial-damage)={s2_rc[0]:.4f}  "
          f"[target: > 0.70]")

    print("\n[Stage 3 binary metrics — on minor+major val buildings]")
    s3_mask       = (labels == 1) | (labels == 2)
    s3_bin_labels = np.where(labels[s3_mask] == 2, 1, 0)  # major=1, minor=0
    s3_preds_bin  = (p_major[s3_mask] >= 0.5).astype(int)
    s3_f1s, s3_mf1, s3_pr, s3_rc = _compute_metrics(s3_preds_bin, s3_bin_labels, num_classes=2)
    print(f"  macro_F1={s3_mf1:.4f}  "
          f"minor: P={s3_pr[0]:.3f} R={s3_rc[0]:.3f} F1={s3_f1s[0]:.3f}  "
          f"major: P={s3_pr[1]:.3f} R={s3_rc[1]:.3f} F1={s3_f1s[1]:.3f}")
    print(f"  n_minor={int(s3_mask.sum()) - int(s3_bin_labels.sum())}  "
          f"n_major={int(s3_bin_labels.sum())}")

    # ---------------------------------------------------------------------------
    # Default thresholds: error propagation + 4-class eval
    # ---------------------------------------------------------------------------
    _error_propagation(labels, p_anydmg, p_dest,
                       args.s1_threshold, args.s2_threshold)

    preds_default = _cascade_predict(p_anydmg, p_dest, p_major,
                                     args.s1_threshold, args.s2_threshold, args.s3_threshold)
    _print_results("Default thresholds", preds_default, labels,
                   args.s1_threshold, args.s2_threshold, args.s3_threshold)
    print(f"\n  CNN 4-class baseline: macro_F1 = 0.515  f1_minor ~0.04  f1_major ~0.10")

    # ---------------------------------------------------------------------------
    # Threshold sweep
    # ---------------------------------------------------------------------------
    if args.sweep_thresholds:
        # s2 grid extended to 0.95 to cover soft-routing territory
        s1_grid = [0.3, 0.4, 0.5, 0.6, 0.7]
        s2_grid = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.80, 0.85, 0.90, 0.95]
        s3_grid = [0.3, 0.4, 0.5, 0.6, 0.7]

        objectives = (["macro_f1", "triage"] if args.sweep_objective == "both"
                      else [args.sweep_objective])

        for obj in objectives:
            print(f"\n[Threshold sweep — objective={obj}] Searching s1 x s2 x s3 ...")
            best_score = -1.0
            best_cfg   = None
            best_f1s   = None

            for s1t in s1_grid:
                for s2t in s2_grid:
                    for s3t in s3_grid:
                        preds = _cascade_predict(p_anydmg, p_dest, p_major, s1t, s2t, s3t)
                        f1s, mf1, _, _ = _compute_metrics(preds, labels, num_classes=4)
                        score = mf1 if obj == "macro_f1" else (f1s[1] + f1s[2])
                        if score > best_score:
                            best_score = score
                            best_cfg   = (s1t, s2t, s3t)
                            best_f1s   = f1s

            s1t, s2t, s3t = best_cfg
            preds_best = _cascade_predict(p_anydmg, p_dest, p_major, s1t, s2t, s3t)
            _error_propagation(labels, p_anydmg, p_dest, s1t, s2t)
            label = ("Best macro_F1 config" if obj == "macro_f1"
                     else "Best triage config (f1_minor + f1_major)")
            _print_results(label, preds_best, labels, s1t, s2t, s3t)
            score_str = (f"macro_F1={best_score:.4f}"
                         if obj == "macro_f1"
                         else f"triage_score={best_score:.4f}  "
                              f"(f1_minor={best_f1s[1]:.3f} + f1_major={best_f1s[2]:.3f})")
            print(f"\n  Best ({obj}): s1={s1t}  s2={s2t}  s3={s3t}  {score_str}")
            print(f"  f1s: no={best_f1s[0]:.3f}  minor={best_f1s[1]:.3f}  "
                  f"major={best_f1s[2]:.3f}  dest={best_f1s[3]:.3f}")
            print(f"\n  CNN 4-class baseline: macro_F1 = 0.515  f1_minor ~0.04  f1_major ~0.10")
            delta = float(np.mean(best_f1s)) - 0.515
            print(f"  Delta macro_F1 vs CNN: {delta:+.4f}")


if __name__ == "__main__":
    main()
