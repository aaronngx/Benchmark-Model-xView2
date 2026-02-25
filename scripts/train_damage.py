#!/usr/bin/env python3
"""
Train damage classifier on oracle crops.
Ref §2.1: Supervised (non-LLM) baselines.

Supports model types:
  six_channel   — [preRGB || postRGB] 6-ch CNN (default)
  pre_post_diff — [preRGB || postRGB || |diff|] 9-ch CNN
  siamese       — dual-stream encoder + fusion
  centroid_patch— fixed-size patch from polygon centroid (6-ch CNN input)

New flags (all default to off/scratch — existing behavior is fully preserved):
  --seed                     global RNG seed for reproducibility (default: 42)
  --run_id                   write val_metrics.jsonl to runs/<run_id>/
  --use_sampler              WeightedRandomSampler for train loader (Step 1)
  --log_batch_class_counts   log first 50 train-batch label histograms, epoch 1 (Step 2B)
  --weight_mode              normalized_invfreq (default) | capped_floored | none (Step 3)
  --w_min / --w_max          clipping bounds for capped_floored (Step 3)
  --use_hard_negative_mining boost no-dmg hard negatives each epoch (Step 4)
  --hnm_mult                 boost multiplier for HNM (Step 4)
  --two_stage                Stage1=3-class + Stage2=2-class minor/major (Step 5)
  --init_mode                scratch (default) | pretrained (Step 6)
  --pretrained_ckpt_path     encoder checkpoint for pretrained init (Step 6)
  --cv_folds_path            path to cv_folds JSON (make_cv_folds.py output)
  --cv_fold                  fold to use as val set (0..k-1); requires cv_folds_path
  --train_disasters          comma-sep disaster IDs for train (e.g. socal-fire)
  --val_disasters            comma-sep disaster IDs for val (e.g. santa-rosa-wildfire)

Usage:
    python scripts/train_damage.py \\
        --index_csv data/processed/index.csv \\
        --crops_dir data/processed/crops_oracle \\
        --out_dir   models/six_channel \\
        --model_type six_channel \\
        [--epochs 30] [--batch 32] [--lr 3e-4] [--size 128]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import random

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def seed_all(seed: int) -> None:
    """Pin all RNG sources for reproducible runs."""
    import torch as _torch
    random.seed(seed)
    np.random.seed(seed)
    _torch.manual_seed(seed)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(seed)
    _torch.backends.cudnn.benchmark     = False
    _torch.backends.cudnn.deterministic = True


def collate(batch):
    import torch
    xs = torch.from_numpy(np.stack([b[0] for b in batch])).float()
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return xs, ys


def _compute_confusion_matrix(all_preds, all_labels, num_classes=4):
    """cm[true_class][pred_class] = count."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(all_labels, all_preds):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t][p] += 1
    return cm


def _compute_val_metrics(all_preds, all_labels, num_classes=4):
    """Returns (f1s, macro_f1, precs, recs) — one value per class."""
    from collections import defaultdict
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for p, t in zip(all_preds, all_labels):
        if p == t:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    f1s, precs, recs = [], [], []
    for c in range(num_classes):
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        rec  = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
        precs.append(prec)
        recs.append(rec)
    return f1s, float(np.mean(f1s)), precs, recs


def _compute_class_weights(records, num_classes, weight_mode, w_min, w_max):
    """Normalized inverse-frequency weights, optionally capped/floored, or all-ones."""
    if weight_mode == "none":
        return np.ones(num_classes, dtype=np.float32)
    counts = np.zeros(num_classes, dtype=np.float32)
    for r in records:
        counts[r["label_idx"]] += 1
    counts = np.where(counts == 0, 1, counts)
    w = 1.0 / counts
    w /= w.sum()
    w *= num_classes
    if weight_mode == "capped_floored":
        w = np.clip(w, w_min, w_max)
    return w


def _get_train_records(ds):
    """Return the underlying record list from any dataset variant."""
    if hasattr(ds, "records"):          # CropDataset
        return ds.records
    if hasattr(ds, "_base"):            # NineChannelDataset
        return ds._base.records
    if hasattr(ds, "recs"):             # inline CentroidDataset
        return ds.recs
    raise AttributeError(f"Cannot find records on {type(ds)}")


def _base_sample_weights(records, class_weights_np):
    """Per-sample weight array from per-class weights (indexed by label_idx)."""
    return np.array(
        [float(class_weights_np[r["label_idx"]]) for r in records],
        dtype=np.float32,
    )


def _build_sampler(sample_weights_np, num_samples):
    """Build WeightedRandomSampler with replacement (invfreq mode)."""
    import torch
    from torch.utils.data import WeightedRandomSampler
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights_np).float(),
        num_samples=num_samples,
        replacement=True,
    )


def _build_sampler_capped(records, base_w_np, cap: float):
    """
    invfreq_capped: same as invfreq but per-sample weight multiplier is capped at `cap`.
    Reduces extreme repetition of the rarest minority items.
    """
    import torch
    from torch.utils.data import WeightedRandomSampler
    raw_w = np.array([float(base_w_np[r["label_idx"]]) for r in records], dtype=np.float32)
    # Compute per-sample multiplier relative to no-damage weight (class 0)
    w_no = base_w_np[0] if base_w_np[0] > 0 else 1.0
    capped = np.minimum(raw_w / w_no, cap) * w_no
    return WeightedRandomSampler(
        weights=torch.from_numpy(capped).float(),
        num_samples=len(records),
        replacement=True,
    )


class _ClassQuotaBatchSampler:
    """
    Ensures each batch contains at least 1 minor (class 1) and 1 major (class 2)
    sample if available in the training set.  Remaining slots filled by invfreq
    WeightedRandomSampler.  Uses replacement within the quota slots.
    """
    def __init__(self, records: list[dict], batch_size: int, base_w_np: np.ndarray) -> None:
        self.batch_size = batch_size
        # Index by class
        self.by_class: dict[int, list[int]] = {}
        for i, r in enumerate(records):
            c = r["label_idx"]
            self.by_class.setdefault(c, []).append(i)
        # Invfreq weights for filling remaining slots
        import torch
        from torch.utils.data import WeightedRandomSampler
        sw = np.array([float(base_w_np[r["label_idx"]]) for r in records], dtype=np.float32)
        self._sampler = WeightedRandomSampler(
            torch.from_numpy(sw).float(), num_samples=len(records), replacement=True
        )
        self._n_batches = len(records) // batch_size

    def __iter__(self):
        import random as _rnd
        fill_indices = list(self._sampler)
        fill_pos = 0
        for _ in range(self._n_batches):
            batch = []
            # Quota: 1 minor + 1 major (if class exists)
            for cls in (1, 2):
                if self.by_class.get(cls):
                    batch.append(_rnd.choice(self.by_class[cls]))
            # Fill remaining slots
            while len(batch) < self.batch_size:
                if fill_pos >= len(fill_indices):
                    fill_indices = list(self._sampler)
                    fill_pos = 0
                batch.append(fill_indices[fill_pos])
                fill_pos += 1
            _rnd.shuffle(batch)
            yield batch

    def __len__(self):
        return self._n_batches


def _load_encoder_weights(model, ckpt_path: str, device):
    """Load encoder submodule weights from checkpoint. Head stays random."""
    import torch
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_state = ckpt.get("model_state_dict", ckpt)

    if hasattr(model, "encoder"):
        enc_attr = "encoder"
    elif hasattr(model, "stream"):
        enc_attr = "stream"
    else:
        print(f"  [init] WARNING: no 'encoder' or 'stream' found; skipping pretrained init.")
        return

    encoder = getattr(model, enc_attr)
    prefix = enc_attr + "."
    enc_state = {k[len(prefix):]: v for k, v in ckpt_state.items() if k.startswith(prefix)}

    if not enc_state:
        avail = list(ckpt_state.keys())[:5]
        print(f"  [init] WARNING: no keys with prefix '{prefix}'. "
              f"Available (first 5): {avail}")
        return

    missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
    print(f"  [init] Loaded encoder ('{enc_attr}') from {ckpt_path} | keys={len(enc_state)}")
    if missing:
        print(f"  [init] Missing keys  : {missing}")
    if unexpected:
        print(f"  [init] Unexpected keys: {unexpected}")
    print(f"  [init] Head layers remain randomly initialized.")


def _two_stage_predict(x_batch, model1, model2, device):
    """
    Stage 1: 3-class (0=no-dmg, 1=damaged, 2=destroyed).
    Stage 2: 2-class (0=minor, 1=major), applied only where Stage 1 predicts 'damaged'.
    Returns a list of 4-class predictions (0=no-dmg,1=minor,2=major,3=destroyed).
    """
    s1_preds = model1(x_batch).argmax(1).cpu().tolist()
    s2_idxs = [i for i, p in enumerate(s1_preds) if p == 1]
    s2_map = {}
    if s2_idxs:
        import torch
        x_s2 = x_batch[torch.tensor(s2_idxs)]
        s2_out = model2(x_s2).argmax(1).cpu().tolist()  # 0=minor, 1=major
        for j, orig_i in enumerate(s2_idxs):
            s2_map[orig_i] = s2_out[j] + 1  # minor→1, major→2
    final = []
    for i, p1 in enumerate(s1_preds):
        if p1 == 0:
            final.append(0)       # no-damage
        elif p1 == 2:
            final.append(3)       # destroyed
        else:
            final.append(s2_map[i])  # minor or major
    return final


# ---------------------------------------------------------------------------
# Prediction policy and threshold helpers
# ---------------------------------------------------------------------------

def predict_with_policy(
    probs: np.ndarray,
    policy: str,
    thresholds: "dict | None",
) -> np.ndarray:
    """
    Apply prediction policy to a (N, C) probability array.

    policy: "argmax" | "per_class_threshold" | "ordinal_threshold"
    thresholds: dict with optional keys:
      per_class_threshold: "minor" (float), "major" (float)
      ordinal_threshold:   "tau_damage" (float), "tau_severe" (float)

    per_class_threshold rule:
      - If p(minor) >= tau_minor  -> minor_candidate
      - If p(major) >= tau_major  -> major_candidate
      - Both candidates: choose the one with larger (p - tau) margin
      - One candidate: choose it.
      - No candidates: fallback to argmax over all 4 classes.

    ordinal_threshold rule (cumulative probabilities):
      p_damage = p(minor) + p(major) + p(destroyed)  [= 1 - p(no_damage)]
      p_severe = p(major) + p(destroyed)
      - If p_severe >= tau_severe:
            predict argmax over {major(2), destroyed(3)}
      - Elif p_damage >= tau_damage:
            predict argmax over {minor(1), major(2), destroyed(3)}
      - Else:
            predict no-damage (0)

    Returns (N,) int ndarray.
    """
    preds = probs.argmax(axis=1).copy()
    if policy == "argmax" or thresholds is None:
        return preds

    if policy == "ordinal_threshold":
        tau_damage = float(thresholds.get("tau_damage", 0.5))
        tau_severe = float(thresholds.get("tau_severe", 0.5))
        p_damage = probs[:, 1] + probs[:, 2] + probs[:, 3]   # 1 - p(no_damage)
        p_severe = probs[:, 2] + probs[:, 3]
        for i in range(len(probs)):
            if p_severe[i] >= tau_severe:
                # choose between major(2) and destroyed(3)
                preds[i] = 2 if probs[i, 2] >= probs[i, 3] else 3
            elif p_damage[i] >= tau_damage:
                # choose among minor(1), major(2), destroyed(3)
                preds[i] = int(np.argmax(probs[i, 1:]) + 1)
            else:
                preds[i] = 0
        return preds

    # per_class_threshold
    tau_minor = thresholds.get("minor")
    tau_major  = thresholds.get("major")
    if tau_minor is None:
        tau_minor = 0.5
    if tau_major is None:
        tau_major = 0.5

    for i in range(len(probs)):
        minor_cand = bool(probs[i, 1] >= tau_minor)
        major_cand = bool(probs[i, 2] >= tau_major)
        if minor_cand and major_cand:
            # Tie-break: larger (prob - threshold) margin wins
            minor_margin = probs[i, 1] - tau_minor
            major_margin = probs[i, 2] - tau_major
            preds[i] = 1 if minor_margin >= major_margin else 2
        elif minor_cand:
            preds[i] = 1
        elif major_cand:
            preds[i] = 2
        # else: keep argmax fallback
    return preds


def _fit_thresholds(
    all_probs: np.ndarray,
    all_labels: np.ndarray,
    target_recall_minor: float,
    target_recall_major: float,
    never_miss: bool,
) -> "tuple[dict, dict]":
    """
    Deterministic per-class threshold fitting (replaces coarse grid search).

    For each class c in {minor(1), major(2)}:
      S_pos = probs[pos_mask, c]
      If n_pos == 0:
        tau = 0.0 (never_miss=True) or 0.5 (never_miss=False)
      elif target_recall >= 1.0:
        tau = max(0.0, min(S_pos) - eps)   <- guarantees every calib positive
      else:
        tau = quantile(S_pos, 1 - target_recall) - eps   <- largest tau meeting target

    Returns (thresholds: dict, calib_summary: dict).
    """
    eps = 1e-6
    thresholds: dict = {}
    calib_summary: dict = {}

    for cls_idx, name, target_recall in [
        (1, "minor", target_recall_minor),
        (2, "major", target_recall_major),
    ]:
        pos_mask  = (all_labels == cls_idx)
        n_pos     = int(pos_mask.sum())
        probs_cls = all_probs[:, cls_idx]

        if n_pos == 0:
            tau = 0.0 if never_miss else 0.5
            label = "0.0 (never_miss: aggressive flagging)" if never_miss else "0.5 (default)"
            print(f"  [{name}] WARNING: no positives in calib — cannot fit tau; "
                  f"using {label}")
            thresholds[name] = float(tau)
            calib_summary[name] = {
                "n_pos": 0, "tau": float(tau),
                "achieved_recall": None, "fp_calib": None,
                "fp_per_1k_no_calib": None,
            }
            continue

        S_pos = probs_cls[pos_mask]

        if target_recall >= 1.0:
            tau = max(0.0, float(S_pos.min()) - eps)
        else:
            tau = float(np.quantile(S_pos, 1.0 - target_recall)) - eps
            tau = max(0.0, tau)

        achieved_recall = float((probs_cls[pos_mask] >= tau).sum()) / n_pos
        fp           = int((probs_cls[~pos_mask] >= tau).sum())
        n_no         = int((all_labels == 0).sum())
        fp_per_1k_no = 1000.0 * int((probs_cls[all_labels == 0] >= tau).sum()) / max(n_no, 1)

        print(f"  [{name}] target_recall={target_recall:.2f}  "
              f"tau={tau:.6f}  achieved_recall={achieved_recall:.4f}  "
              f"FP_calib={fp}  FP/1k-no={fp_per_1k_no:.1f}")

        thresholds[name]    = float(tau)
        calib_summary[name] = {
            "n_pos":              n_pos,
            "tau":                round(float(tau), 8),
            "achieved_recall":    round(achieved_recall, 4),
            "fp_calib":           fp,
            "fp_per_1k_no_calib": round(fp_per_1k_no, 2),
        }

    return thresholds, calib_summary


def _fit_thresholds_ordinal(
    all_probs: np.ndarray,
    all_labels: np.ndarray,
    target_recall_damage: float,
    target_recall_severe: float,
    never_miss: bool,
) -> "tuple[dict, dict]":
    """
    Fit cumulative (ordinal) thresholds for two detection tasks:
      - damage detection: minor-or-worse (label in {1,2,3})
      - severe detection: major-or-worse (label in {2,3})

    Scores used:
      S_damage = p(minor) + p(major) + p(destroyed)   [= 1 - p(no_damage)]
      S_severe = p(major) + p(destroyed)

    For target_recall >= 1.0:
      tau = max(0.0, min(S_pos) - eps)   <- guarantees every calib positive
    Else:
      tau = quantile(S_pos, 1 - target_recall) - eps

    Returns (thresholds: dict, calib_summary: dict).
    """
    eps = 1e-6
    thresholds: dict = {}
    calib_summary: dict = {}

    p_damage = all_probs[:, 1] + all_probs[:, 2] + all_probs[:, 3]
    p_severe = all_probs[:, 2] + all_probs[:, 3]

    tasks = [
        ("tau_damage", p_damage, np.isin(all_labels, [1, 2, 3]), target_recall_damage,
         "damage (minor-or-worse)"),
        ("tau_severe", p_severe, np.isin(all_labels, [2, 3]),    target_recall_severe,
         "severe (major-or-worse)"),
    ]

    for key, scores, pos_mask, target_recall, desc in tasks:
        n_pos = int(pos_mask.sum())

        if n_pos == 0:
            tau = 0.0 if never_miss else 0.5
            label = "0.0 (never_miss)" if never_miss else "0.5 (default)"
            print(f"  [{key}] WARNING: no positives in calib — cannot fit; using {label}")
            thresholds[key] = float(tau)
            calib_summary[key] = {
                "n_pos": 0, "tau": float(tau),
                "achieved_recall": None, "fp_per_1k_no_calib": None,
            }
            continue

        S_pos = scores[pos_mask]

        if target_recall >= 1.0:
            tau = max(0.0, float(S_pos.min()) - eps)
        else:
            tau = float(np.quantile(S_pos, 1.0 - target_recall)) - eps
            tau = max(0.0, tau)

        achieved_recall = float((scores[pos_mask] >= tau).sum()) / n_pos
        n_no   = int((all_labels == 0).sum())
        fp_per_1k_no = 1000.0 * int((scores[all_labels == 0] >= tau).sum()) / max(n_no, 1)

        print(f"  [{key}] {desc}  target_recall={target_recall:.2f}  "
              f"tau={tau:.6f}  achieved_recall={achieved_recall:.4f}  "
              f"FP/1k-no={fp_per_1k_no:.1f}")

        thresholds[key] = round(float(tau), 8)
        calib_summary[key] = {
            "n_pos":              n_pos,
            "tau":                round(float(tau), 8),
            "achieved_recall":    round(achieved_recall, 4),
            "fp_per_1k_no_calib": round(fp_per_1k_no, 2),
        }

    return thresholds, calib_summary


def _make_eval_metrics(
    all_preds: "list | np.ndarray",
    all_labels: "list | np.ndarray",
    policy: str,
    thresholds: "dict | None",
) -> dict:
    """
    Full evaluation metrics dict — includes explicit FN counts for minor/major.

    Keys: policy, macro_f1, per_class, n_minor_test, FN_minor_test,
          recall_minor_test, n_major_test, FN_major_test, recall_major_test,
          FP_per_1000_no_minor, FP_per_1000_no_major, confusion_matrix,
          thresholds_used.
    """
    _CLS = ["no-damage", "minor-damage", "major-damage", "destroyed"]
    preds  = np.asarray(all_preds)
    labels = np.asarray(all_labels)

    f1s, macro_f1, precs, recs = _compute_val_metrics(
        preds.tolist(), labels.tolist(), num_classes=4
    )
    cm = _compute_confusion_matrix(preds.tolist(), labels.tolist(), num_classes=4)

    NO, MINOR, MAJOR = 0, 1, 2
    n_no    = int((labels == NO).sum())
    n_minor = int((labels == MINOR).sum())
    n_major = int((labels == MAJOR).sum())
    FN_minor = int(((labels == MINOR) & (preds != MINOR)).sum())
    FN_major = int(((labels == MAJOR) & (preds != MAJOR)).sum())
    recall_minor = round((n_minor - FN_minor) / n_minor, 4) if n_minor > 0 else None
    recall_major = round((n_major - FN_major) / n_major, 4) if n_major > 0 else None
    fp_minor_from_no = int(cm[NO][MINOR])
    fp_major_from_no = int(cm[NO][MAJOR])

    return {
        "policy":               policy,
        "macro_f1":             round(macro_f1, 4),
        "per_class":            {
            _CLS[i]: {
                "f1":   round(f1s[i], 4),
                "prec": round(precs[i], 4),
                "rec":  round(recs[i], 4),
            } for i in range(4)
        },
        "n_minor_test":         n_minor,
        "FN_minor_test":        FN_minor,
        "recall_minor_test":    recall_minor,
        "n_major_test":         n_major,
        "FN_major_test":        FN_major,
        "recall_major_test":    recall_major,
        "FP_per_1000_no_minor": round(1000.0 * fp_minor_from_no / max(n_no, 1), 2),
        "FP_per_1000_no_major": round(1000.0 * fp_major_from_no / max(n_no, 1), 2),
        "confusion_matrix":     cm.tolist(),
        "thresholds_used":      thresholds,
    }


def _make_eval_metrics_ordinal(
    all_preds: "list | np.ndarray",
    all_labels: "list | np.ndarray",
    all_probs: "np.ndarray | None",
    policy: str,
    thresholds: "dict | None",
) -> dict:
    """
    Evaluation metrics including explicit damage/severe detection tasks.

    damage detection: minor-or-worse (label in {1,2,3}) as positive
    severe detection:  major-or-worse (label in {2,3}) as positive

    FP_damage_per_1000_no: FP from no-damage buildings classified as any damage
    FP_severe_per_1000_nonsevere: FP from non-severe classified as severe
    """
    base = _make_eval_metrics(all_preds, all_labels, policy, thresholds)

    preds  = np.asarray(all_preds)
    labels = np.asarray(all_labels)

    # Damage detection task (minor-or-worse)
    pos_damage  = np.isin(labels, [1, 2, 3])
    pred_damage = np.isin(preds,  [1, 2, 3])
    n_pos_damage  = int(pos_damage.sum())
    n_no          = int((labels == 0).sum())
    FN_damage     = int((pos_damage & ~pred_damage).sum())
    FP_damage     = int((~pos_damage & pred_damage).sum())
    recall_damage = round((n_pos_damage - FN_damage) / n_pos_damage, 4) if n_pos_damage > 0 else None
    fp_damage_per_1k_no = round(1000.0 * int(((labels == 0) & pred_damage).sum()) / max(n_no, 1), 2)

    # Severe detection task (major-or-worse)
    pos_severe      = np.isin(labels, [2, 3])
    pred_severe     = np.isin(preds,  [2, 3])
    n_pos_severe    = int(pos_severe.sum())
    n_nonsevere     = int((~pos_severe).sum())
    FN_severe       = int((pos_severe & ~pred_severe).sum())
    FP_severe       = int((~pos_severe & pred_severe).sum())
    recall_severe   = round((n_pos_severe - FN_severe) / n_pos_severe, 4) if n_pos_severe > 0 else None
    fp_severe_per_1k_nonsevere = round(1000.0 * FP_severe / max(n_nonsevere, 1), 2)

    base.update({
        "n_pos_damage":              n_pos_damage,
        "FN_damage":                 FN_damage,
        "recall_damage":             recall_damage,
        "FP_damage_per_1000_no":     fp_damage_per_1k_no,
        "n_pos_severe":              n_pos_severe,
        "FN_severe":                 FN_severe,
        "recall_severe":             recall_severe,
        "FP_severe_per_1000_nonsevere": fp_severe_per_1k_nonsevere,
    })
    return base


def _predict_cascade(
    p_damage: np.ndarray,
    p_severe: np.ndarray,
    severity_logits: np.ndarray,
    tau_damage: float = 0.5,
    tau_severe: float = 0.5,
) -> np.ndarray:
    """
    Cascade prediction rule for MultiHeadCNN outputs.

    severity head indices: 0=minor, 1=major, 2=destroyed

    - if p_severe >= tau_severe:
        argmax over severity indices {1,2} → map to 4-class {major(2), destroyed(3)}
    - elif p_damage >= tau_damage:
        argmax over severity (0..2) → map to 4-class {minor(1), major(2), destroyed(3)}
    - else: no-damage (0)
    """
    N = len(p_damage)
    preds = np.zeros(N, dtype=np.int64)
    for i in range(N):
        if p_severe[i] >= tau_severe:
            # restricted argmax over severity indices 1 and 2 (major, destroyed)
            idx = int(np.argmax(severity_logits[i, 1:])) + 1  # 1 or 2 in severity space
            preds[i] = idx + 1  # severity-1→4-class-2, severity-2→4-class-3
        elif p_damage[i] >= tau_damage:
            idx = int(np.argmax(severity_logits[i]))  # 0,1,2 in severity space
            preds[i] = idx + 1  # severity-0→1, 1→2, 2→3
        else:
            preds[i] = 0
    return preds


def _fit_thresholds_cascade(
    p_damage: np.ndarray,
    p_severe: np.ndarray,
    all_labels: np.ndarray,
    target_recall_damage: float = 1.0,
    target_recall_severe: float = 1.0,
    never_miss: bool = True,
) -> "tuple[dict, dict]":
    """
    Fit binary thresholds tau_damage and tau_severe from cascade head sigmoid outputs.

    p_damage: sigmoid(damage_logit) on calibration set — (N,)
    p_severe: sigmoid(severe_logit) on calibration set — (N,)
    all_labels: 4-class ground truth {0,1,2,3} — (N,)

    Positive sets:
      damage positives: labels in {1,2,3}
      severe  positives: labels in {2,3}
    """
    eps = 1e-6
    thresholds: dict = {}
    calib_summary: dict = {}

    tasks = [
        ("tau_damage", p_damage, np.isin(all_labels, [1, 2, 3]),
         target_recall_damage, "damage (minor-or-worse)"),
        ("tau_severe", p_severe, np.isin(all_labels, [2, 3]),
         target_recall_severe, "severe (major-or-worse)"),
    ]

    for key, scores, pos_mask, target_recall, desc in tasks:
        n_pos = int(pos_mask.sum())

        if n_pos == 0:
            tau = 0.0 if never_miss else 0.5
            label = "0.0 (never_miss)" if never_miss else "0.5 (default)"
            print(f"  [cascade/{key}] WARNING: no positives in calib — using {label}")
            thresholds[key] = float(tau)
            calib_summary[key] = {
                "n_pos": 0, "tau": float(tau),
                "achieved_recall": None, "fp_per_1k_no_calib": None,
            }
            continue

        S_pos = scores[pos_mask]

        if target_recall >= 1.0:
            tau = max(0.0, float(S_pos.min()) - eps)
        else:
            tau = float(np.quantile(S_pos, 1.0 - target_recall)) - eps
            tau = max(0.0, tau)

        achieved_recall = float((scores[pos_mask] >= tau).sum()) / n_pos
        n_no   = int((all_labels == 0).sum())
        fp_per_1k_no = 1000.0 * int((scores[all_labels == 0] >= tau).sum()) / max(n_no, 1)

        print(f"  [cascade/{key}] {desc}  target_recall={target_recall:.2f}  "
              f"tau={tau:.6f}  achieved_recall={achieved_recall:.4f}  "
              f"FP/1k-no={fp_per_1k_no:.1f}")

        thresholds[key] = round(float(tau), 8)
        calib_summary[key] = {
            "n_pos":              n_pos,
            "tau":                round(float(tau), 8),
            "achieved_recall":    round(achieved_recall, 4),
            "fp_per_1k_no_calib": round(fp_per_1k_no, 2),
        }

    return thresholds, calib_summary


def _fit_temperature_scalar(logits: np.ndarray, y_binary: np.ndarray) -> float:
    """
    Fit a temperature scalar T minimizing NLL for binary sigmoid calibration.
    Uses scipy.optimize.minimize_scalar (bounded search over [0.01, 10]).
    Falls back to T=1.0 if scipy is unavailable.
    """
    try:
        from scipy.optimize import minimize_scalar as _ms
    except ImportError:
        return 1.0
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y_binary, dtype=np.float64)
    eps = 1e-7

    def nll(T: float) -> float:
        T = max(T, 1e-8)
        s = 1.0 / (1.0 + np.exp(-logits / T))
        return -float(np.mean(y * np.log(s + eps) + (1 - y) * np.log(1 - s + eps)))

    res = _ms(nll, bounds=(0.01, 10.0), method="bounded")
    return float(res.x)


def _make_eval_metrics_cascade(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
    thresholds: "dict | None",
    temp_scale: bool = False,
) -> dict:
    """
    Cascade evaluation metrics (wraps _make_eval_metrics_ordinal).
    Adds tau_damage/tau_severe and temp_scale_stage1 fields.
    """
    base = _make_eval_metrics_ordinal(
        all_preds, all_labels, None, "cascade_threshold", thresholds
    )
    base["tau_damage_used"]   = thresholds.get("tau_damage") if thresholds else None
    base["tau_severe_used"]   = thresholds.get("tau_severe") if thresholds else None
    base["temp_scale_stage1"] = temp_scale
    return base


def _tile_grouped_calib_split(
    records: list,
    calib_fraction: float,
    seed: int,
) -> "tuple[list, list]":
    """
    Tile-grouped split of records into (train_fit, calib).

    Tries to include some minority (minor/major) tiles in calib for threshold
    fitting.  Logs a warning if none are available.

    Returns (train_fit_recs, calib_recs).  No tile appears in both sets.
    """
    import random as _rnd_inner
    rng = _rnd_inner.Random(seed)

    tile_ids      = sorted({r["tile_id"] for r in records})
    minority_set  = {r["tile_id"] for r in records if r["label_idx"] in (1, 2)}
    minority_list = sorted(minority_set)
    majority_list = [t for t in tile_ids if t not in minority_set]

    n_calib_total = max(1, int(len(tile_ids) * calib_fraction))
    n_min_calib   = max(0, int(len(minority_list) * calib_fraction))
    n_maj_calib   = max(0, n_calib_total - n_min_calib)

    rng.shuffle(minority_list)
    rng.shuffle(majority_list)

    calib_tiles = set(minority_list[:n_min_calib] + majority_list[:n_maj_calib])

    if not (calib_tiles & minority_set):
        print("  [NestedCV] WARNING: calib set has no minor/major tiles — "
              "threshold fitting will use fallback values.")

    train_fit = [r for r in records if r["tile_id"] not in calib_tiles]
    calib     = [r for r in records if r["tile_id"] in calib_tiles]
    return train_fit, calib


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        print("torch not installed. Run: pip install torch")
        sys.exit(1)

    from disaster_bench.data.dataset import (
        DAMAGE_CLASSES, build_crop_records, train_val_split, CropDataset,
        NineChannelDataset,
    )
    from disaster_bench.models.damage.classifiers import (
        build_classifier, build_multihead_classifier, save_checkpoint,
    )

    # ---------------------------------------------------------------------------
    # Never-miss mode: force target_recall=1.0 and threshold_policy
    # ---------------------------------------------------------------------------
    if getattr(args, "never_miss_mode", 0):
        args.target_recall_minor = 1.0
        args.target_recall_major = 1.0
        # Use cascade_threshold when cascade_mode is on; else ordinal_threshold
        if getattr(args, "threshold_policy", "argmax") in ("argmax", "per_class_threshold"):
            if getattr(args, "cascade_mode", "off") == "multihead":
                args.threshold_policy = "cascade_threshold"
            else:
                args.threshold_policy = "ordinal_threshold"
        print("[NeverMiss] FN=0 mode active: "
              "target_recall=1.0, "
              f"threshold_policy={args.threshold_policy}")

    # Validate pretrained args
    if args.init_mode == "pretrained" and not args.pretrained_ckpt_path:
        print("Error: --pretrained_ckpt_path is required when --init_mode pretrained",
              file=sys.stderr)
        sys.exit(1)

    # Seed for reproducibility (sampler, weight init, augmentation)
    seed_all(args.seed)
    print(f"Seed: {args.seed}")

    # Step 0: run output directory
    run_dir = Path("runs") / args.run_id if args.run_id else None
    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model type: {args.model_type}")

    # -----------------------------------------------------------------------
    # Records
    # -----------------------------------------------------------------------
    if args.model_type == "centroid_patch":
        from disaster_bench.data.dataset import build_centroid_records
        print("Building centroid records from index...")
        records = build_centroid_records(args.index_csv)
    else:
        print("Building crop records...")
        records = build_crop_records(args.index_csv, args.crops_dir)

    label_dist = Counter(r["label"] for r in records)
    print(f"  Total buildings: {len(records)}")
    for cls in DAMAGE_CLASSES:
        print(f"  {cls:20s}: {label_dist.get(cls, 0)}")

    # -----------------------------------------------------------------------
    # Train / val split: disaster split > CV fold > default 80/20
    # -----------------------------------------------------------------------
    def _disaster_id(tile_id: str) -> str:
        return tile_id.rsplit("_", 1)[0]

    if args.train_disasters and args.val_disasters:
        _train_dis = set(args.train_disasters.split(","))
        _val_dis   = set(args.val_disasters.split(","))
        train_recs = [r for r in records if _disaster_id(r["tile_id"]) in _train_dis]
        val_recs   = [r for r in records if _disaster_id(r["tile_id"]) in _val_dis]
        _tr_tiles  = len({r["tile_id"] for r in train_recs})
        _va_tiles  = len({r["tile_id"] for r in val_recs})
        print(f"\nDisaster split — train: {sorted(_train_dis)}  val: {sorted(_val_dis)}")
        print(f"  Train: {len(train_recs)} buildings across {_tr_tiles} tiles")
        print(f"  Val:   {len(val_recs)} buildings across {_va_tiles} tiles")
        tr_dist = Counter(r["label"] for r in train_recs)
        va_dist = Counter(r["label"] for r in val_recs)
        print("  Train dist:", {cls: tr_dist.get(cls, 0) for cls in DAMAGE_CLASSES})
        print("  Val   dist:", {cls: va_dist.get(cls, 0) for cls in DAMAGE_CLASSES})
    elif args.cv_folds_path and args.cv_fold is not None:
        with open(args.cv_folds_path) as _f:
            _fold_data = json.load(_f)
        _tile_to_fold = _fold_data["tile_to_fold"]
        _val_fold     = args.cv_fold
        _val_tiles    = {t for t, fi in _tile_to_fold.items() if fi == _val_fold}
        train_recs = [r for r in records if r["tile_id"] not in _val_tiles]
        val_recs   = [r for r in records if r["tile_id"] in _val_tiles]
        print(f"\nCV fold {_val_fold}/{_fold_data['k']}: "
              f"train={len(train_recs)}  val={len(val_recs)}")
        tr_dist = Counter(r["label"] for r in train_recs)
        va_dist = Counter(r["label"] for r in val_recs)
        print("  Train:", {cls: tr_dist.get(cls, 0) for cls in DAMAGE_CLASSES})
        print("  Val:  ", {cls: va_dist.get(cls, 0) for cls in DAMAGE_CLASSES})
    else:
        train_recs, val_recs = train_val_split(records, val_fraction=args.val_fraction)
        print(f"\nTrain: {len(train_recs)}  Val: {len(val_recs)}")

    # -----------------------------------------------------------------------
    # Nested CV: further split outer-train tiles into train_fit + calib
    # outer_test_recs stays completely hidden until post-training evaluation.
    # -----------------------------------------------------------------------
    outer_test_recs = None
    if getattr(args, "nested_cv", 0) and args.cv_folds_path and args.cv_fold is not None:
        outer_test_recs = val_recs          # hold out; never used during training
        _calib_seed = getattr(args, "calib_seed", None) or args.seed
        train_recs, val_recs = _tile_grouped_calib_split(
            train_recs,
            calib_fraction=getattr(args, "calib_fraction", 0.2),
            seed=_calib_seed,
        )
        from collections import Counter as _Ctr
        _ncv_dist = _Ctr(r["label"] for r in val_recs)
        print(f"\n[NestedCV fold={args.cv_fold}] "
              f"train_fit={len(train_recs)}  calib={len(val_recs)}  "
              f"outer_test={len(outer_test_recs)}")
        print(f"  Calib dist: { {cls: _ncv_dist.get(cls, 0) for cls in DAMAGE_CLASSES} }")

    # -----------------------------------------------------------------------
    # Step 5: two-stage label remapping
    # -----------------------------------------------------------------------
    S1_REMAP  = {0: 0, 1: 1, 2: 1, 3: 2}   # no-dmg=0, damaged(minor+major)=1, dest=2
    S2_REMAP  = {1: 0, 2: 1}                # minor=0, major=1
    S2_FILTER = {1, 2}

    if args.two_stage:
        s1_train_recs = [dict(r, label_idx=S1_REMAP[r["label_idx"]]) for r in train_recs]
        s2_train_recs = [dict(r, label_idx=S2_REMAP[r["label_idx"]])
                         for r in train_recs if r["label_idx"] in S2_FILTER]
        print(f"  Stage1 train: {len(s1_train_recs)} (3-class)  "
              f"Stage2 train: {len(s2_train_recs)} (2-class minor/major only)")

    # -----------------------------------------------------------------------
    # Datasets
    # -----------------------------------------------------------------------
    if args.model_type == "centroid_patch":
        from disaster_bench.data.dataset import load_centroid_patch, LABEL2IDX
        import random

        class CentroidDataset:
            def __init__(self, recs, size=128, augment=False):
                self.recs    = recs
                self.size    = size
                self.augment = augment

            def __len__(self):
                return len(self.recs)

            def __getitem__(self, idx):
                from PIL import Image
                r = self.recs[idx]
                pre_img  = np.array(Image.open(r["pre_path"]).convert("RGB"))
                post_img = np.array(Image.open(r["post_path"]).convert("RGB"))
                x = load_centroid_patch(pre_img, post_img, r["cx"], r["cy"], self.size)
                if self.augment:
                    if random.random() > 0.5:
                        x = x[:, :, ::-1].copy()
                    if random.random() > 0.5:
                        x = x[:, ::-1, :].copy()
                return x, r["label_idx"]

        if args.two_stage:
            train_ds    = CentroidDataset(s1_train_recs, size=args.size, augment=True)
            s2_train_ds = CentroidDataset(s2_train_recs, size=args.size, augment=True)
        else:
            train_ds    = CentroidDataset(train_recs, size=args.size, augment=True)
            s2_train_ds = None
        val_ds      = CentroidDataset(val_recs, size=args.size, augment=False)
        train_ds_base = None

    else:
        aug_config = {
            "rotate90":     args.aug_rotate90,
            "affine":       args.aug_affine,
            "color_jitter": args.aug_color_jitter,
            "noise":        args.aug_noise,
        }
        if args.two_stage:
            train_ds_base    = CropDataset(s1_train_recs, size=args.size, augment=True,
                                           preload=True, aug_config=aug_config)
            s2_train_ds_base = CropDataset(s2_train_recs, size=args.size, augment=True,
                                           preload=True, aug_config=aug_config)
        else:
            train_ds_base    = CropDataset(train_recs, size=args.size, augment=True,
                                           preload=True, aug_config=aug_config)
            s2_train_ds_base = None
        val_ds_base = CropDataset(val_recs, size=args.size, augment=False, preload=True)

        if args.model_type == "pre_post_diff":
            train_ds    = NineChannelDataset(train_ds_base)
            s2_train_ds = NineChannelDataset(s2_train_ds_base) if s2_train_ds_base else None
            val_ds      = NineChannelDataset(val_ds_base)
        else:
            train_ds    = train_ds_base
            s2_train_ds = s2_train_ds_base
            val_ds      = val_ds_base

    # Outer test dataset (nested CV only — never used during the training loop)
    outer_test_ds     = None
    outer_test_loader = None
    if outer_test_recs is not None:
        if args.model_type == "centroid_patch":
            outer_test_ds = CentroidDataset(outer_test_recs, size=args.size, augment=False)
        else:
            _ot_base = CropDataset(outer_test_recs, size=args.size, augment=False, preload=True)
            outer_test_ds = (NineChannelDataset(_ot_base)
                             if args.model_type == "pre_post_diff" else _ot_base)
        outer_test_loader = DataLoader(outer_test_ds, batch_size=args.batch, shuffle=False,
                                       collate_fn=collate, num_workers=0)
        print(f"[NestedCV] outer_test preloaded: {len(outer_test_recs)} buildings")

    # -----------------------------------------------------------------------
    # Step 3: loss weights
    # -----------------------------------------------------------------------
    if args.two_stage:
        # Stage 1: 3-class weights from s1_train_recs
        s1_w_np = _compute_class_weights(s1_train_recs, 3,
                                          args.weight_mode, args.w_min, args.w_max)
        weights = torch.from_numpy(s1_w_np).float()
        print(f"Stage1 weights (3-class, {args.weight_mode}): "
              f"no-dmg={weights[0]:.4f}  damaged={weights[1]:.4f}  dest={weights[2]:.4f}")
        # Stage 2: 2-class weights from s2_train_recs
        s2_w_np = _compute_class_weights(s2_train_recs, 2,
                                          args.weight_mode, args.w_min, args.w_max)
        s2_weights = torch.from_numpy(s2_w_np).float()
        print(f"Stage2 weights (2-class, {args.weight_mode}): "
              f"minor={s2_weights[0]:.4f}  major={s2_weights[1]:.4f}")
        # Base (unclipped) s1 weights for sampler
        s1_base_w_np = _compute_class_weights(s1_train_recs, 3,
                                               "normalized_invfreq", 0.0, 1e9)
    else:
        # 4-class weights (existing behavior when --loss ce)
        raw_w_np = _compute_class_weights(train_recs, 4,
                                           args.weight_mode, args.w_min, args.w_max)
        weights  = torch.from_numpy(raw_w_np).float()
        print(f"Class weights ({args.weight_mode}): "
              f"{ {c: round(float(w), 4) for c, w in zip(DAMAGE_CLASSES, weights)} }")
        # Base (unclipped) weights for sampler
        base_w_np = _compute_class_weights(train_recs, 4, "normalized_invfreq", 0.0, 1e9)

    # -----------------------------------------------------------------------
    # New long-tail losses (cb_ce / focal / ldam_drw)
    # Computed here after train_recs is established (not inside two_stage branch)
    # -----------------------------------------------------------------------
    _loss_counts_np = None   # will be set below if needed
    _loss_info      = {}
    if not args.two_stage and args.loss != "ce":
        from disaster_bench.training.losses import build_criterion as _build_criterion
        _new_criterion, _loss_info = _build_criterion(
            loss_type        = args.loss,
            train_records    = train_recs,
            beta             = args.beta,
            gamma            = args.gamma,
            max_m            = args.max_m,
            s                = args.ldam_s,
            drw_start_epoch  = args.drw_start_epoch,
        )
        from disaster_bench.training.losses import compute_class_counts as _cc
        _loss_counts_np = _cc(train_recs)
        print(f"Loss: {args.loss}  info={_loss_info}")

    # -----------------------------------------------------------------------
    # Logit adjustment prior (computed from raw train labels, not sampler dist)
    # -----------------------------------------------------------------------
    _log_prior = None
    if args.logit_adjust != "none" and not args.two_stage:
        from disaster_bench.training.losses import make_log_prior as _mlp
        _log_prior = _mlp(train_recs, num_classes=4)
        print(f"Logit adjustment: tau={args.logit_adjust_tau}  "
              f"prior={[round(float(p), 4) for p in _log_prior.exp().tolist()]}")

    # -----------------------------------------------------------------------
    # Step 1: DataLoaders (optional WeightedRandomSampler)
    # -----------------------------------------------------------------------
    sample_weights_np = None   # kept in scope for HNM (Step 4)
    train_records_list = _get_train_records(train_ds)

    # Auto-enable sampler if HNM requested without sampler
    need_sampler = bool(args.use_sampler) or bool(args.use_hard_negative_mining)
    if args.use_hard_negative_mining and not args.use_sampler:
        print("[HNM] use_sampler=0 but use_hard_negative_mining=1; "
              "switching to sampler automatically.")

    _sampler_mode = args.sampler_mode if need_sampler else None
    if need_sampler:
        base_w = s1_base_w_np if args.two_stage else base_w_np
        sample_weights_np = _base_sample_weights(train_records_list, base_w)

        if _sampler_mode == "invfreq_capped":
            sampler = _build_sampler_capped(train_records_list, base_w, args.sampler_cap)
            train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                                      collate_fn=collate, num_workers=0)
            print(f"Sampler: invfreq_capped (cap={args.sampler_cap})")
        elif _sampler_mode == "class_quota_batch":
            batch_sampler = _ClassQuotaBatchSampler(train_records_list, args.batch, base_w)
            train_loader = DataLoader(train_ds, batch_sampler=batch_sampler,
                                      collate_fn=collate, num_workers=0)
            print("Sampler: class_quota_batch (>=1 minor, >=1 major per batch)")
        else:
            # invfreq (default)
            sampler = _build_sampler(sample_weights_np, len(train_records_list))
            train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                                      collate_fn=collate, num_workers=0)
            print("Sampler: invfreq (WeightedRandomSampler)")
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                  collate_fn=collate, num_workers=0)

    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            collate_fn=collate, num_workers=0)

    # Stage 2 DataLoader
    if args.two_stage and s2_train_ds is not None:
        s2_records_list = _get_train_records(s2_train_ds)
        if args.use_sampler:
            s2_base_w = _compute_class_weights(s2_train_recs, 2, "normalized_invfreq", 0.0, 1e9)
            s2_sample_w = _base_sample_weights(s2_records_list, s2_base_w)
            s2_sampler = _build_sampler(s2_sample_w, len(s2_records_list))
            s2_train_loader = DataLoader(s2_train_ds, batch_size=args.batch, sampler=s2_sampler,
                                         collate_fn=collate, num_workers=0)
        else:
            s2_train_loader = DataLoader(s2_train_ds, batch_size=args.batch, shuffle=True,
                                         collate_fn=collate, num_workers=0)
    else:
        s2_train_loader = None

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    _cascade_mode = bool(getattr(args, "cascade_mode", "off") == "multihead")

    if _cascade_mode:
        model = build_multihead_classifier(args.model_type, dropout=args.dropout).to(device)
        model2 = None
        num_classes_s1 = None   # not applicable (multi-head outputs)
        num_classes_s2 = None
        print(f"Model: MultiHeadCNN (backbone={args.model_type})")
    elif args.two_stage:
        model  = build_classifier(args.model_type, num_classes=3, dropout=args.dropout).to(device)
        model2 = build_classifier(args.model_type, num_classes=2, dropout=args.dropout).to(device)
        num_classes_s1 = 3
        num_classes_s2 = 2
    else:
        model  = build_classifier(args.model_type, num_classes=4, dropout=args.dropout).to(device)
        model2 = None
        num_classes_s1 = 4
        num_classes_s2 = None

    print(f"Parameters (Stage1): {sum(p.numel() for p in model.parameters()):,}")
    if model2 is not None:
        print(f"Parameters (Stage2): {sum(p.numel() for p in model2.parameters()):,}")

    # Step 6: pretrained encoder init
    if args.init_mode == "pretrained":
        _load_encoder_weights(model, args.pretrained_ckpt_path, device)
        if model2 is not None:
            _load_encoder_weights(model2, args.pretrained_ckpt_path, device)

    # -----------------------------------------------------------------------
    # Loss / optimizer / scheduler
    # -----------------------------------------------------------------------
    if not args.two_stage and args.loss != "ce":
        # New long-tail loss (cb_ce / focal / ldam_drw)
        criterion = _new_criterion
        # Move any buffers to device
        criterion = criterion.to(device) if hasattr(criterion, 'to') else criterion
        print(f"Criterion: {args.loss}")
    elif args.weight_mode == "none":
        criterion = torch.nn.CrossEntropyLoss()
        print("Criterion: unweighted CrossEntropyLoss (weight_mode=none)")
    else:
        weights = weights.to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.two_stage and model2 is not None:
        if args.weight_mode == "none":
            criterion2 = torch.nn.CrossEntropyLoss()
        else:
            s2_weights = s2_weights.to(device)
            criterion2 = torch.nn.CrossEntropyLoss(weight=s2_weights)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=args.epochs)

    # Cascade mode: binary + severity criteria (used when _cascade_mode=True)
    _criterion_damage = _criterion_severe = _criterion_severity = None
    _cascade_focal = False
    if _cascade_mode:
        from collections import Counter as _CtrC
        _lbl_c   = _CtrC(r["label_idx"] for r in train_recs)
        _n_no_c  = max(_lbl_c.get(0, 0), 1)
        _n_dmg_c = max(_lbl_c.get(1, 0) + _lbl_c.get(2, 0) + _lbl_c.get(3, 0), 1)
        _n_sev_c = max(_lbl_c.get(2, 0) + _lbl_c.get(3, 0), 1)
        _n_non_sev_c = max(len(train_recs) - _n_sev_c, 1)
        _stage1_loss = getattr(args, "stage1_loss", "bce")
        _stage2_loss = getattr(args, "stage2_loss", "ce")
        if _stage1_loss == "bce":
            _criterion_damage = torch.nn.BCEWithLogitsLoss()
            _criterion_severe = torch.nn.BCEWithLogitsLoss()
        elif _stage1_loss == "cb_bce":
            _pw_dmg = torch.tensor([_n_no_c / _n_dmg_c], device=device)
            _pw_sev = torch.tensor([_n_non_sev_c / _n_sev_c], device=device)
            _criterion_damage = torch.nn.BCEWithLogitsLoss(pos_weight=_pw_dmg)
            _criterion_severe = torch.nn.BCEWithLogitsLoss(pos_weight=_pw_sev)
        else:  # focal — computed inline in training loop
            _cascade_focal = True
        if _stage2_loss == "cb_ce":
            _sev_w = np.array([max(_lbl_c.get(1, 0), 1), max(_lbl_c.get(2, 0), 1),
                               max(_lbl_c.get(3, 0), 1)], dtype=np.float32)
            _sev_w = (1.0 / _sev_w) * 3.0 / (1.0 / _sev_w).sum()
            _criterion_severity = torch.nn.CrossEntropyLoss(
                weight=torch.tensor(_sev_w, device=device))
        else:
            _criterion_severity = torch.nn.CrossEntropyLoss()
        print(f"[Cascade] stage1_loss={_stage1_loss}  stage2_loss={_stage2_loss}")

    # -----------------------------------------------------------------------
    # Checkpoint paths
    # -----------------------------------------------------------------------
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path  = out_dir / "best.pt"
    last_path  = out_dir / "last.pt"
    best2_path = out_dir / "best_s2.pt"
    last2_path = out_dir / "last_s2.pt"

    start_epoch = 1
    best_val_f1 = -1.0

    if best_path.exists() and args.resume:
        print(f"Resuming from {best_path} ...")
        ckpt_r = torch.load(str(best_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt_r["model_state_dict"])
        best_val_f1 = ckpt_r.get("val_macro_f1", -1.0)
        start_epoch = ckpt_r.get("epoch", 0) + 1
        print(f"  Resuming at epoch {start_epoch}, best_f1={best_val_f1:.4f}")
        if args.two_stage and model2 is not None and best2_path.exists():
            ckpt_r2 = torch.load(str(best2_path), map_location=device, weights_only=False)
            model2.load_state_dict(ckpt_r2["model_state_dict"])
            print(f"  Loaded Stage2 from {best2_path}")

    f1s, macro_f1 = [0.0] * 4, 0.0

    # -----------------------------------------------------------------------
    # eval_only: skip training, load existing best checkpoint, go straight
    # to threshold fitting + outer-test evaluation (crossfit_pool pooling).
    # -----------------------------------------------------------------------
    _eval_only = bool(getattr(args, "eval_only", False))
    if _eval_only:
        if not best_path.exists():
            print(f"[eval_only] ERROR: no checkpoint at {best_path}. "
                  "Run training first.", file=sys.stderr)
            sys.exit(1)
        print(f"[eval_only] Skipping training — loading {best_path}")
        ckpt_r = torch.load(str(best_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt_r["model_state_dict"])
        best_val_f1 = ckpt_r.get("val_macro_f1", 0.0)
        print(f"[eval_only] Checkpoint loaded (best_val_f1={best_val_f1:.4f})")

    # -----------------------------------------------------------------------
    # Training loop  (skipped entirely when --eval_only 1)
    # -----------------------------------------------------------------------
    _train_start = 0 if _eval_only else start_epoch
    _train_end   = 0 if _eval_only else args.epochs + 1
    for epoch in range(_train_start, _train_end):

        # --- DRW: switch to class-balanced weights at drw_start_epoch ---
        if (not args.two_stage and args.loss == "ldam_drw"
                and epoch == args.drw_start_epoch
                and _loss_counts_np is not None):
            from disaster_bench.training.losses import apply_drw_weights as _adw
            _adw(criterion, _loss_counts_np, beta=args.beta, device=device)
            print(f"  [DRW] Switched to class-balanced weights at epoch {epoch}")

        # --- Stage 1 train ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        t0 = time.perf_counter()

        # Step 2B: batch histogram logging (epoch 1 only, first 50 batches)
        _do_log = (args.log_batch_class_counts and epoch == 1 and run_dir is not None)
        _batch_log = run_dir / "train_batch_label_hist_epoch1.jsonl" if run_dir else None

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            if _cascade_mode:
                _d_logit, _s_logit, _sev_logits = model(x)
                _y_dmg = (y > 0).float()
                _y_sev = (y >= 2).float()
                if _cascade_focal:
                    import torch.nn.functional as _F
                    _gamma_f = float(getattr(args, "gamma", 2.0))
                    def _bfocal(logit, tgt, g):
                        _bce = _F.binary_cross_entropy_with_logits(logit, tgt, reduction="none")
                        _pt  = torch.sigmoid(logit) * tgt + (1 - torch.sigmoid(logit)) * (1 - tgt)
                        return ((1 - _pt) ** g * _bce).mean()
                    _L_dmg = _bfocal(_d_logit, _y_dmg, _gamma_f)
                    _L_sev = _bfocal(_s_logit, _y_sev, _gamma_f)
                else:
                    _L_dmg = _criterion_damage(_d_logit, _y_dmg)
                    _L_sev = _criterion_severe(_s_logit, _y_sev)
                _dmg_mask = y > 0
                if _dmg_mask.any():
                    _L_severity = _criterion_severity(_sev_logits[_dmg_mask], y[_dmg_mask] - 1)
                else:
                    _L_severity = _d_logit.sum() * 0.0
                loss = _L_dmg + _L_sev + _L_severity
                with torch.no_grad():
                    _pd_np = torch.sigmoid(_d_logit).cpu().numpy()
                    _ps_np = torch.sigmoid(_s_logit).cpu().numpy()
                    _sl_np = _sev_logits.detach().cpu().numpy()
                    _cp = _predict_cascade(_pd_np, _ps_np, _sl_np, 0.5, 0.5)
                    train_correct += int((np.array(_cp) == y.cpu().numpy()).sum())
            else:
                logits = model(x)
                loss   = criterion(logits, y)
                train_correct += (logits.argmax(1) == y).sum().item()
            loss.backward()
            optimizer.step()
            train_loss  += loss.item() * len(y)
            train_total += len(y)

            if _do_log and batch_idx < 50:
                counts = [int((y == c).sum().item()) for c in range(4)]
                with open(_batch_log, "a", encoding="utf-8") as bf:
                    bf.write(json.dumps({"batch_idx": batch_idx, "counts": counts}) + "\n")

        # --- Stage 2 train (two_stage only) ---
        s2_loss_avg = None
        if args.two_stage and model2 is not None and s2_train_loader is not None:
            model2.train()
            s2_loss_sum, s2_total = 0.0, 0
            for x2, y2 in s2_train_loader:
                x2, y2 = x2.to(device), y2.to(device)
                optimizer2.zero_grad()
                logits2 = model2(x2)
                loss2   = criterion2(logits2, y2)
                loss2.backward()
                optimizer2.step()
                s2_loss_sum += loss2.item() * len(y2)
                s2_total    += len(y2)
            s2_loss_avg = s2_loss_sum / max(s2_total, 1)
            scheduler2.step()

        scheduler.step()
        ep_time = time.perf_counter() - t0

        # --- Val ---
        model.eval()
        if model2 is not None:
            model2.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                if _cascade_mode:
                    _d_l, _s_l, _sv_l = model(x)
                    _pd = torch.sigmoid(_d_l).cpu().numpy()
                    _ps = torch.sigmoid(_s_l).cpu().numpy()
                    _sl = _sv_l.cpu().numpy()
                    all_preds.extend(_predict_cascade(_pd, _ps, _sl, 0.5, 0.5).tolist())
                elif args.two_stage and model2 is not None:
                    all_preds.extend(_two_stage_predict(x, model, model2, device))
                else:
                    logits = model(x)
                    if _log_prior is not None:
                        from disaster_bench.training.losses import logit_adjust as _la
                        logits = _la(logits, _log_prior, tau=args.logit_adjust_tau)
                        if args.logit_adjust_train and model.training:
                            pass  # already applied above
                    all_preds.extend(logits.argmax(1).cpu().tolist())
                all_labels.extend(y.tolist())


        f1s, macro_f1, precs, recs_pc = _compute_val_metrics(all_preds, all_labels, num_classes=4)

        # --- Print epoch summary ---
        s2_info = f" s2_loss={s2_loss_avg:.4f}" if s2_loss_avg is not None else ""
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss/max(train_total,1):.4f} "
            f"train_acc={train_correct/max(train_total,1):.3f}"
            f"{s2_info} | "
            f"val_macro_f1={macro_f1:.4f} "
            f"[{' '.join(f'{f:.3f}' for f in f1s)}]  {ep_time:.1f}s",
            flush=True,
        )

        # --- Step 0: extended val metrics block ---
        cm = _compute_confusion_matrix(all_preds, all_labels, num_classes=4)
        NO, MINOR, MAJOR = 0, 1, 2
        N_no             = int(cm[NO].sum())
        FP_minor_from_no = int(cm[NO][MINOR])
        FP_major_from_no = int(cm[NO][MAJOR])
        rate_minor       = 1000.0 * FP_minor_from_no / max(N_no, 1)
        rate_major       = 1000.0 * FP_major_from_no / max(N_no, 1)
        pred_minor       = int(cm[:, MINOR].sum())
        pred_major       = int(cm[:, MAJOR].sum())

        print(f"  FP/1k-no: minor={rate_minor:.1f}  major={rate_major:.1f} "
              f"| pred_minor={pred_minor}  pred_major={pred_major}")
        cls_names = ["no", "minor", "major", "dest"]
        print("  " + "  ".join(
            f"{n}: P={precs[i]:.3f} R={recs_pc[i]:.3f} F1={f1s[i]:.3f}"
            for i, n in enumerate(cls_names)
        ), flush=True)

        # --- Step 0: write JSONL ---
        if run_dir is not None:
            entry = {
                "epoch":                 epoch,
                "fp_per_1000_no_minor":  round(rate_minor, 3),
                "fp_per_1000_no_major":  round(rate_major, 3),
                "pred_minor":            pred_minor,
                "pred_major":            pred_major,
                "macro_f1":              round(macro_f1, 4),
                "f1_no":                 round(f1s[0], 4),
                "f1_minor":              round(f1s[1], 4),
                "f1_major":              round(f1s[2], 4),
                "f1_dest":               round(f1s[3], 4),
                "prec_no":               round(precs[0], 4),
                "prec_minor":            round(precs[1], 4),
                "prec_major":            round(precs[2], 4),
                "prec_dest":             round(precs[3], 4),
                "rec_no":                round(recs_pc[0], 4),
                "rec_minor":             round(recs_pc[1], 4),
                "rec_major":             round(recs_pc[2], 4),
                "rec_dest":              round(recs_pc[3], 4),
            }
            with open(run_dir / "val_metrics.jsonl", "a", encoding="utf-8") as jf:
                jf.write(json.dumps(entry) + "\n")

        # --- Save best checkpoint ---
        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            save_checkpoint(
                model, str(best_path),
                model_type=args.model_type,
                epoch=epoch,
                val_macro_f1=macro_f1,
                per_class_f1={DAMAGE_CLASSES[i]: round(f1s[i], 4) for i in range(4)},
                input_size=args.size,
                num_classes=num_classes_s1,
            )
            print(f"  -> saved best.pt (macro_f1={macro_f1:.4f})")
            if args.two_stage and model2 is not None:
                save_checkpoint(
                    model2, str(best2_path),
                    model_type=args.model_type,
                    epoch=epoch,
                    val_macro_f1=macro_f1,
                    per_class_f1={},
                    input_size=args.size,
                    num_classes=num_classes_s2,
                )

        # --- Step 4: Hard Negative Mining ---
        if args.use_hard_negative_mining:
            model.eval()
            hnm_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=False,
                                    collate_fn=collate, num_workers=0)
            all_train_preds = []
            with torch.no_grad():
                for xb, _ in hnm_loader:
                    if _cascade_mode:
                        _dl, _sl, _svl = model(xb.to(device))
                        _pd = torch.sigmoid(_dl).cpu().numpy()
                        _ps = torch.sigmoid(_sl).cpu().numpy()
                        all_train_preds.extend(
                            _predict_cascade(_pd, _ps, _svl.cpu().numpy(), 0.5, 0.5).tolist())
                    else:
                        all_train_preds.extend(model(xb.to(device)).argmax(1).cpu().tolist())

            # Reset to base weights each epoch, then boost hard negatives
            base_w = s1_base_w_np if args.two_stage else base_w_np
            sample_weights_np = _base_sample_weights(train_records_list, base_w)
            hard_neg_count = 0
            for i, (r, pred) in enumerate(zip(train_records_list, all_train_preds)):
                lbl = r["label_idx"]
                if args.two_stage:
                    # Stage1 labels: 0=no-dmg, 1=damaged, 2=dest
                    # Hard negative: GT=no-dmg (0) predicted as damaged (1)
                    if lbl == 0 and pred == 1:
                        sample_weights_np[i] *= args.hnm_mult
                        hard_neg_count += 1
                else:
                    # 4-class: GT=0 (no-dmg) predicted as minor(1) or major(2)
                    if lbl == 0 and pred in (1, 2):
                        sample_weights_np[i] *= args.hnm_mult
                        hard_neg_count += 1

            print(f"  [HNM] boosted {hard_neg_count} hard negatives "
                  f"(mult={args.hnm_mult:.1f})")
            new_sampler = _build_sampler(sample_weights_np, len(train_records_list))
            train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=new_sampler,
                                      collate_fn=collate, num_workers=0)

    # -----------------------------------------------------------------------
    # Save final checkpoints  (skipped in eval_only mode)
    # -----------------------------------------------------------------------
    if not _eval_only:
        save_checkpoint(
            model, str(last_path),
            model_type=args.model_type,
            epoch=args.epochs,
            val_macro_f1=macro_f1,
            per_class_f1={DAMAGE_CLASSES[i]: round(f1s[i], 4) for i in range(4)},
            input_size=args.size,
            num_classes=num_classes_s1,
        )
        if args.two_stage and model2 is not None:
            save_checkpoint(
                model2, str(last2_path),
                model_type=args.model_type,
                epoch=args.epochs,
                val_macro_f1=macro_f1,
                per_class_f1={},
                input_size=args.size,
                num_classes=num_classes_s2,
            )
        print(f"\nDone. Best val macro_f1={best_val_f1:.4f}  -> {out_dir}/best.pt")

    # -----------------------------------------------------------------------
    # Per-class threshold fitting (on val/calib set, using best checkpoint)
    # Deterministic quantile-based fitting; guarantees recall=1.0 when requested.
    # In nested_cv mode val_loader == calib_loader (outer test stays hidden).
    # -----------------------------------------------------------------------
    _thresholds    = None
    _calib_summary = {}

    _calib_mode = getattr(args, "calib_mode", "inner_split")

    # Helper: run inference on a DataLoader and return (probs, labels) arrays
    def _infer_loader(loader):
        model.eval()
        _probs_list, _labels_list = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                logits = model(x)
                if _log_prior is not None:
                    from disaster_bench.training.losses import logit_adjust as _la
                    logits = _la(logits, _log_prior, tau=args.logit_adjust_tau)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                _probs_list.extend(probs.tolist())
                _labels_list.extend(y.tolist())
        return np.array(_probs_list), np.array(_labels_list)

    # Helper: cascade inference — returns (p_damage, p_severe, severity_logits, labels)
    def _infer_loader_cascade(loader):
        model.eval()
        _pd_list, _ps_list, _sl_list, _lbl_list = [], [], [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                _dl, _sl, _svl = model(x)
                _pd_list.extend(torch.sigmoid(_dl).cpu().numpy().tolist())
                _ps_list.extend(torch.sigmoid(_sl).cpu().numpy().tolist())
                _sl_list.extend(_svl.cpu().numpy().tolist())
                _lbl_list.extend(y.tolist())
        return (np.array(_pd_list), np.array(_ps_list),
                np.array(_sl_list), np.array(_lbl_list))

    _need_threshold_fit = (args.threshold_policy in ("per_class_threshold", "ordinal_threshold")
                           and not args.two_stage and not _cascade_mode)
    _need_cascade_threshold_fit = (
        _cascade_mode and args.threshold_policy == "cascade_threshold"
    )

    if _need_threshold_fit:
        _nm   = bool(getattr(args, "never_miss_mode", 0))
        _dset = "calib" if outer_test_recs is not None else "val"
        print(f"\n[Threshold] Fitting thresholds ({args.threshold_policy}) on {_dset} set ...")
        ckpt_best = torch.load(str(best_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt_best["model_state_dict"])
        _all_probs_c, _all_labels_c = _infer_loader(val_loader)

        if args.threshold_policy == "per_class_threshold":
            _thresholds, _calib_summary = _fit_thresholds(
                _all_probs_c, _all_labels_c,
                args.target_recall_minor, args.target_recall_major,
                never_miss=_nm,
            )
        else:  # ordinal_threshold
            _thresholds, _calib_summary = _fit_thresholds_ordinal(
                _all_probs_c, _all_labels_c,
                target_recall_damage=args.target_recall_minor,
                target_recall_severe=args.target_recall_major,
                never_miss=_nm,
            )

        # Save calib predictions for crossfit_pool pooling by sibling folds
        if (outer_test_recs is not None and _calib_mode == "crossfit_pool"
                and args.cv_fold is not None and run_dir is not None):
            _tile_ids_c = [r.get("tile_id", "") for r in val_recs]
            _calib_npz_path = out_dir / "calib_preds.npz"
            np.savez_compressed(
                str(_calib_npz_path),
                y_true=_all_labels_c,
                probs=_all_probs_c,
                tile_ids=np.array(_tile_ids_c, dtype=str),
            )
            print(f"  [crossfit_pool] Saved calib preds -> {_calib_npz_path}")

        thr_path = out_dir / "thresholds.json"
        with open(thr_path, "w") as f:
            json.dump({
                "policy":              args.threshold_policy,
                "thresholds":          _thresholds,
                "target_recall_minor": args.target_recall_minor,
                "target_recall_major": args.target_recall_major,
                "calib_summary":       _calib_summary,
            }, f, indent=2)
        print(f"  Saved {thr_path}")

    # -----------------------------------------------------------------------
    # Cascade threshold fitting (on calib set, using best checkpoint)
    # Fits tau_damage and tau_severe from dedicated binary head outputs.
    # -----------------------------------------------------------------------
    _cascade_thresholds = None
    _cascade_temp_scale = bool(getattr(args, "temp_scale_stage1", 0))
    _cascade_T_damage   = 1.0
    _cascade_T_severe   = 1.0

    print(f"\n[CASCADE DEBUG] _cascade_mode={_cascade_mode}  "
          f"_need_cascade_threshold_fit={_need_cascade_threshold_fit}  "
          f"threshold_policy={getattr(args, 'threshold_policy', 'N/A')}  "
          f"_eval_only={_eval_only}  "
          f"outer_test_recs={'set' if outer_test_recs is not None else 'None'}  "
          f"run_dir={run_dir}", flush=True)

    if _need_cascade_threshold_fit:
        _nm   = bool(getattr(args, "never_miss_mode", 0))
        _dset = "calib" if outer_test_recs is not None else "val"
        print(f"\n[Cascade Threshold] Fitting tau_damage/tau_severe on {_dset} set ...", flush=True)
        try:
            ckpt_best = torch.load(str(best_path), map_location=device, weights_only=False)
            model.load_state_dict(ckpt_best["model_state_dict"])
            _cpd_c, _cps_c, _csl_c, _clbl_c = _infer_loader_cascade(val_loader)

            # Optional temperature scaling on calib logits before threshold fitting
            if _cascade_temp_scale:
                import torch as _tc
                # We need raw logits, not sigmoid probs, for temp scaling
                # Re-collect raw logits from the calib set
                model.eval()
                _dl_list, _sl_list2 = [], []
                with _tc.no_grad():
                    for _xb, _ in val_loader:
                        _dl_b, _sl_b, _ = model(_xb.to(device))
                        _dl_list.extend(_dl_b.cpu().numpy().tolist())
                        _sl_list2.extend(_sl_b.cpu().numpy().tolist())
                _dl_np = np.array(_dl_list)
                _sl_np = np.array(_sl_list2)
                _y_dmg_np = np.isin(_clbl_c, [1, 2, 3]).astype(float)
                _y_sev_np = np.isin(_clbl_c, [2, 3]).astype(float)
                _cascade_T_damage = _fit_temperature_scalar(_dl_np, _y_dmg_np)
                _cascade_T_severe = _fit_temperature_scalar(_sl_np, _y_sev_np)
                _cpd_c = 1.0 / (1.0 + np.exp(-_dl_np / _cascade_T_damage))
                _cps_c = 1.0 / (1.0 + np.exp(-_sl_np / _cascade_T_severe))
                print(f"  [temp_scale] T_damage={_cascade_T_damage:.4f}  "
                      f"T_severe={_cascade_T_severe:.4f}")

            _cascade_thresholds, _cascade_calib_summary = _fit_thresholds_cascade(
                _cpd_c, _cps_c, _clbl_c,
                target_recall_damage=args.target_recall_minor,
                target_recall_severe=args.target_recall_major,
                never_miss=_nm,
            )

            # Save cascade calib preds for crossfit_pool pooling by sibling folds
            if (outer_test_recs is not None and _calib_mode == "crossfit_pool"
                    and args.cv_fold is not None and run_dir is not None):
                _tile_ids_cc = [r.get("tile_id", "") for r in val_recs]
                _casc_npz_path = out_dir / "calib_preds_cascade.npz"
                np.savez_compressed(
                    str(_casc_npz_path),
                    y_true=_clbl_c,
                    p_damage=_cpd_c,
                    p_severe=_cps_c,
                    severity_logits=_csl_c,
                    tile_ids=np.array(_tile_ids_cc, dtype=str),
                )
                print(f"  [crossfit_pool] Saved cascade calib preds -> {_casc_npz_path}", flush=True)

            _casc_thr_path = out_dir / "thresholds_cascade.json"
            with open(_casc_thr_path, "w") as f:
                json.dump({
                    "policy":              "cascade_threshold",
                    "thresholds":          _cascade_thresholds,
                    "target_recall_minor": args.target_recall_minor,
                    "target_recall_major": args.target_recall_major,
                    "temp_scale_stage1":   _cascade_temp_scale,
                    "T_damage":            round(_cascade_T_damage, 6),
                    "T_severe":            round(_cascade_T_severe, 6),
                    "calib_summary":       _cascade_calib_summary,
                }, f, indent=2)
            print(f"  Saved {_casc_thr_path}", flush=True)
        except Exception:
            import traceback as _tb
            print("\n[CASCADE ERROR] Exception in cascade threshold fitting:", flush=True)
            _tb.print_exc()
            raise


    if run_dir is not None:
        from disaster_bench.training.losses import compute_class_counts as _cc, \
            compute_class_prior as _cp
        _tr_counts = _cc(train_recs).tolist()
        _tr_prior  = _cp(train_recs).tolist()
        summary = {
            "run_id":          args.run_id,
            "seed":            args.seed,
            "model_type":      args.model_type,
            "epochs":          args.epochs,
            "batch":           args.batch,
            "lr":              args.lr,
            "loss":            args.loss,
            "loss_info":       _loss_info,
            "weight_mode":     args.weight_mode,
            "sampler":         bool(args.use_sampler),
            "sampler_mode":    args.sampler_mode,
            "sampler_cap":     args.sampler_cap,
            "logit_adjust":    args.logit_adjust,
            "logit_adjust_tau": args.logit_adjust_tau,
            "threshold_policy": args.threshold_policy,
            "calib_mode":      _calib_mode,
            "thresholds":      _thresholds,
            "aug_config":      {
                "rotate90":    args.aug_rotate90,
                "affine":      args.aug_affine,
                "color_jitter": args.aug_color_jitter,
                "noise":       args.aug_noise,
            },
            "train_class_counts": {DAMAGE_CLASSES[i]: int(_tr_counts[i]) for i in range(4)},
            "train_class_prior":  {DAMAGE_CLASSES[i]: round(_tr_prior[i], 6) for i in range(4)},
            "best_val_macro_f1":  round(best_val_f1, 4),
            "n_train":  len(train_recs),
            "n_val":    len(val_recs),
        }
        with open(run_dir / "run_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved {run_dir}/run_summary.json")

    # -----------------------------------------------------------------------
    # Nested CV: evaluate on outer (held-out) test fold — no leakage
    # -----------------------------------------------------------------------
    if outer_test_recs is not None and outer_test_loader is not None and run_dir is not None:
        print("\n[NestedCV] Evaluating on outer test fold (held-out, never seen during training) ...",
              flush=True)
        try:
            ckpt_best = torch.load(str(best_path), map_location=device, weights_only=False)
            model.load_state_dict(ckpt_best["model_state_dict"])
        except Exception:
            import traceback as _tb
            print("[CASCADE ERROR] Failed to load best.pt for outer-test eval:", flush=True)
            _tb.print_exc()
            raise

        # ================================================================
        # CASCADE MODE outer-test eval
        # ================================================================
        if _cascade_mode:
            print(f"  [cascade] Running outer-test inference ...", flush=True)
            try:
                _cpd_t, _cps_t, _csl_t, _all_labels_t = _infer_loader_cascade(outer_test_loader)
                print(f"  [cascade] Inference done: {len(_all_labels_t)} samples", flush=True)
            except Exception:
                import traceback as _tb
                print("[CASCADE ERROR] _infer_loader_cascade failed on outer test:", flush=True)
                _tb.print_exc()
                raise

            # --- crossfit_pool for cascade: pool calib_preds_cascade.npz from siblings ---
            _cascade_test_thresholds = _cascade_thresholds  # fallback: inner-split
            if (_calib_mode == "crossfit_pool" and args.cv_fold is not None
                    and args.cv_folds_path is not None):
                print("  [crossfit_pool/cascade] Loading sibling-fold cascade calib preds ...")
                _cv_k_c = 5
                try:
                    with open(args.cv_folds_path) as _f:
                        _fold_data_c = json.load(_f)
                        _cv_k_c = _fold_data_c.get("k", 5)
                except Exception:
                    pass
                _cpool_pd, _cpool_ps, _cpool_sl, _cpool_lbl = [], [], [], []
                for _sib in range(_cv_k_c):
                    if _sib == args.cv_fold:
                        continue
                    _sib_cnpz = Path(f"models/cv5/fold{_sib}") / "calib_preds_cascade.npz"
                    if not _sib_cnpz.exists():
                        _sib_cnpz = Path(args.out_dir).parent / f"fold{_sib}" / "calib_preds_cascade.npz"
                    if _sib_cnpz.exists():
                        _cd = np.load(str(_sib_cnpz), allow_pickle=True)
                        _cpool_pd.append(_cd["p_damage"])
                        _cpool_ps.append(_cd["p_severe"])
                        _cpool_sl.append(_cd["severity_logits"])
                        _cpool_lbl.append(_cd["y_true"])
                        print(f"    loaded fold{_sib}: {len(_cd['y_true'])} samples")
                    else:
                        print(f"    WARNING: calib_preds_cascade.npz not found for fold{_sib} "
                              f"({_sib_cnpz}) — skipping")
                if _cpool_pd:
                    _pp_pd  = np.concatenate(_cpool_pd,  axis=0)
                    _pp_ps  = np.concatenate(_cpool_ps,  axis=0)
                    _pp_lbl = np.concatenate(_cpool_lbl, axis=0)
                    print(f"  [crossfit_pool/cascade] Pooled {len(_pp_lbl)} samples from "
                          f"{len(_cpool_pd)} sibling folds. Refitting cascade thresholds ...")
                    _nm_c = bool(getattr(args, "never_miss_mode", 0))
                    _cascade_test_thresholds, _casc_pool_summary = _fit_thresholds_cascade(
                        _pp_pd, _pp_ps, _pp_lbl,
                        target_recall_damage=args.target_recall_minor,
                        target_recall_severe=args.target_recall_major,
                        never_miss=_nm_c,
                    )
                    _casc_pool_path = run_dir / "thresholds_crossfit.json"
                    with open(_casc_pool_path, "w") as _f:
                        json.dump({
                            "policy": "cascade_threshold",
                            "thresholds": _cascade_test_thresholds,
                            "calib_summary": _casc_pool_summary,
                            "n_pooled_samples": int(len(_pp_lbl)),
                            "sibling_folds": [s for s in range(_cv_k_c) if s != args.cv_fold],
                        }, _f, indent=2)
                    print(f"  [crossfit_pool/cascade] Saved {_casc_pool_path}")
                else:
                    print("  [crossfit_pool/cascade] No sibling cascade calib files found; "
                          "falling back to inner-split thresholds.")
                    # Still write thresholds_crossfit.json so fold-0-only runs produce the file
                    if _cascade_thresholds is not None and run_dir is not None:
                        _casc_pool_path = run_dir / "thresholds_crossfit.json"
                        with open(_casc_pool_path, "w") as _f:
                            json.dump({
                                "policy": "cascade_threshold",
                                "thresholds": _cascade_thresholds,
                                "calib_summary": _cascade_calib_summary,
                                "n_pooled_samples": 0,
                                "sibling_folds": [],
                                "note": "inner-split fallback (no sibling calib files found)",
                            }, _f, indent=2)
                        print(f"  [crossfit_pool/cascade] Saved fallback {_casc_pool_path}")

            # --- Cascade default (tau=0.5) proxy for comparison ---
            _preds_casc_argmax = _predict_cascade(_cpd_t, _cps_t, _csl_t, 0.5, 0.5)
            casc_argmax_metrics = _make_eval_metrics_cascade(
                _preds_casc_argmax, _all_labels_t, None, temp_scale=False
            )
            with open(run_dir / "test_metrics_cascade_argmax.json", "w") as f:
                json.dump(casc_argmax_metrics, f, indent=2)
            print(f"  [cascade/argmax-proxy]  macro_f1={casc_argmax_metrics['macro_f1']:.4f}  "
                  f"FN_damage={casc_argmax_metrics['FN_damage']}  "
                  f"FN_severe={casc_argmax_metrics['FN_severe']}")

            # --- Cascade threshold metrics ---
            if _cascade_test_thresholds is not None:
                _tau_dmg = float(_cascade_test_thresholds.get("tau_damage", 0.5))
                _tau_sev = float(_cascade_test_thresholds.get("tau_severe", 0.5))
                _preds_casc = _predict_cascade(_cpd_t, _cps_t, _csl_t, _tau_dmg, _tau_sev)
                casc_metrics = _make_eval_metrics_cascade(
                    _preds_casc, _all_labels_t, _cascade_test_thresholds,
                    temp_scale=_cascade_temp_scale,
                )
                with open(run_dir / "test_metrics_cascade.json", "w") as f:
                    json.dump(casc_metrics, f, indent=2)
                print(f"  [cascade]  macro_f1={casc_metrics['macro_f1']:.4f}  "
                      f"FN_damage={casc_metrics['FN_damage']}  "
                      f"FN_severe={casc_metrics['FN_severe']}  "
                      f"recall_damage={casc_metrics['recall_damage']}  "
                      f"recall_severe={casc_metrics['recall_severe']}")
                _fn_casc_total = casc_metrics["FN_damage"] + casc_metrics["FN_severe"]
                if _fn_casc_total == 0:
                    print("  [cascade] FN_damage=0 FN_severe=0 -> never-miss achieved!")

                # --- Cascade missed positives debug dump ---
                _miss_rows_c = []
                _lbl_arr_c   = np.asarray(_all_labels_t)
                _prd_arr_c   = np.asarray(_preds_casc)
                _miss_dmg_c  = np.isin(_lbl_arr_c, [1, 2, 3]) & (_prd_arr_c == 0)
                _miss_sev_c  = np.isin(_lbl_arr_c, [2, 3]) & ~np.isin(_prd_arr_c, [2, 3])
                _miss_mask_c = _miss_dmg_c | _miss_sev_c
                if _miss_mask_c.any():
                    for _mi in np.where(_miss_mask_c)[0]:
                        _row = {
                            "idx":            int(_mi),
                            "true_label":     int(_lbl_arr_c[_mi]),
                            "pred_label":     int(_prd_arr_c[_mi]),
                            "p_damage":       round(float(_cpd_t[_mi]), 6),
                            "p_severe":       round(float(_cps_t[_mi]), 6),
                            "severity_probs": [round(float(v), 6) for v in
                                               np.exp(_csl_t[_mi]) / np.exp(_csl_t[_mi]).sum()],
                            "missed_damage":  bool(_miss_dmg_c[_mi]),
                            "missed_severe":  bool(_miss_sev_c[_mi]),
                        }
                        if _mi < len(outer_test_recs):
                            _row["tile_id"] = outer_test_recs[_mi].get("tile_id", "")
                        _miss_rows_c.append(_row)
                    _casc_miss_path = run_dir / "missed_positives_cascade.json"
                    with open(_casc_miss_path, "w") as _f:
                        json.dump(_miss_rows_c, _f, indent=2)
                    print(f"  [cascade/debug] {len(_miss_rows_c)} missed -> {_casc_miss_path}")

        # ================================================================
        # NON-CASCADE outer-test eval (existing code, unchanged)
        # ================================================================
        else:
            _all_probs_t, _all_labels_t = _infer_loader(outer_test_loader)

            # --- crossfit_pool: pool calib preds from sibling folds ---
            _test_thresholds = _thresholds   # default: use inner-split thresholds
            if (_calib_mode == "crossfit_pool" and args.cv_fold is not None
                    and args.cv_folds_path is not None):
                print(f"  [crossfit_pool] Loading sibling-fold calib preds ...")
                _pool_probs_list, _pool_labels_list = [], []
                _cv_k = 5   # default; could introspect the folds JSON
                try:
                    with open(args.cv_folds_path) as _f:
                        _fold_data_pool = json.load(_f)
                        _cv_k = _fold_data_pool.get("k", 5)
                except Exception:
                    pass
                for _sib in range(_cv_k):
                    if _sib == args.cv_fold:
                        continue   # exclude current fold to prevent leakage
                    _sib_npz = Path(f"models/cv5/fold{_sib}") / "calib_preds.npz"
                    if not _sib_npz.exists():
                        # Try out_dir-relative path
                        _out_base = Path(args.out_dir).parent
                        _sib_npz = _out_base / f"fold{_sib}" / "calib_preds.npz"
                    if _sib_npz.exists():
                        _d = np.load(str(_sib_npz), allow_pickle=True)
                        _pool_probs_list.append(_d["probs"])
                        _pool_labels_list.append(_d["y_true"])
                        print(f"    loaded fold{_sib}: {len(_d['y_true'])} samples")
                    else:
                        print(f"    WARNING: calib_preds.npz not found for fold{_sib} "
                              f"({_sib_npz}) — skipping")
                if _pool_probs_list:
                    _pool_probs  = np.concatenate(_pool_probs_list,  axis=0)
                    _pool_labels = np.concatenate(_pool_labels_list, axis=0)
                    print(f"  [crossfit_pool] Pooled {len(_pool_labels)} calib samples from "
                          f"{len(_pool_probs_list)} sibling folds. Refitting thresholds ...")
                    _nm = bool(getattr(args, "never_miss_mode", 0))
                    if args.threshold_policy == "per_class_threshold":
                        _test_thresholds, _pool_summary = _fit_thresholds(
                            _pool_probs, _pool_labels,
                            args.target_recall_minor, args.target_recall_major,
                            never_miss=_nm,
                        )
                    else:
                        _test_thresholds, _pool_summary = _fit_thresholds_ordinal(
                            _pool_probs, _pool_labels,
                            target_recall_damage=args.target_recall_minor,
                            target_recall_severe=args.target_recall_major,
                            never_miss=_nm,
                        )
                    thr_pool_path = run_dir / "thresholds_crossfit.json"
                    with open(thr_pool_path, "w") as _f:
                        json.dump({
                            "policy": args.threshold_policy,
                            "thresholds": _test_thresholds,
                            "calib_summary": _pool_summary,
                            "n_pooled_samples": int(len(_pool_labels)),
                            "sibling_folds": [s for s in range(_cv_k) if s != args.cv_fold],
                        }, _f, indent=2)
                    print(f"  [crossfit_pool] Saved {thr_pool_path}")
                else:
                    print("  [crossfit_pool] No sibling calib files found; "
                          "falling back to inner-split thresholds.")

            # --- Argmax metrics ---
            _preds_argmax  = predict_with_policy(_all_probs_t, "argmax", None)
            argmax_metrics = _make_eval_metrics_ordinal(
                _preds_argmax, _all_labels_t, _all_probs_t, "argmax", None
            )
            with open(run_dir / "test_metrics_argmax.json", "w") as f:
                json.dump(argmax_metrics, f, indent=2)
            print(f"  [test/argmax]  macro_f1={argmax_metrics['macro_f1']:.4f}  "
                  f"FN_minor={argmax_metrics['FN_minor_test']}  "
                  f"FN_major={argmax_metrics['FN_major_test']}  "
                  f"FN_damage={argmax_metrics['FN_damage']}  "
                  f"FN_severe={argmax_metrics['FN_severe']}")

            # --- Thresholded / ordinal metrics ---
            if _test_thresholds is not None:
                _policy = args.threshold_policy
                _preds_thresh  = predict_with_policy(_all_probs_t, _policy, _test_thresholds)
                thresh_metrics = _make_eval_metrics_ordinal(
                    _preds_thresh, _all_labels_t, _all_probs_t, _policy, _test_thresholds
                )
                _out_fname = "test_metrics_ordinal.json" if _policy == "ordinal_threshold" \
                             else "test_metrics_thresholded.json"
                with open(run_dir / _out_fname, "w") as f:
                    json.dump(thresh_metrics, f, indent=2)
                print(f"  [test/{_policy}]  macro_f1={thresh_metrics['macro_f1']:.4f}  "
                      f"FN_damage={thresh_metrics['FN_damage']}  "
                      f"FN_severe={thresh_metrics['FN_severe']}  "
                      f"recall_damage={thresh_metrics['recall_damage']}  "
                      f"recall_severe={thresh_metrics['recall_severe']}")
                _fn_total = thresh_metrics["FN_damage"] + thresh_metrics["FN_severe"]
                if _fn_total == 0:
                    print(f"  [{_policy}] FN_damage=0 FN_severe=0 -> never-miss achieved!")

                # --- Debug dump of missed positives ---
                _missed_rows = []
                _labels_arr = np.asarray(_all_labels_t)
                _preds_arr  = np.asarray(_preds_thresh)
                _p_damage   = _all_probs_t[:, 1] + _all_probs_t[:, 2] + _all_probs_t[:, 3]
                _p_severe   = _all_probs_t[:, 2] + _all_probs_t[:, 3]
                # Missed damage: true label in {1,2,3} but predicted as 0
                _miss_damage_mask = np.isin(_labels_arr, [1, 2, 3]) & (_preds_arr == 0)
                # Missed severe: true label in {2,3} but predicted as not in {2,3}
                _miss_severe_mask = np.isin(_labels_arr, [2, 3]) & ~np.isin(_preds_arr, [2, 3])
                _missed_mask = _miss_damage_mask | _miss_severe_mask
                if _missed_mask.any():
                    _miss_idxs = np.where(_missed_mask)[0]
                    for _mi in _miss_idxs:
                        _row = {
                            "idx":       int(_mi),
                            "true_label": int(_labels_arr[_mi]),
                            "pred_label": int(_preds_arr[_mi]),
                            "p_no":       round(float(_all_probs_t[_mi, 0]), 6),
                            "p_minor":    round(float(_all_probs_t[_mi, 1]), 6),
                            "p_major":    round(float(_all_probs_t[_mi, 2]), 6),
                            "p_destroyed": round(float(_all_probs_t[_mi, 3]), 6),
                            "p_damage":   round(float(_p_damage[_mi]), 6),
                            "p_severe":   round(float(_p_severe[_mi]), 6),
                            "missed_damage": bool(_miss_damage_mask[_mi]),
                            "missed_severe": bool(_miss_severe_mask[_mi]),
                        }
                        if outer_test_recs is not None and _mi < len(outer_test_recs):
                            _row["tile_id"] = outer_test_recs[_mi].get("tile_id", "")
                        _missed_rows.append(_row)
                    _debug_path = run_dir / f"missed_positives_{_policy}.json"
                    with open(_debug_path, "w") as _f:
                        json.dump(_missed_rows, _f, indent=2)
                    print(f"  [debug] {len(_missed_rows)} missed positives -> {_debug_path}")
            else:
                print("  [test] No thresholds fitted — only test_metrics_argmax.json written. "
                      "Add --threshold_policy per_class_threshold or ordinal_threshold to also write "
                      "test_metrics_thresholded.json / test_metrics_ordinal.json.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Train damage classifier")

    # Existing args (unchanged defaults)
    p.add_argument("--index_csv",    default="data/processed/index.csv")
    p.add_argument("--crops_dir",    default="data/processed/crops_oracle")
    p.add_argument("--out_dir",      default="models/six_channel")
    p.add_argument("--model_type",   default="six_channel",
                   choices=["six_channel", "pre_post_diff", "siamese", "centroid_patch"])
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch",        type=int,   default=32)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--size",         type=int,   default=128)
    p.add_argument("--dropout",      type=float, default=0.4)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--device",       default=None, help="cuda | cpu | auto")
    p.add_argument("--resume",       action="store_true")

    # Reproducibility
    p.add_argument("--seed",         type=int, default=42,
                   help="Global RNG seed (random, numpy, torch, cuda)")

    # Step 0 — tracking
    p.add_argument("--run_id",       type=str, default=None,
                   help="Write val_metrics.jsonl to runs/<run_id>/")

    # Step 1 — sampler
    p.add_argument("--use_sampler",  type=int, choices=[0, 1], default=0,
                   help="Use WeightedRandomSampler (replaces shuffle=True)")

    # Step 2B — batch histogram
    p.add_argument("--log_batch_class_counts", type=int, choices=[0, 1], default=0,
                   help="Log first 50 train-batch label histograms epoch 1 (needs --run_id)")

    # Step 3 — weight mode
    p.add_argument("--weight_mode",  type=str, default="normalized_invfreq",
                   choices=["normalized_invfreq", "capped_floored", "none"])
    p.add_argument("--w_min",        type=float, default=0.25,
                   help="Min loss weight for capped_floored")
    p.add_argument("--w_max",        type=float, default=5.0,
                   help="Max loss weight for capped_floored")

    # Step 4 — hard negative mining
    p.add_argument("--use_hard_negative_mining", type=int, choices=[0, 1], default=0)
    p.add_argument("--hnm_mult",     type=float, default=5.0,
                   help="Sampling weight multiplier for hard negatives")

    # Step 5 — two-stage
    p.add_argument("--two_stage",    type=int, choices=[0, 1], default=0,
                   help="Stage1=3-class then Stage2=2-class (minor/major)")

    # Step 6 — pretrained encoder
    p.add_argument("--init_mode",    type=str, default="scratch",
                   choices=["scratch", "pretrained"])
    p.add_argument("--pretrained_ckpt_path", type=str, default=None,
                   help="Encoder checkpoint path (required if init_mode=pretrained)")

    # CV — fold-based train/val split (overrides default 80/20 when both are given)
    p.add_argument("--cv_folds_path", type=str, default=None,
                   help="Path to cv_folds JSON (output of make_cv_folds.py)")
    p.add_argument("--cv_fold",       type=int, default=None,
                   help="Which fold to hold out as val (0..k-1)")

    # Disaster split — overrides both CV and default 80/20 when both are given
    p.add_argument("--train_disasters", type=str, default=None,
                   help="Comma-separated disaster IDs for train, e.g. socal-fire")
    p.add_argument("--val_disasters",   type=str, default=None,
                   help="Comma-separated disaster IDs for val, e.g. santa-rosa-wildfire")

    # Long-tail losses (Deliverable A)
    p.add_argument("--loss", type=str, default="ce",
                   choices=["ce", "cb_ce", "focal", "ldam_drw"],
                   help="Loss function: ce (default), cb_ce, focal, ldam_drw")
    p.add_argument("--beta",            type=float, default=0.9999,
                   help="CB beta for cb_ce / focal / ldam_drw DRW phase (default 0.9999)")
    p.add_argument("--gamma",           type=float, default=2.0,
                   help="Focal loss gamma (default 2.0)")
    p.add_argument("--max_m",           type=float, default=0.5,
                   help="LDAM max margin (default 0.5)")
    p.add_argument("--ldam_s",          type=float, default=30.0,
                   help="LDAM logit scale factor (default 30.0)")
    p.add_argument("--drw_start_epoch", type=int,   default=10,
                   help="DRW: epoch at which to switch to class-balanced weights (default 10)")

    # Logit adjustment (Deliverable B)
    p.add_argument("--logit_adjust",       type=str, default="none",
                   choices=["none", "train_prior"],
                   help="Logit adjustment at eval time (default none)")
    p.add_argument("--logit_adjust_tau",   type=float, default=1.0,
                   help="Logit adjustment tau (default 1.0)")
    p.add_argument("--logit_adjust_train", type=int, choices=[0, 1], default=0,
                   help="Apply logit adjustment during training loss (default 0=off)")

    # Sampler mode (Deliverable C)
    p.add_argument("--sampler_mode", type=str, default="invfreq",
                   choices=["invfreq", "invfreq_capped", "class_quota_batch"],
                   help="Sampler strategy when --use_sampler 1 (default invfreq)")
    p.add_argument("--sampler_cap",  type=float, default=5.0,
                   help="Per-sample weight cap multiplier for invfreq_capped (default 5.0)")

    # Augmentations (Deliverable D)
    p.add_argument("--aug_rotate90",     type=float, default=0.0,
                   help="Probability of random 90/180/270 rotation (default 0.0=off)")
    p.add_argument("--aug_affine",       type=float, default=0.0,
                   help="Probability of small affine warp (default 0.0=off)")
    p.add_argument("--aug_color_jitter", type=float, default=0.0,
                   help="Probability of brightness/contrast jitter (default 0.0=off)")
    p.add_argument("--aug_noise",        type=float, default=0.0,
                   help="Probability of additive Gaussian noise (default 0.0=off)")

    # Threshold policy (Deliverable E)
    p.add_argument("--threshold_policy", type=str, default="argmax",
                   choices=["argmax", "per_class_threshold", "ordinal_threshold",
                            "cascade_threshold"],
                   help="Prediction policy: argmax (default), per_class_threshold, "
                        "ordinal_threshold, or cascade_threshold (requires --cascade_mode multihead)")
    p.add_argument("--target_recall_minor", type=float, default=0.80,
                   help="Target recall for minor-damage threshold (default 0.80)")
    p.add_argument("--target_recall_major", type=float, default=0.80,
                   help="Target recall for major-damage threshold (default 0.80)")

    # Nested CV + never-miss mode (Deliverables A / C / D)
    p.add_argument("--nested_cv",       type=int, choices=[0, 1], default=0,
                   help="Nested CV: train on inner split, eval on outer fold; "
                        "requires --cv_folds_path + --cv_fold (default 0)")
    p.add_argument("--calib_fraction",  type=float, default=0.2,
                   help="Fraction of outer-train tiles held for calibration (default 0.2)")
    p.add_argument("--calib_seed",      type=int, default=None,
                   help="RNG seed for calib split (default: same as --seed)")
    p.add_argument("--selection_metric", type=str, default="macro_f1",
                   help="Metric for best-checkpoint selection on calib set (default macro_f1)")
    p.add_argument("--never_miss_mode", type=int, choices=[0, 1], default=0,
                   help="Force target_recall=1.0 for minor+major (FN=0 mode) (default 0)")
    p.add_argument("--calib_mode", type=str, default="inner_split",
                   choices=["inner_split", "crossfit_pool"],
                   help="Calibration mode: inner_split (default) or crossfit_pool "
                        "(pool OOF preds from sibling folds for stable threshold fitting)")
    p.add_argument("--eval_only", type=int, choices=[0, 1], default=0,
                   help="Skip training; load existing best.pt and run outer-test eval only "
                        "(use after all folds are trained with --calib_mode crossfit_pool)")

    # Cascade mode (multi-head CNN, separate binary + severity heads)
    p.add_argument("--cascade_mode", type=str, default="off",
                   choices=["off", "multihead"],
                   help="Cascade model mode: off (default) or multihead "
                        "(shared backbone + damage/severe/severity heads)")
    p.add_argument("--stage1_loss", type=str, default="bce",
                   choices=["bce", "focal", "cb_bce"],
                   help="Loss for binary damage/severe heads (default bce)")
    p.add_argument("--stage2_loss", type=str, default="ce",
                   choices=["ce", "cb_ce"],
                   help="Loss for severity (3-class) head (default ce)")
    p.add_argument("--temp_scale_stage1", type=int, choices=[0, 1], default=0,
                   help="Fit temperature scalar for damage/severe heads before threshold fitting "
                        "(default 0)")

    run(p.parse_args())


if __name__ == "__main__":
    main()
