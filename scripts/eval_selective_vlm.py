#!/usr/bin/env python3
"""
Evaluate selective CNN→VLM override strategy.

Reads CNN probs (full val set) and VLM predictions (uncertain subset).
Tests multiple override rules and reports full-val metrics vs CNN-only baseline.

Override rules:
  A — Replace all uncertain-subset predictions with VLM output
  B — Replace only when VLM differs from CNN AND VLM confidence is high
  C — Only allow VLM to override into minor/major (not no-damage or destroyed)

VLM prediction CSV format (from run_vlm.py):
  building_id, pred_label, ...

Usage:
  python scripts/eval_selective_vlm.py \\
    --cnn_probs reports/fusion/cnn_probs.csv \\
    --vlm_preds reports/vlm/preds_uncertain50.csv \\
    --vlm_preds reports/vlm/preds_uncertain100.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


LABEL2IDX = {
    "no-damage":    0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed":    3,
}
IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(preds: np.ndarray, labels: np.ndarray):
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


def _print_metrics(label: str, preds: np.ndarray, gts: np.ndarray):
    f1s, mf1, precs, recs = _compute_metrics(preds, gts)
    print(f"\n[{label}]  macro_F1={mf1:.4f}")
    cls = ["no-dmg", "minor", "major", "dest"]
    for i, c in enumerate(cls):
        print(f"  {c}: P={precs[i]:.3f} R={recs[i]:.3f} F1={f1s[i]:.3f}")
    return f1s, mf1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate selective VLM override")
    parser.add_argument("--cnn_probs", default="reports/fusion/cnn_probs.csv")
    parser.add_argument("--vlm_preds", nargs="+", required=True,
                        help="One or more VLM prediction CSVs (evaluated separately)")
    parser.add_argument("--vlm_confidence_threshold", type=float, default=0.8,
                        help="Rule B: min VLM confidence to allow override (default 0.8)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load CNN probs (full val set)
    # ------------------------------------------------------------------
    cnn_data: dict[str, dict] = {}
    with open(args.cnn_probs, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cnn_data[row["building_id"]] = row

    all_ids = sorted(cnn_data.keys())
    labels  = np.array([int(cnn_data[bid]["gt_label_idx"]) for bid in all_ids])
    cnn_preds = np.array([int(cnn_data[bid]["pred_argmax"]) for bid in all_ids])
    confidence = np.array([float(cnn_data[bid]["confidence_max"]) for bid in all_ids])

    print(f"Val set: {len(all_ids)} buildings")
    label_dist = Counter(int(l) for l in labels)
    print(f"GT distribution: { {IDX2LABEL[k]: v for k,v in sorted(label_dist.items())} }")

    # Baseline
    f1s_base, mf1_base = _print_metrics("CNN Baseline (full val set)", cnn_preds, labels)
    id_to_idx = {bid: i for i, bid in enumerate(all_ids)}

    for vlm_path in args.vlm_preds:
        print(f"\n{'='*60}")
        print(f"VLM predictions: {vlm_path}")

        # ------------------------------------------------------------------
        # Load VLM predictions
        # ------------------------------------------------------------------
        vlm_raw: dict[str, dict] = {}
        with open(vlm_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Support building_id or tile_id:uid column names
                bid = row.get("building_id", "")
                if not bid:
                    tid = row.get("tile_id", "")
                    uid = row.get("uid", "")
                    bid = f"{tid}:{uid}" if tid and uid else ""
                if bid:
                    vlm_raw[bid] = row

        matched = {bid: row for bid, row in vlm_raw.items() if bid in id_to_idx}
        print(f"VLM predictions loaded: {len(vlm_raw)} total, "
              f"{len(matched)} matched to val set")

        if not matched:
            print("  WARNING: No matched buildings. Skipping.", file=sys.stderr)
            continue

        # Parse VLM predicted labels
        def _parse_vlm_label(row: dict) -> int | None:
            # Try several column names from different run_vlm.py output formats
            for col in ("pred_label", "pred_label_idx", "vlm_pred", "prediction"):
                val = row.get(col, "").strip()
                if val:
                    # Numeric
                    if val.isdigit():
                        return int(val)
                    # String label
                    if val in LABEL2IDX:
                        return LABEL2IDX[val]
                    # Partial match
                    for label, idx in LABEL2IDX.items():
                        if label.startswith(val) or val in label:
                            return idx
            return None

        vlm_preds_map: dict[str, int] = {}
        n_parse_fail = 0
        for bid, row in matched.items():
            idx = _parse_vlm_label(row)
            if idx is None:
                n_parse_fail += 1
            else:
                vlm_preds_map[bid] = idx

        if n_parse_fail > 0:
            print(f"  WARNING: {n_parse_fail} rows could not be parsed. "
                  f"Check column names in {vlm_path}")

        subset_ids = sorted(vlm_preds_map.keys())
        subset_idxs = [id_to_idx[bid] for bid in subset_ids]
        vlm_subset_preds = np.array([vlm_preds_map[bid] for bid in subset_ids])
        cnn_subset_preds = cnn_preds[subset_idxs]
        subset_labels    = labels[subset_idxs]

        print(f"\n  Subset: {len(subset_ids)} buildings")
        subset_label_dist = Counter(int(l) for l in subset_labels)
        print(f"  GT distribution: { {IDX2LABEL[k]: v for k,v in sorted(subset_label_dist.items())} }")

        # VLM accuracy on uncertain subset
        _print_metrics(f"VLM-only on uncertain subset ({Path(vlm_path).stem})",
                       vlm_subset_preds, subset_labels)
        _print_metrics("CNN-only on same uncertain subset",
                       cnn_subset_preds, subset_labels)

        # ------------------------------------------------------------------
        # Apply override rules on full val set
        # ------------------------------------------------------------------
        for rule_name, desc in [
            ("A", "Replace all uncertain-subset predictions with VLM"),
            ("B", f"Replace only when VLM differs from CNN AND VLM label not 'no-damage'"),
            ("C", "Only allow VLM to override into minor/major"),
        ]:
            preds_override = cnn_preds.copy()
            n_changed = 0
            n_correct_before = 0
            n_correct_after  = 0

            for bid in subset_ids:
                i = id_to_idx[bid]
                vlm_p = vlm_preds_map[bid]
                cnn_p = cnn_preds[i]
                gt    = labels[i]

                if rule_name == "A":
                    new_p = vlm_p

                elif rule_name == "B":
                    # Only override when VLM disagrees with CNN
                    # and VLM doesn't predict no-damage (class 0)
                    if vlm_p != cnn_p and vlm_p != 0:
                        new_p = vlm_p
                    else:
                        new_p = cnn_p

                elif rule_name == "C":
                    # Only allow override into minor(1) or major(2)
                    if vlm_p in (1, 2):
                        new_p = vlm_p
                    else:
                        new_p = cnn_p

                if new_p != cnn_p:
                    n_changed += 1
                    was_correct = (cnn_p == gt)
                    now_correct = (new_p == gt)
                    n_correct_before += int(was_correct)
                    n_correct_after  += int(now_correct)

                preds_override[i] = new_p

            f1s_ov, mf1_ov = _print_metrics(
                f"Rule {rule_name}: {desc}", preds_override, labels)

            delta_macro  = mf1_ov - mf1_base
            delta_minor  = f1s_ov[1] - f1s_base[1]
            delta_major  = f1s_ov[2] - f1s_base[2]
            n_fixed      = n_correct_after - n_correct_before
            print(f"  Changes: {n_changed} predictions altered  |  "
                  f"Corrections: {n_fixed:+d} (net)")
            if n_changed > 0:
                cost_per_fix = n_changed / max(abs(n_fixed), 1)
                print(f"  Cost per corrected case: {cost_per_fix:.1f} VLM calls / correction")
            print(f"  d_macro_F1={delta_macro:+.4f}  "
                  f"d_f1_minor={delta_minor:+.3f}  d_f1_major={delta_major:+.3f}")

        print(f"\n[Practical note]")
        print(f"  If VLM changes many predictions but fixes very few, the practical")
        print(f"  value is low regardless of F1 movement — check cost_per_fix above.")


if __name__ == "__main__":
    main()
