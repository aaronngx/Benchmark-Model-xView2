#!/usr/bin/env python3
"""
Rule-based fusion of binary ensemble scores + VLM second-opinion labels.

For buildings reviewed by the VLM:
  final_minor_score = ensemble_minor_score + alpha * 1[VLM=minor-damage]
  final_major_score = ensemble_major_score + alpha * 1[VLM=major-damage]
  final_triage_score = max(final_minor_score, final_major_score)

For buildings NOT reviewed by the VLM (not in the shortlist):
  scores are kept unchanged from the ensemble.

Output is a re-ranked CSV of all buildings, sorted by final_triage_score,
with a 'vlm_reviewed' flag and the VLM's label attached where available.

Usage:
  python scripts/fuse_vlm_scores.py \
      --minor_scores models/binary_ensemble/minor_cv/val_scores.csv \
      --major_scores models/binary_ensemble/major_cv/val_scores.csv \
      --vlm_preds    reports/vlm/predictions_cascade.csv \
      --out_csv      reports/vlm/fused_scores.csv \
      --alpha        0.3

Then evaluate with:
  python scripts/eval_at_fp_budgets.py \
      --scores_csv reports/vlm/fused_scores.csv \
      --score_col  final_triage_score
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


MINOR_LABEL = "minor-damage"
MAJOR_LABEL = "major-damage"
DEST_LABEL  = "destroyed"


def load_scores(csv_path: str) -> dict[str, float]:
    """Return {building_id: calib_score} from a val_scores.csv."""
    store: dict[str, float] = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            building_id = f"{row['tile_id']}:{row['uid']}"
            store[building_id] = float(row["calib_score"])
    return store


def load_vlm_preds(csv_path: str) -> dict[str, dict]:
    """Return {building_id: {pred_label, reasoning, ...}} from VLM predictions CSV."""
    store: dict[str, dict] = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # Skip filtered rows (no actual VLM call made)
            if row.get("filter_applied", ""):
                continue
            building_id = f"{row['tile_id']}:{row['uid']}"
            store[building_id] = {
                "vlm_label":     row.get("pred_label", "no-damage"),
                "vlm_reasoning": row.get("reasoning", "")[:120],
                "gt_label":      row.get("gt_label", ""),
            }
    return store


def main() -> None:
    p = argparse.ArgumentParser(
        description="Fuse binary ensemble scores with VLM second-opinion labels"
    )
    p.add_argument("--minor_scores", required=True,
                   help="val_scores.csv from minor ensemble")
    p.add_argument("--major_scores", required=True,
                   help="val_scores.csv from major ensemble")
    p.add_argument("--vlm_preds",    required=True,
                   help="VLM predictions CSV from run_vlm.py (cascade mode)")
    p.add_argument("--alpha",        type=float, default=0.3,
                   help="Score boost for VLM agreement (default 0.3)")
    p.add_argument("--out_csv",      default="reports/vlm/fused_scores.csv",
                   help="Output fused scores CSV")
    args = p.parse_args()

    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------
    print(f"Loading minor scores: {args.minor_scores}")
    minor_scores = load_scores(args.minor_scores)

    print(f"Loading major scores: {args.major_scores}")
    major_scores = load_scores(args.major_scores)

    print(f"Loading VLM preds:    {args.vlm_preds}")
    vlm_preds = load_vlm_preds(args.vlm_preds)
    print(f"  {len(vlm_preds)} buildings with actual VLM calls (filter_applied='' rows)")

    # ------------------------------------------------------------------
    # Fuse scores for all buildings
    # ------------------------------------------------------------------
    all_ids = set(minor_scores) | set(major_scores)
    records = []
    n_boosted_minor = 0
    n_boosted_major = 0

    for building_id in all_ids:
        minor_s = minor_scores.get(building_id, 0.0)
        major_s = major_scores.get(building_id, 0.0)
        vlm     = vlm_preds.get(building_id)
        vlm_reviewed = vlm is not None
        vlm_label    = vlm["vlm_label"]    if vlm else ""
        vlm_reasoning = vlm["vlm_reasoning"] if vlm else ""
        gt_label     = vlm["gt_label"]     if vlm else ""

        # Apply alpha boost
        final_minor = minor_s
        final_major = major_s
        if vlm_reviewed:
            if vlm_label == MINOR_LABEL:
                final_minor += args.alpha
                n_boosted_minor += 1
            elif vlm_label == MAJOR_LABEL:
                final_major += args.alpha
                n_boosted_major += 1
            # destroyed or no-damage: no boost (scores unchanged)
            # Optionally downweight: if VLM says no-damage, could subtract alpha
            # but start with additive-only to avoid penalizing false VLM negatives

        tile_id, uid = building_id.split(":", 1)
        records.append({
            "building_id":       building_id,
            "tile_id":           tile_id,
            "uid":               uid,
            "minor_score":       round(minor_s,     6),
            "major_score":       round(major_s,     6),
            "final_minor_score": round(final_minor, 6),
            "final_major_score": round(final_major, 6),
            "final_triage_score": round(max(final_minor, final_major), 6),
            "vlm_reviewed":      int(vlm_reviewed),
            "vlm_label":         vlm_label,
            "vlm_reasoning":     vlm_reasoning,
            "gt_label":          gt_label,
        })

    # Sort by final_triage_score descending
    records.sort(key=lambda r: r["final_triage_score"], reverse=True)

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "building_id", "tile_id", "uid",
        "minor_score", "major_score",
        "final_minor_score", "final_major_score", "final_triage_score",
        "vlm_reviewed", "vlm_label", "vlm_reasoning", "gt_label",
    ]
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(records)

    print(f"\nFusion summary (alpha={args.alpha}):")
    print(f"  Total buildings:    {len(records)}")
    print(f"  VLM-reviewed:       {len(vlm_preds)}")
    print(f"  Boosted (minor):    {n_boosted_minor}")
    print(f"  Boosted (major):    {n_boosted_major}")
    print(f"\nOutput -> {out}")
    print(f"Top-10 by final_triage_score:")
    for r in records[:10]:
        tag = f"[VLM={r['vlm_label']}]" if r["vlm_reviewed"] else ""
        print(f"  {r['building_id'][:50]:<52}  "
              f"triage={r['final_triage_score']:.4f}  "
              f"gt={r['gt_label']:<16} {tag}")


if __name__ == "__main__":
    main()
