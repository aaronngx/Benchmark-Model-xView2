#!/usr/bin/env python3
"""
Generate a shortlist of building IDs for Stage-2 VLM review.

Inputs:
  --minor_scores   val_scores.csv from binary_ensemble/minor_cv/
  --major_scores   val_scores.csv from binary_ensemble/major_cv/
  --cnn_preds      CSV with columns: tile_id, uid, pred_class, max_softmax,
                   prob_no, prob_minor, prob_major, prob_dest
                   (optional; produced by scripts/export_cnn_preds.py)

Selection logic (applied in order):
  1. Compute triage_score = max(minor_calib_score, major_calib_score)
  2. Sort descending by triage_score, take top --topk (default 300)
  3. Optional: remove very-confident destroyed
                 (CNN pred==destroyed AND max_softmax > --dest_thresh)
  4. Optional: remove very-confident no-damage
                 (CNN pred==no-damage AND max_softmax > --nodamage_thresh
                  AND triage_score < --low_triage_thresh)
  5. Keep at most --max_ids buildings (default 200)

Output:
  --out_csv    CSV with columns: building_id, tile_id, uid, minor_score,
               major_score, triage_score (+ CNN columns if available)
  --out_ids    Plain text file: one building_id per line (for run_vlm.py)

Usage:
  # Ensemble only (no CNN filter)
  python scripts/select_vlm_candidates.py \
      --minor_scores models/binary_ensemble/minor_cv/val_scores.csv \
      --major_scores models/binary_ensemble/major_cv/val_scores.csv \
      --out_csv reports/vlm/candidates.csv \
      --out_ids reports/vlm/candidates_ids.txt

  # With CNN confidence filter
  python scripts/select_vlm_candidates.py \
      --minor_scores models/binary_ensemble/minor_cv/val_scores.csv \
      --major_scores models/binary_ensemble/major_cv/val_scores.csv \
      --cnn_preds    reports/vlm/cnn_preds_val.csv \
      --out_csv      reports/vlm/candidates.csv \
      --out_ids      reports/vlm/candidates_ids.txt
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


DAMAGE_CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]


def load_scores(csv_path: str) -> dict[tuple[str, str], float]:
    """Return {(tile_id, uid): calib_score} from a val_scores.csv."""
    store: dict[tuple[str, str], float] = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            store[(row["tile_id"], row["uid"])] = float(row["calib_score"])
    return store


def load_cnn_preds(csv_path: str) -> dict[tuple[str, str], dict]:
    """Return {(tile_id, uid): {pred_class, max_softmax, prob_*}} from CNN preds CSV."""
    store: dict[tuple[str, str], dict] = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["tile_id"], row["uid"])
            store[key] = {
                "pred_class":   row.get("pred_class", ""),
                "max_softmax":  float(row.get("max_softmax", 0.0)),
                "prob_no":      float(row.get("prob_no",    0.0)),
                "prob_minor":   float(row.get("prob_minor", 0.0)),
                "prob_major":   float(row.get("prob_major", 0.0)),
                "prob_dest":    float(row.get("prob_dest",  0.0)),
            }
    return store


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate VLM candidate shortlist from binary ensemble scores"
    )
    p.add_argument("--minor_scores", required=True,
                   help="val_scores.csv from minor ensemble (e.g. models/binary_ensemble/minor_cv/val_scores.csv)")
    p.add_argument("--major_scores", required=True,
                   help="val_scores.csv from major ensemble (e.g. models/binary_ensemble/major_cv/val_scores.csv)")
    p.add_argument("--cnn_preds",    default=None,
                   help="Optional CNN preds CSV with pred_class, max_softmax, prob_* columns")
    p.add_argument("--topk",         type=int, default=300,
                   help="Take top-K buildings by triage_score before filtering (default 300)")
    p.add_argument("--max_ids",      type=int, default=200,
                   help="Final shortlist size after filtering (default 200)")
    p.add_argument("--dest_thresh",  type=float, default=0.90,
                   help="Remove if CNN pred=destroyed AND max_softmax > thresh (default 0.90)")
    p.add_argument("--nodamage_thresh", type=float, default=0.95,
                   help="Remove if CNN pred=no-damage AND max_softmax > thresh "
                        "AND triage_score < --low_triage_thresh (default 0.95)")
    p.add_argument("--low_triage_thresh", type=float, default=0.05,
                   help="Triage score below which a no-damage prediction is removed (default 0.05)")
    p.add_argument("--out_csv", default="reports/vlm/candidates.csv",
                   help="Output CSV with scores and metadata")
    p.add_argument("--out_ids", default="reports/vlm/candidates_ids.txt",
                   help="Output text file: one building_id per line")
    args = p.parse_args()

    # ------------------------------------------------------------------
    # Load scores
    # ------------------------------------------------------------------
    print(f"Loading minor scores from {args.minor_scores} ...")
    minor_scores = load_scores(args.minor_scores)
    print(f"  {len(minor_scores)} buildings")

    print(f"Loading major scores from {args.major_scores} ...")
    major_scores = load_scores(args.major_scores)
    print(f"  {len(major_scores)} buildings")

    cnn_preds: dict = {}
    if args.cnn_preds:
        print(f"Loading CNN preds from {args.cnn_preds} ...")
        cnn_preds = load_cnn_preds(args.cnn_preds)
        print(f"  {len(cnn_preds)} buildings")

    # ------------------------------------------------------------------
    # Build combined record per building
    # ------------------------------------------------------------------
    all_keys = set(minor_scores) | set(major_scores)
    records = []
    for (tile_id, uid) in all_keys:
        minor_s  = minor_scores.get((tile_id, uid), 0.0)
        major_s  = major_scores.get((tile_id, uid), 0.0)
        triage_s = max(minor_s, major_s)
        building_id = f"{tile_id}:{uid}"
        rec = {
            "building_id":   building_id,
            "tile_id":       tile_id,
            "uid":           uid,
            "minor_score":   minor_s,
            "major_score":   major_s,
            "triage_score":  triage_s,
        }
        if cnn_preds:
            cnn = cnn_preds.get((tile_id, uid), {})
            rec["pred_class"]  = cnn.get("pred_class", "")
            rec["max_softmax"] = cnn.get("max_softmax", 0.0)
            rec["prob_no"]     = cnn.get("prob_no",    0.0)
            rec["prob_minor"]  = cnn.get("prob_minor", 0.0)
            rec["prob_major"]  = cnn.get("prob_major", 0.0)
            rec["prob_dest"]   = cnn.get("prob_dest",  0.0)
        records.append(rec)

    # ------------------------------------------------------------------
    # Step 1: top-K by triage score
    # ------------------------------------------------------------------
    records.sort(key=lambda r: r["triage_score"], reverse=True)
    candidates = records[: args.topk]
    print(f"\nStep 1 — top-{args.topk} by triage_score: {len(candidates)} buildings")
    if candidates:
        print(f"  triage range: {candidates[-1]['triage_score']:.4f} – {candidates[0]['triage_score']:.4f}")

    # ------------------------------------------------------------------
    # Step 2: optional CNN-based filtering (requires --cnn_preds)
    # ------------------------------------------------------------------
    if cnn_preds:
        before = len(candidates)
        filtered = []
        n_rm_dest   = 0
        n_rm_nodmg  = 0
        for rec in candidates:
            pc  = rec.get("pred_class", "")
            ms  = rec.get("max_softmax", 0.0)
            ts  = rec["triage_score"]

            # Remove very-confident destroyed (CNN already handles this well)
            if pc == "destroyed" and ms > args.dest_thresh:
                n_rm_dest += 1
                continue

            # Remove very-confident no-damage with low ensemble signal
            if pc == "no-damage" and ms > args.nodamage_thresh and ts < args.low_triage_thresh:
                n_rm_nodmg += 1
                continue

            filtered.append(rec)

        candidates = filtered
        print(f"Step 2 — CNN filter: removed {n_rm_dest} destroyed + {n_rm_nodmg} no-damage  "
              f"→ {len(candidates)} remaining (was {before})")
    else:
        print("Step 2 — CNN filter: skipped (no --cnn_preds provided)")

    # ------------------------------------------------------------------
    # Step 3: cap to max_ids (keep highest triage scores)
    # ------------------------------------------------------------------
    candidates = candidates[: args.max_ids]
    print(f"Step 3 — cap to max_ids={args.max_ids}: {len(candidates)} final candidates")

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    out_csv  = Path(args.out_csv)
    out_ids  = Path(args.out_ids)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_ids.parent.mkdir(parents=True, exist_ok=True)

    base_cols = ["building_id", "tile_id", "uid",
                 "minor_score", "major_score", "triage_score"]
    cnn_cols  = ["pred_class", "max_softmax",
                 "prob_no", "prob_minor", "prob_major", "prob_dest"] if cnn_preds else []
    all_cols  = base_cols + cnn_cols

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(candidates)
    print(f"\nWrote {len(candidates)} rows -> {out_csv}")

    with open(out_ids, "w", encoding="utf-8") as f:
        for rec in candidates:
            f.write(rec["building_id"] + "\n")
    print(f"Wrote {len(candidates)} IDs  -> {out_ids}")

    # Quick stats
    if cnn_preds:
        class_counts: dict[str, int] = {}
        for rec in candidates:
            c = rec.get("pred_class", "unknown")
            class_counts[c] = class_counts.get(c, 0) + 1
        print("\nCNN pred distribution in shortlist:")
        for c, n in sorted(class_counts.items(), key=lambda x: -x[1]):
            print(f"  {c:<18}: {n}")


if __name__ == "__main__":
    main()
