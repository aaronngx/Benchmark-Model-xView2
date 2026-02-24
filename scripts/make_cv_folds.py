#!/usr/bin/env python3
"""
Create a stratified tile-level k-fold assignment for cross-validation.

Tiles are grouped so each fold receives a balanced share of:
  - minor-damage buildings  (rarest class — highest priority)
  - major-damage buildings
  - destroyed buildings
  - total tiles

Algorithm: greedy stratified assignment
  1. Compute tile-level label counts.
  2. Score each tile by rarity weight (rare-content tiles first).
  3. Assign each tile to the fold with minimum current imbalance vs targets.

Usage:
  python scripts/make_cv_folds.py \\
    --index_csv data/processed/index.csv \\
    --k 5 \\
    --seed 42 \\
    --out_path data/processed/cv_folds_k5_seed42.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

RARITY_WEIGHTS = {"minor-damage": 10, "major-damage": 8, "destroyed": 2}


def build_tile_stats(index_csv: str, crops_dir: str) -> dict[str, dict]:
    """Return {tile_id: {minor, major, destroyed, total}} from crop records."""
    from disaster_bench.data.dataset import build_crop_records

    records = build_crop_records(index_csv, crops_dir)
    stats: dict[str, dict] = defaultdict(lambda: {"minor": 0, "major": 0,
                                                   "destroyed": 0, "total": 0})
    for r in records:
        t = r["tile_id"]
        stats[t]["total"] += 1
        if r["label"] == "minor-damage":
            stats[t]["minor"] += 1
        elif r["label"] == "major-damage":
            stats[t]["major"] += 1
        elif r["label"] == "destroyed":
            stats[t]["destroyed"] += 1
    return dict(stats)


def greedy_assign(tile_stats: dict[str, dict], k: int, seed: int) -> dict[str, int]:
    """
    Greedily assign each tile to a fold, prioritising rare-content tiles first.
    Returns {tile_id: fold_id}.
    """
    # Sort tiles: high rarity-score first; break ties by tile_id for determinism
    def rarity_score(tile):
        s = tile_stats[tile]
        return 10 * s["minor"] + 8 * s["major"] + 2 * s["destroyed"]

    tiles = sorted(tile_stats.keys(), key=lambda t: (-rarity_score(t), t))

    # Shuffle tiles with same score using seeded RNG so deterministic
    rng = random.Random(seed)
    # Stable-shuffle within equal-score groups
    from itertools import groupby
    grouped = []
    for _, grp in groupby(tiles, key=rarity_score):
        grp_list = list(grp)
        rng.shuffle(grp_list)
        grouped.extend(grp_list)
    tiles = grouped

    # Fold accumulators
    fold_counts = [{"minor": 0, "major": 0, "destroyed": 0, "total": 0}
                   for _ in range(k)]

    # Targets (equal share)
    n_tiles = len(tiles)
    total_minor    = sum(s["minor"]    for s in tile_stats.values())
    total_major    = sum(s["major"]    for s in tile_stats.values())
    total_destroyed = sum(s["destroyed"] for s in tile_stats.values())

    target_minor    = total_minor    / k
    target_major    = total_major    / k
    target_destroyed = total_destroyed / k
    target_tiles    = n_tiles        / k

    assignment: dict[str, int] = {}

    for tile in tiles:
        s = tile_stats[tile]
        best_fold = -1
        best_cost = float("inf")

        for f in range(k):
            fc = fold_counts[f]
            # Cost = sum of squared deviations from targets after adding this tile
            cost = (
                ((fc["minor"]    + s["minor"]    - target_minor)    ** 2) +
                ((fc["major"]    + s["major"]    - target_major)    ** 2) +
                ((fc["destroyed"] + s["destroyed"] - target_destroyed) ** 2) +
                ((fc["total"]    + s["total"]    - target_tiles)    ** 2)
            )
            if cost < best_cost:
                best_cost = cost
                best_fold = f

        assignment[tile] = best_fold
        fold_counts[best_fold]["minor"]    += s["minor"]
        fold_counts[best_fold]["major"]    += s["major"]
        fold_counts[best_fold]["destroyed"] += s["destroyed"]
        fold_counts[best_fold]["total"]    += s["total"]

    return assignment


def print_fold_summary(tile_stats: dict[str, dict],
                       assignment: dict[str, int], k: int) -> None:
    """Print per-fold summary table and warn on empty minor/major folds."""
    fold_data = [{"tiles": 0, "minor_tiles": 0, "minor_bldgs": 0,
                  "major_tiles": 0, "major_bldgs": 0,
                  "dest_tiles": 0,  "dest_bldgs": 0}
                 for _ in range(k)]

    for tile, fold in assignment.items():
        s = tile_stats[tile]
        fd = fold_data[fold]
        fd["tiles"] += 1
        if s["minor"] > 0:
            fd["minor_tiles"] += 1
            fd["minor_bldgs"] += s["minor"]
        if s["major"] > 0:
            fd["major_tiles"] += 1
            fd["major_bldgs"] += s["major"]
        if s["destroyed"] > 0:
            fd["dest_tiles"] += 1
            fd["dest_bldgs"] += s["destroyed"]

    hdr = (f"{'Fold':>5}  {'Tiles':>6}  "
           f"{'MinTiles':>9}  {'MinBldg':>8}  "
           f"{'MajTiles':>9}  {'MajBldg':>8}  "
           f"{'DstTiles':>9}  {'DstBldg':>8}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for f, fd in enumerate(fold_data):
        print(f"  {f:>3}  {fd['tiles']:>6}  "
              f"{fd['minor_tiles']:>9}  {fd['minor_bldgs']:>8}  "
              f"{fd['major_tiles']:>9}  {fd['major_bldgs']:>8}  "
              f"{fd['dest_tiles']:>9}  {fd['dest_bldgs']:>8}")

    print()
    for f, fd in enumerate(fold_data):
        if fd["minor_tiles"] == 0:
            print(f"WARNING: fold {f} has 0 minor-damage tiles")
        if fd["major_tiles"] == 0:
            print(f"WARNING: fold {f} has 0 major-damage tiles")


def main() -> None:
    p = argparse.ArgumentParser(description="Create stratified k-fold tile assignment")
    p.add_argument("--index_csv",  default="data/processed/index.csv")
    p.add_argument("--crops_dir",  default="data/processed/crops_oracle")
    p.add_argument("--k",          type=int, default=5)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--out_path",   default="data/processed/cv_folds_k5_seed42.json")
    args = p.parse_args()

    print(f"Building tile stats from {args.index_csv} ...")
    tile_stats = build_tile_stats(args.index_csv, args.crops_dir)
    print(f"  Total tiles: {len(tile_stats)}")
    print(f"  Tiles with minor: {sum(1 for s in tile_stats.values() if s['minor'] > 0)}")
    print(f"  Tiles with major: {sum(1 for s in tile_stats.values() if s['major'] > 0)}")

    print(f"\nAssigning {len(tile_stats)} tiles to {args.k} folds (seed={args.seed}) ...")
    assignment = greedy_assign(tile_stats, args.k, args.seed)

    print_fold_summary(tile_stats, assignment, args.k)

    out = {"k": args.k, "seed": args.seed, "tile_to_fold": assignment}
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {args.out_path}")


if __name__ == "__main__":
    main()
