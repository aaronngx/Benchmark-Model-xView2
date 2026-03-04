#!/usr/bin/env python3
"""
Build a stratified evaluation sample for VLM benchmarking.

Selects buildings balanced across damage classes:
  - minor and major: ALL available (too few to subsample)
  - no-damage and destroyed: up to --n_per_class (default 50)

Saves: data/processed/vlm_eval_sample.csv
Columns: tile_id, uid, gt_label, pre_path, post_path, label_json_path

Usage:
  python scripts/make_vlm_sample.py
  python scripts/make_vlm_sample.py --n_per_class 100
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from disaster_bench.data.io import read_index_csv, load_label_json, get_buildings_from_label


def main() -> None:
    p = argparse.ArgumentParser(description="Build stratified VLM eval sample")
    p.add_argument("--index_csv",   default="data/processed/index.csv")
    p.add_argument("--n_per_class", type=int, default=50,
                   help="Max buildings for no-damage and destroyed (default 50)")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--out_csv",     default="data/processed/vlm_eval_sample.csv")
    args = p.parse_args()

    rng = random.Random(args.seed)
    rows = read_index_csv(args.index_csv)

    # Collect all buildings with their labels
    by_class: dict[str, list[dict]] = {
        "no-damage": [], "minor-damage": [], "major-damage": [], "destroyed": []
    }

    for row in rows:
        label_path = row.get("label_json_path", "")
        if not label_path:
            continue
        try:
            label_data = load_label_json(label_path)
            buildings  = get_buildings_from_label(label_data, use_xy=True)
        except Exception:
            continue

        for b in buildings:
            subtype = b.get("subtype", "no-damage")
            if subtype not in by_class:
                continue
            by_class[subtype].append({
                "tile_id":          row["tile_id"],
                "uid":              b["uid"],
                "gt_label":         subtype,
                "pre_path":         row.get("pre_path", ""),
                "post_path":        row.get("post_path", ""),
                "label_json_path":  label_path,
            })

    # Sample
    sample = []
    for cls, records in by_class.items():
        if cls in ("minor-damage", "major-damage"):
            chosen = records          # take all rare class buildings
        else:
            rng.shuffle(records)
            chosen = records[:args.n_per_class]
        sample.extend(chosen)
        print(f"  {cls:<15}: {len(records):4d} total  ->  {len(chosen):3d} sampled")

    rng.shuffle(sample)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["tile_id", "uid", "gt_label", "pre_path", "post_path", "label_json_path"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(sample)

    print(f"\nTotal: {len(sample)} buildings saved -> {out_path}")


if __name__ == "__main__":
    main()
