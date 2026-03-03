#!/usr/bin/env python3
"""
Materialise the canonical building-level dataset file: data/processed/buildings_v2.csv

Uses the exact same build_crop_records() filtering logic as train_damage.py so the
resulting CSV is byte-for-byte reproducible with what training sees.

Output columns:
  building_id   — stable string "{tile_id}:{uid}"
  tile_id       — tile identifier (e.g. socal-fire_00000001)
  disaster      — disaster name derived from tile_id
  uid           — per-tile polygon uid from label JSON
  pre_path      — absolute path to pre_bbox.png crop
  post_path     — absolute path to post_bbox.png crop
  label         — damage class string (no-damage / minor-damage / major-damage / destroyed)
  label_idx     — integer label (0/1/2/3)

Usage:
  python scripts/build_buildings_csv.py
  python scripts/build_buildings_csv.py --index_csv data/processed/index.csv \\
                                         --crops_dir data/processed/crops_oracle \\
                                         --out data/processed/buildings_v2.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from disaster_bench.data.dataset import build_crop_records, DAMAGE_CLASSES


MANIFEST_PATH = Path("data/processed/dataset_manifest.json")
DATASET_VERSION = "v2_filtered_unclassified_removed"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_buildings_csv(records: list[dict], out_path: Path) -> None:
    fieldnames = ["building_id", "tile_id", "disaster", "uid",
                  "pre_path", "post_path", "label", "label_idx"]
    # Sort for deterministic order (tile_id then uid)
    records_sorted = sorted(records, key=lambda r: (r["tile_id"], r["uid"]))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records_sorted:
            tile_id = r["tile_id"]
            disaster = tile_id.rsplit("_", 1)[0]
            writer.writerow({
                "building_id": f"{tile_id}:{r['uid']}",
                "tile_id":     tile_id,
                "disaster":    disaster,
                "uid":         r["uid"],
                "pre_path":    r["pre_path"],
                "post_path":   r["post_path"],
                "label":       r["label"],
                "label_idx":   r["label_idx"],
            })


def main() -> None:
    p = argparse.ArgumentParser(description="Build canonical building-level CSV")
    p.add_argument("--index_csv", default="data/processed/index.csv")
    p.add_argument("--crops_dir", default="data/processed/crops_oracle")
    p.add_argument("--out",       default="data/processed/buildings_v2.csv")
    p.add_argument("--manifest",  default=str(MANIFEST_PATH))
    args = p.parse_args()

    out_path      = Path(args.out)
    manifest_path = Path(args.manifest)
    index_path    = Path(args.index_csv)

    # -----------------------------------------------------------------------
    # Build records using exact same logic as train_damage.py
    # -----------------------------------------------------------------------
    print(f"Reading index:  {args.index_csv}")
    print(f"Reading crops:  {args.crops_dir}")
    records = build_crop_records(args.index_csv, args.crops_dir)

    dist = Counter(r["label"] for r in records)
    n_total = len(records)
    n_tiles = len({r["tile_id"] for r in records})
    disasters = sorted({r["tile_id"].rsplit("_", 1)[0] for r in records})

    print(f"\nBuildings found: {n_total}")
    for cls in DAMAGE_CLASSES:
        print(f"  {cls:20s}: {dist.get(cls, 0)}")
    print(f"Tiles: {n_tiles}")
    print(f"Disasters: {disasters}")

    # -----------------------------------------------------------------------
    # Write CSV (sorted for determinism)
    # -----------------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_buildings_csv(records, out_path)
    print(f"\nWrote {out_path}  ({out_path.stat().st_size:,} bytes)")

    # -----------------------------------------------------------------------
    # Compute SHA256 of the CSV
    # -----------------------------------------------------------------------
    buildings_hash = sha256_file(out_path)
    hash_path = out_path.with_suffix(out_path.suffix + ".sha256")
    hash_path.write_text(buildings_hash + "\n", encoding="utf-8")
    print(f"SHA256: {buildings_hash}")
    print(f"Wrote  {hash_path}")

    # -----------------------------------------------------------------------
    # Compute SHA256 of index.csv and cv_folds_json
    # -----------------------------------------------------------------------
    index_hash = sha256_file(index_path)
    print(f"index.csv SHA256: {index_hash}")

    cv_folds_path = Path("data/processed/cv_folds_k5_seed42.json")
    cv_folds_hash = sha256_file(cv_folds_path) if cv_folds_path.exists() else None
    if cv_folds_hash:
        print(f"cv_folds SHA256:  {cv_folds_hash}")

    # Count tiles in index.csv
    with open(index_path, encoding="utf-8") as f:
        n_index_tiles = sum(1 for _ in csv.DictReader(f))

    # -----------------------------------------------------------------------
    # Write / update dataset_manifest.json
    # -----------------------------------------------------------------------
    manifest = {
        "_description": "Canonical dataset lock for xView2 wildfire benchmark. "
                        "Use this to verify reproducibility of benchmark runs.",
        "dataset_version":   DATASET_VERSION,
        "locked_date":       "2026-02-26",
        "index_csv":         str(index_path),
        "index_csv_sha256":  index_hash,
        "buildings_csv":     str(out_path),
        "buildings_csv_sha256": buildings_hash,
        "n_index_tiles":     n_index_tiles,
        "n_tiles_in_scope":  n_tiles,
        "disasters":         disasters,
        "n_buildings_total": n_total,
        "n_buildings_excluded_unclassified": 182,
        "class_distribution": {cls: dist.get(cls, 0) for cls in DAMAGE_CLASSES},
        "cv_folds_json":     str(cv_folds_path) if cv_folds_path.exists() else None,
        "cv_folds_sha256":   cv_folds_hash,
        "note": (
            "v2 = filtered scope: 182 'un-classified' label entries excluded. "
            "Older runs (track2a, track2b, track3_final) used v1_unfiltered "
            "(n_eval=8498). Do not compare directly to v2 runs."
        ),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {manifest_path}")

    # -----------------------------------------------------------------------
    # Final verification
    # -----------------------------------------------------------------------
    check_hash = sha256_file(out_path)
    assert check_hash == buildings_hash, "Hash verification failed!"
    print("\nVerification: OK — hash matches")
    print(f"\nDataset locked as '{DATASET_VERSION}'")
    print(f"  buildings_csv:  {out_path}")
    print(f"  sha256:         {buildings_hash}")
    print(f"  manifest:       {manifest_path}")


if __name__ == "__main__":
    main()
