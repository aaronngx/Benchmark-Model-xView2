#!/usr/bin/env python3
"""
Compute and cache 4 pixel-level change features per building.

Features (computed at 128px, float32):
  mean_abs_diff  — mean(|post - pre|) across all pixels and channels
  pct_changed    — fraction of pixels where |post - pre| > 0.08 (fixed tau in [0,1])
  ssim           — grayscale SSIM (skimage.metrics.structural_similarity)
  edge_diff      — mean |Canny(post_gray) - Canny(pre_gray)|

Source priority:
  1. pre_raw.png / post_raw.png   (no GT outline — default, clean)
  2. pre_bbox.png / post_bbox.png  (legacy outlined — only with --allow_outlined_fallback 1)

Saves:
  data/processed/change_feats.npy     — (N, 4) float32
  data/processed/change_feats_ids.npy — (N,)   building_id strings  "{tile_id}:{uid}"

Usage:
  python scripts/extract_change_features.py
  python scripts/extract_change_features.py --mask_region 1   # compute only on building pixels
  python scripts/extract_change_features.py --allow_outlined_fallback 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

SIZE = 128          # resize to match training resolution
DIFF_TAU = 0.08    # fixed threshold for pct_changed


def _load_gray(path: Path, size: int) -> np.ndarray:
    """Load image, resize, convert to float32 [0,1] grayscale."""
    from PIL import Image
    with Image.open(path) as im:
        arr = np.array(im.convert("RGB").resize((size, size), Image.BILINEAR),
                       dtype=np.float32) / 255.0
    return arr.mean(axis=2)   # (H, W)


def _load_rgb(path: Path, size: int) -> np.ndarray:
    """Load image, resize to (H, W, 3) float32 [0,1]."""
    from PIL import Image
    with Image.open(path) as im:
        arr = np.array(im.convert("RGB").resize((size, size), Image.BILINEAR),
                       dtype=np.float32) / 255.0
    return arr   # (H, W, 3)


def _canny_edges(gray: np.ndarray) -> np.ndarray:
    """Return binary Canny edge map (0.0/1.0) or magnitude, float32."""
    try:
        from skimage.feature import canny
        return canny(gray.astype(float), sigma=1.0).astype(np.float32)
    except ImportError:
        # Fallback: simple Sobel gradient magnitude
        from scipy.ndimage import sobel
        sx = sobel(gray, axis=0)
        sy = sobel(gray, axis=1)
        mag = np.hypot(sx, sy)
        return (mag / (mag.max() + 1e-8)).astype(np.float32)


def compute_change_features(
    pre_path: Path,
    post_path: Path,
    size: int = SIZE,
    mask_path: Path | None = None,
) -> np.ndarray:
    """
    Returns (4,) float32: [mean_abs_diff, pct_changed, ssim, edge_diff].
    If mask_path is given, restricts computation to building pixels (mask > 0).
    """
    pre_rgb  = _load_rgb(pre_path,  size)   # (H,W,3)
    post_rgb = _load_rgb(post_path, size)

    if mask_path is not None and mask_path.exists():
        from PIL import Image
        with Image.open(mask_path) as im:
            mask_arr = np.array(im.convert("RGB").resize((size, size), Image.NEAREST),
                                dtype=np.float32)
        mask = (mask_arr.max(axis=2) > 0).astype(np.float32)  # (H,W) binary
    else:
        mask = None

    diff = np.abs(post_rgb - pre_rgb)   # (H,W,3)

    if mask is not None:
        m3 = mask[:, :, np.newaxis]
        n_pix = max(float(mask.sum()), 1.0)
        mean_abs_diff = float((diff * m3).sum()) / (n_pix * 3)
        pct_changed   = float(((diff.mean(axis=2) > DIFF_TAU) * mask).sum()) / n_pix
    else:
        mean_abs_diff = float(diff.mean())
        pct_changed   = float((diff.mean(axis=2) > DIFF_TAU).mean())

    # SSIM on grayscale
    pre_gray  = pre_rgb.mean(axis=2)
    post_gray = post_rgb.mean(axis=2)
    try:
        from skimage.metrics import structural_similarity
        ssim_val = float(structural_similarity(
            pre_gray, post_gray, data_range=1.0
        ))
    except ImportError:
        # Fallback: normalized cross-correlation
        p = pre_gray - pre_gray.mean()
        q = post_gray - post_gray.mean()
        denom = np.sqrt((p**2).sum() * (q**2).sum()) + 1e-8
        ssim_val = float((p * q).sum() / denom)

    # Edge difference
    pre_edges  = _canny_edges(pre_gray)
    post_edges = _canny_edges(post_gray)
    if mask is not None:
        edge_diff = float((np.abs(post_edges - pre_edges) * mask).sum()) / max(float(mask.sum()), 1.0)
    else:
        edge_diff = float(np.abs(post_edges - pre_edges).mean())

    return np.array([mean_abs_diff, pct_changed, ssim_val, edge_diff], dtype=np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description="Cache per-building change features")
    p.add_argument("--index_csv",  default="data/processed/index.csv")
    p.add_argument("--crops_dir",  default="data/processed/crops_oracle")
    p.add_argument("--buildings_csv", default="data/processed/buildings_v2.csv")
    p.add_argument("--out_feats",  default="data/processed/change_feats.npy")
    p.add_argument("--out_ids",    default="data/processed/change_feats_ids.npy")
    p.add_argument("--size",       type=int, default=SIZE)
    p.add_argument("--mask_region", type=int, choices=[0, 1], default=0,
                   help="Restrict features to building pixels via pre_masked.png (default 0)")
    p.add_argument("--allow_outlined_fallback", type=int, choices=[0, 1], default=0,
                   help="Fall back to pre_bbox.png if pre_raw.png is missing (default 0=strict)")
    args = p.parse_args()

    from disaster_bench.data.dataset import build_crop_records

    # Use buildings_v2.csv ordering if it exists (deterministic, matches NPZ order)
    buildings_csv = Path(args.buildings_csv)
    if buildings_csv.exists():
        import csv
        with open(buildings_csv, encoding="utf-8") as f:
            records = list(csv.DictReader(f))
        print(f"Loaded {len(records)} buildings from {buildings_csv}")
    else:
        print(f"{buildings_csv} not found — falling back to build_crop_records()")
        raw_recs = build_crop_records(args.index_csv, args.crops_dir)
        records = [{"building_id": f"{r['tile_id']}:{r['uid']}",
                    "tile_id": r["tile_id"], "uid": r["uid"],
                    "pre_path": r["pre_path"], "post_path": r["post_path"]}
                   for r in raw_recs]

    crops_dir = Path(args.crops_dir)
    feats_list  = []
    ids_list    = []
    n_raw       = 0
    n_outlined  = 0
    n_missing   = 0

    for i, row in enumerate(records):
        building_id = row.get("building_id") or f"{row['tile_id']}:{row['uid']}"
        uid     = row.get("uid") or building_id.split(":")[-1]
        tile_id = row.get("tile_id") or building_id.rsplit(":", 1)[0]

        uid_dir = crops_dir / tile_id / uid

        pre_raw  = uid_dir / "pre_raw.png"
        post_raw = uid_dir / "post_raw.png"

        if pre_raw.exists() and post_raw.exists():
            pre_p, post_p = pre_raw, post_raw
            n_raw += 1
        elif args.allow_outlined_fallback:
            pre_bbox  = uid_dir / "pre_bbox.png"
            post_bbox = uid_dir / "post_bbox.png"
            if pre_bbox.exists() and post_bbox.exists():
                pre_p, post_p = pre_bbox, post_bbox
                n_outlined += 1
                if n_outlined == 1:
                    print("  WARNING: falling back to pre_bbox.png for some buildings. "
                          "Features may include outline artifact.", file=sys.stderr)
            else:
                n_missing += 1
                feats_list.append(np.zeros(4, dtype=np.float32))
                ids_list.append(building_id)
                continue
        else:
            print(f"\nERROR: pre_raw.png not found for {building_id}.\n"
                  f"  Run: python scripts/make_oracle_crops.py\n"
                  f"  Or use --allow_outlined_fallback 1 to fall back to pre_bbox.png\n"
                  f"  (Note: pre_bbox.png may include a GT polygon outline artifact).",
                  file=sys.stderr)
            sys.exit(1)

        mask_p = (uid_dir / "pre_masked.png") if args.mask_region else None

        try:
            feat = compute_change_features(pre_p, post_p, size=args.size, mask_path=mask_p)
        except Exception as e:
            print(f"  WARNING: failed for {building_id}: {e}", file=sys.stderr)
            feat = np.zeros(4, dtype=np.float32)

        feats_list.append(feat)
        ids_list.append(building_id)

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(records)} done...")

    feats = np.stack(feats_list, axis=0)
    ids   = np.array(ids_list)

    out_feats = Path(args.out_feats)
    out_ids   = Path(args.out_ids)
    out_feats.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_feats, feats)
    np.save(out_ids,   ids)

    print(f"\nDone.")
    print(f"  raw crops used:      {n_raw}")
    print(f"  outlined fallbacks:  {n_outlined}")
    print(f"  missing (zeroed):    {n_missing}")
    print(f"  feature shape:       {feats.shape}")
    print(f"  features:            mean_abs_diff  pct_changed  ssim  edge_diff")
    print(f"  mean values:         {feats.mean(axis=0).round(4)}")
    print(f"  std  values:         {feats.std(axis=0).round(4)}")
    print(f"Saved {out_feats}")
    print(f"Saved {out_ids}")


if __name__ == "__main__":
    main()
