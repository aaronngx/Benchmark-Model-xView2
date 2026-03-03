#!/usr/bin/env python3
"""
VLM damage classification — crops directly from tile images using JSON pixel boundaries.

For each building in the dataset:
  1. Read WKT polygon from the post-disaster label JSON (pixel coordinates)
  2. Compute bounding box + padding from pixel boundaries
  3. Crop that region from the full pre- and post-disaster tile images
  4. Send the pre/post crop pair to the VLM (NO ground-truth labels used as input)
  5. Write predictions to CSV for evaluation against ground truth

Usage:
  python scripts/run_vlm.py --model openai/gpt-4o-mini --max_crops 20
  python scripts/run_vlm.py --model openai/gpt-4o --grounded --max_crops 50
  python scripts/run_vlm.py --model meta/llama-4-scout-17b-16e-instruct --max_crops 20
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from disaster_bench.data.io import (
    read_index_csv,
    load_label_json,
    get_buildings_from_label,
    get_label_canvas_size,
    load_image,
)
from disaster_bench.data.polygons import parse_and_scale_building, scale_factors


# ---------------------------------------------------------------------------
# Crop from tile image using JSON pixel boundaries
# ---------------------------------------------------------------------------

def crop_from_json_bounds(
    tile_img: np.ndarray,
    wkt: str,
    img_w: int,
    img_h: int,
    json_w: int,
    json_h: int,
    pad_fraction: float = 0.25,
    min_pad: int = 8,
) -> np.ndarray | None:
    """
    Crop a building from a full tile image using WKT pixel coordinates from the JSON.
    Returns (H, W, 3) uint8 crop, or None if the polygon is invalid.

    - WKT coordinates are in JSON canvas space; scaled to image pixel space.
    - Padding is applied around the tight bbox (25% of the larger bbox dimension).
    - Labels are NOT used — only the polygon geometry for locating the building.
    """
    sx, sy = scale_factors(img_w, img_h, json_w, json_h)
    poly, bbox = parse_and_scale_building(wkt, sx, sy)
    if poly is None or bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), img_w - 1))
    y1 = max(0, min(int(y1), img_h - 1))
    x2 = max(x1 + 1, min(int(x2), img_w))
    y2 = max(y1 + 1, min(int(y2), img_h))

    bbox_w, bbox_h = x2 - x1, y2 - y1
    pad = max(int(pad_fraction * max(bbox_w, bbox_h)), min_pad)
    px1 = max(0, x1 - pad)
    py1 = max(0, y1 - pad)
    px2 = min(img_w, x2 + pad)
    py2 = min(img_h, y2 + pad)

    crop = tile_img[py1:py2, px1:px2]
    if crop.size == 0:
        return None
    return crop.copy()


def resize_crop(crop: np.ndarray, size: int) -> np.ndarray:
    """Resize (H, W, 3) uint8 to (size, size) with LANCZOS."""
    from PIL import Image as PILImage
    img = PILImage.fromarray(crop)
    img = img.resize((size, size), PILImage.LANCZOS)
    return np.array(img)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="VLM damage classification from JSON pixel boundaries")
    p.add_argument("--model",      default="openai/gpt-4o-mini",
                   help="GitHub Models namespaced model ID (default: openai/gpt-4o-mini)")
    p.add_argument("--index_csv",  default="data/processed/index.csv")
    p.add_argument("--grounded",   action="store_true",
                   help="Include pixel-level change features in prompt")
    p.add_argument("--max_crops",  type=int, default=None,
                   help="Limit number of buildings (for testing/rate-limit safety)")
    p.add_argument("--pad_fraction", type=float, default=0.25,
                   help="Padding around bbox as fraction of bbox size (default 0.25)")
    p.add_argument("--vlm_size",   type=int, default=256,
                   help="Resize crops to this px before sending to VLM (default 256, 0=natural)")
    p.add_argument("--out_csv",    default="reports/vlm/predictions.csv")
    args = p.parse_args()

    from disaster_bench.models.damage.vlm_wrapper import get_vlm
    vlm = get_vlm(model=args.model)
    print(f"Model:     {args.model}")
    print(f"Grounded:  {args.grounded}")
    print(f"VLM size:  {args.vlm_size}px {'(natural)' if args.vlm_size == 0 else ''}")
    print(f"Max crops: {args.max_crops or 'unlimited'}")

    rows = read_index_csv(args.index_csv)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    predictions = []
    n_done = 0
    n_errors = 0
    total_tokens = 0
    t_start = time.perf_counter()

    for row in rows:
        if args.max_crops is not None and n_done >= args.max_crops:
            break

        tile_id    = row["tile_id"]
        pre_path   = row.get("pre_path", "")
        post_path  = row.get("post_path", "")
        label_path = row.get("label_json_path", "")
        if not label_path or not pre_path or not post_path:
            continue

        # Load full tile images
        try:
            pre_tile  = load_image(pre_path)
            post_tile = load_image(post_path)
        except Exception as e:
            print(f"  WARN: could not load tile images for {tile_id}: {e}")
            continue

        img_h, img_w = post_tile.shape[:2]

        # Load JSON — polygon coordinates only, NOT damage labels
        try:
            label_data = load_label_json(label_path)
            buildings  = get_buildings_from_label(label_data, use_xy=True)
        except Exception as e:
            print(f"  WARN: could not load label JSON for {tile_id}: {e}")
            continue

        json_w, json_h = get_label_canvas_size(label_data)
        if json_w <= 0:
            json_w, json_h = 1024, 1024

        # Ground-truth labels for evaluation only — stored but never sent to VLM
        gt_by_uid = {b["uid"]: b.get("subtype", "no-damage") for b in buildings}

        for b in buildings:
            if args.max_crops is not None and n_done >= args.max_crops:
                break

            uid = b["uid"]
            gt_label = gt_by_uid[uid]   # used only for CSV output, never sent to VLM

            # Crop from tile using JSON pixel boundaries
            pre_crop  = crop_from_json_bounds(
                pre_tile,  b["wkt"], img_w, img_h, json_w, json_h, args.pad_fraction)
            post_crop = crop_from_json_bounds(
                post_tile, b["wkt"], img_w, img_h, json_w, json_h, args.pad_fraction)

            if pre_crop is None or post_crop is None:
                n_errors += 1
                continue

            # Resize before sending to VLM
            if args.vlm_size > 0:
                pre_crop  = resize_crop(pre_crop,  args.vlm_size)
                post_crop = resize_crop(post_crop, args.vlm_size)

            # Call VLM — no labels, no ground truth, no mask
            try:
                result = vlm.classify(pre_crop, post_crop, grounded=False, geometry=None)
            except Exception as e:
                print(f"  WARN: VLM call failed for {uid}: {e}")
                n_errors += 1
                continue

            total_tokens += result.get("tokens_used", 0)
            n_done += 1

            predictions.append({
                "tile_id":       tile_id,
                "uid":           uid,
                "gt_label":      gt_label,          # ground truth (for eval only)
                "pred_label":    result.get("damage_level", "no-damage"),
                "reasoning":     result.get("reasoning", "")[:200],
                "latency_ms":    round(result.get("latency_ms", 0), 1),
                "tokens_used":   result.get("tokens_used", 0),
                "model":         args.model,
                "parse_error":   result.get("parse_error", ""),
            })

            if n_done % 10 == 0 or n_done == 1:
                elapsed = time.perf_counter() - t_start
                print(f"  [{n_done}] {tile_id}/{uid[:8]}  "
                      f"pred={result.get('damage_level'):<14} "
                      f"gt={gt_label:<14} "
                      f"lat={result.get('latency_ms', 0):.0f}ms  "
                      f"tokens={total_tokens}")

    # Write CSV
    cols = ["tile_id", "uid", "gt_label", "pred_label",
            "reasoning", "latency_ms", "tokens_used", "model", "parse_error"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(predictions)

    elapsed = time.perf_counter() - t_start
    print(f"\nDone: {n_done} buildings in {elapsed:.1f}s "
          f"({elapsed/max(n_done,1)*1000:.0f} ms/building) | "
          f"errors={n_errors} | tokens={total_tokens}")
    print(f"Saved -> {out_path}")

    # Quick accuracy summary
    if predictions:
        correct = sum(1 for r in predictions if r["pred_label"] == r["gt_label"])
        print(f"Exact match accuracy: {correct}/{n_done} = {correct/n_done:.1%}")


if __name__ == "__main__":
    main()
