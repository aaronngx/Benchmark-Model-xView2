#!/usr/bin/env python3
"""
VLM damage classification — crops directly from tile images using JSON pixel boundaries.

For each building in the dataset:
  1. Read WKT polygon from the post-disaster label JSON (pixel coordinates)
  2. Compute bounding box + padding from pixel boundaries
  3. Crop that region from the full pre- and post-disaster tile images
  4. Optionally pre-filter with CNN + binary ensemble (skip likely no-damage buildings)
  5. Send the pre/post crop pair to the VLM (NO ground-truth labels used as input)
  6. Write predictions to CSV for evaluation against ground truth

Pre-filter modes (--prefilter):
  none            — send all buildings to VLM (default; use for fair eval on sample)
  cnn             — skip if CNN prob(no-damage) > threshold (default 0.95)
  ensemble        — skip if both minor and major ensemble scores < threshold (default 0.1)
  cnn+ensemble    — apply both filters in sequence (recommended for full dataset)

Usage:
  # Fair evaluation on stratified sample (no filter)
  python scripts/run_vlm.py --model openai/gpt-4o-mini --sample_csv data/processed/vlm_eval_sample.csv

  # Production mode on full dataset (both filters)
  python scripts/run_vlm.py --model openai/gpt-4o-mini --prefilter cnn+ensemble \
      --cnn_ckpt models/sampler_noweights_s1/best.pt \
      --ensemble_minor_dir models/binary_ensemble/minor_cv \
      --ensemble_major_dir models/binary_ensemble/major_cv
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


def make_masked_crop(
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
    Return a padded crop identical to crop_from_json_bounds but with all
    pixels outside the building polygon blacked out (R=G=B=0).
    Returns None if the polygon is invalid.
    """
    from PIL import Image as PILImage, ImageDraw
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

    crop = tile_img[py1:py2, px1:px2].copy()
    if crop.size == 0:
        return None

    # Rasterize polygon in crop coordinate space (offset by px1, py1)
    h, w = crop.shape[:2]
    coords = [(cx - px1, cy - py1) for cx, cy in poly.exterior.coords]
    mask_img = PILImage.new("L", (w, h), 0)
    ImageDraw.Draw(mask_img).polygon(coords, fill=255)
    mask = np.array(mask_img)

    masked = crop.copy()
    masked[mask == 0] = 0
    return masked


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
    p.add_argument("--index_csv",  default="data/processed/index.csv",
                   help="Full index CSV (used when --sample_csv is not set)")
    p.add_argument("--sample_csv", default=None,
                   help="Stratified sample CSV from make_vlm_sample.py (recommended)")
    p.add_argument("--grounded",   action="store_true",
                   help="Include pixel-level change features in prompt")
    p.add_argument("--max_crops",  type=int, default=None,
                   help="Limit number of buildings (for testing/rate-limit safety)")
    p.add_argument("--pad_fraction", type=float, default=0.25,
                   help="Padding around bbox as fraction of bbox size (default 0.25)")
    p.add_argument("--vlm_size",   type=int, default=256,
                   help="Resize crops to this px before sending to VLM (default 256, 0=natural)")
    p.add_argument("--out_csv",    default=None,
                   help="Output CSV path (default: reports/vlm/predictions_<model>.csv)")
    # Pre-filter arguments
    p.add_argument("--prefilter", default="none",
                   choices=["none", "cnn", "ensemble", "cnn+ensemble"],
                   help="Pre-filter mode to skip likely no-damage buildings before VLM")
    p.add_argument("--cnn_ckpt",  default="models/sampler_noweights_s1/best.pt",
                   help="SixChannelCNN checkpoint for CNN pre-filter")
    p.add_argument("--ensemble_minor_dir", default="models/binary_ensemble/minor_cv",
                   help="Directory with minor val_scores.csv for ensemble pre-filter")
    p.add_argument("--ensemble_major_dir", default="models/binary_ensemble/major_cv",
                   help="Directory with major val_scores.csv for ensemble pre-filter")
    p.add_argument("--cnn_threshold",      type=float, default=0.95,
                   help="CNN filter: skip if P(no-damage) > threshold (default 0.95)")
    p.add_argument("--ensemble_threshold", type=float, default=0.1,
                   help="Ensemble filter: skip if BOTH minor and major score < threshold (default 0.1)")
    p.add_argument("--cnn_dest_thresh", type=float, default=0.85,
                   help="CNN filter: also skip if P(destroyed) > threshold (default 0.85). "
                        "Only active when --prefilter includes 'cnn'.")
    p.add_argument("--masked_crops",   action="store_true",
                   help="Send 4 images to VLM: context crop + building-only masked crop "
                        "(pre and post for each). Requires shapely (already a dependency).")
    p.add_argument("--prompt_mode",    default="4class",
                   choices=["4class", "minor_major"],
                   help="Prompt mode: '4class' = full damage scale (default); "
                        "'minor_major' = restricted minor-vs-major prompt for filtered candidates.")
    p.add_argument("--call_delay_s", type=float, default=2.0,
                   help="Seconds to sleep between VLM API calls to avoid rate limits (default 2.0)")
    p.add_argument("--resume",       action="store_true",
                   help="Resume an interrupted run: skip buildings already written to --out_csv "
                        "and append new rows. Safe to re-run after a rate-limit crash.")
    p.add_argument("--building_ids_path", default=None,
                   help="File of building IDs to process (one 'tile_id:uid' per line, or CSV with "
                        "building_id column). When set, only these buildings are sent to the VLM. "
                        "Use select_vlm_candidates.py to generate this file.")
    args = p.parse_args()

    # Auto-name output CSV from model id if not specified
    if args.out_csv is None:
        model_slug = args.model.replace("/", "_").replace("-", "_")
        args.out_csv = f"reports/vlm/predictions_{model_slug}.csv"

    # ------------------------------------------------------------------
    # Load building ID allowlist (for Stage-2 cascade mode)
    # ------------------------------------------------------------------
    _allowed_ids: set[str] | None = None
    if args.building_ids_path:
        import csv as _csv_ids
        id_path = Path(args.building_ids_path)
        if not id_path.exists():
            print(f"ERROR: --building_ids_path not found: {id_path}", file=__import__("sys").stderr)
            raise SystemExit(1)
        _allowed_ids = set()
        with open(id_path, encoding="utf-8") as _f:
            first_line = _f.readline().strip()
        # Detect CSV (has header with 'building_id' column) vs plain ID list
        if "building_id" in first_line:
            with open(id_path, encoding="utf-8") as _f:
                for _row in _csv_ids.DictReader(_f):
                    _allowed_ids.add(_row["building_id"].strip())
        else:
            # Plain text: one tile_id:uid per line
            with open(id_path, encoding="utf-8") as _f:
                for _line in _f:
                    _line = _line.strip()
                    if _line:
                        _allowed_ids.add(_line)
        print(f"Building ID allowlist: {len(_allowed_ids)} IDs from {id_path}")

    from disaster_bench.models.damage.vlm_wrapper import get_vlm
    vlm = get_vlm(model=args.model)
    print(f"Model:        {args.model}")
    print(f"Grounded:     {args.grounded}")
    print(f"Prefilter:    {args.prefilter}")
    print(f"Prompt mode:  {args.prompt_mode}")
    print(f"Masked crops: {args.masked_crops}")
    print(f"VLM size:     {args.vlm_size}px {'(natural)' if args.vlm_size == 0 else ''}")
    print(f"Max crops:    {args.max_crops or 'unlimited'}")
    print(f"Out CSV:      {args.out_csv}")

    # ---------------------------------------------------------------------------
    # Load pre-filter assets
    # ---------------------------------------------------------------------------
    # Ensemble score lookup: {(tile_id, uid): calib_score}
    ens_minor_scores: dict = {}
    ens_major_scores: dict = {}
    if args.prefilter in ("ensemble", "cnn+ensemble"):
        import csv as _csv2
        for dir_arg, store, label in [
            (args.ensemble_minor_dir, ens_minor_scores, "minor"),
            (args.ensemble_major_dir, ens_major_scores, "major"),
        ]:
            csv_path = Path(dir_arg) / "val_scores.csv"
            if csv_path.exists():
                with open(csv_path, encoding="utf-8") as f:
                    for row in _csv2.DictReader(f):
                        store[(row["tile_id"], row["uid"])] = float(row["calib_score"])
                print(f"  Ensemble {label}: {len(store)} scores loaded from {csv_path}")
            else:
                print(f"  WARN: ensemble {label} val_scores.csv not found at {csv_path}")

    # CNN model for CNN filter
    cnn_model = None
    cnn_device = None
    if args.prefilter in ("cnn", "cnn+ensemble"):
        import torch
        from disaster_bench.models.damage.classifiers import load_classifier
        cnn_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_path = Path(args.cnn_ckpt)
        if not ckpt_path.exists():
            print(f"  WARN: CNN checkpoint not found at {ckpt_path}; CNN filter disabled")
        else:
            cnn_model, _ = load_classifier(str(ckpt_path), device=str(cnn_device))
            cnn_model.eval()
            print(f"  CNN model loaded from {ckpt_path} on {cnn_device}")

    # Load from stratified sample or full index
    if args.sample_csv:
        print(f"Sample:    {args.sample_csv}")
        import csv as _csv
        with open(args.sample_csv, encoding="utf-8") as f:
            sample_rows = list(_csv.DictReader(f))
        # Group by tile so we load each tile image once
        from collections import defaultdict
        tiles: dict = defaultdict(list)
        for r in sample_rows:
            tiles[r["tile_id"]].append(r)
        use_sample = True
    else:
        rows = read_index_csv(args.index_csv)
        use_sample = False
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # All CSV columns — includes new VLM structured-output fields from updated schema
    _CSV_COLS = [
        "tile_id", "uid", "gt_label", "pred_label",
        "reasoning", "key_evidence", "considered_classes",
        "destroyed_gate_passed", "why_not_destroyed", "why_destroyed",
        "major_gate_passed", "why_major", "why_not_major",
        "confidence", "latency_ms", "tokens_used", "model",
        "parse_error", "filter_applied",
    ]

    # ------------------------------------------------------------------
    # Resume: load already-completed building IDs, open CSV for append
    # ------------------------------------------------------------------
    _done_ids: set[str] = set()
    if args.resume and out_path.exists():
        with open(out_path, encoding="utf-8") as _f:
            for _row in csv.DictReader(_f):
                if not _row.get("filter_applied"):
                    _done_ids.add(f"{_row['tile_id']}:{_row['uid']}")
        print(f"Resume mode: {len(_done_ids)} VLM-done buildings loaded, will skip")
        _csv_file = open(out_path, "a", newline="", encoding="utf-8")
        _csv_writer = csv.DictWriter(_csv_file, fieldnames=_CSV_COLS, extrasaction="ignore")
    else:
        _csv_file = open(out_path, "w", newline="", encoding="utf-8")
        _csv_writer = csv.DictWriter(_csv_file, fieldnames=_CSV_COLS, extrasaction="ignore")
        _csv_writer.writeheader()
        _csv_file.flush()

    predictions = []
    n_done = 0
    n_errors = 0
    n_filtered_cnn = 0
    n_filtered_ens = 0
    total_tokens = 0
    t_start = time.perf_counter()

    def _process_building(tile_id, uid, gt_label, wkt,
                          pre_tile, post_tile, img_w, img_h, json_w, json_h):
        nonlocal n_done, n_errors, total_tokens, n_filtered_cnn, n_filtered_ens

        # Stage-2 cascade: skip buildings not in the allowlist
        if _allowed_ids is not None and f"{tile_id}:{uid}" not in _allowed_ids:
            return

        # Resume: skip buildings already written to the output CSV
        if f"{tile_id}:{uid}" in _done_ids:
            return

        pre_crop  = crop_from_json_bounds(
            pre_tile,  wkt, img_w, img_h, json_w, json_h, args.pad_fraction)
        post_crop = crop_from_json_bounds(
            post_tile, wkt, img_w, img_h, json_w, json_h, args.pad_fraction)

        if pre_crop is None or post_crop is None:
            n_errors += 1
            return

        # ------------------------------------------------------------------
        # Pre-filter: CNN (prob_nodamage > threshold → skip to VLM)
        # ------------------------------------------------------------------
        if args.prefilter in ("cnn", "cnn+ensemble") and cnn_model is not None:
            import torch
            pre_128  = resize_crop(pre_crop,  128)
            post_128 = resize_crop(post_crop, 128)
            x = np.concatenate([pre_128, post_128], axis=2).transpose(2, 0, 1).astype(np.float32) / 255.0
            x_t = torch.from_numpy(x).unsqueeze(0).to(cnn_device)
            with torch.no_grad():
                probs = torch.softmax(cnn_model(x_t), dim=1)[0]
            prob_nodamage  = probs[0].item()
            prob_destroyed = probs[3].item()
            # Skip confident no-damage
            if prob_nodamage > args.cnn_threshold:
                n_filtered_cnn += 1
                _row = {
                    "tile_id": tile_id, "uid": uid, "gt_label": gt_label,
                    "pred_label": "no-damage", "reasoning": "",
                    "latency_ms": 0, "tokens_used": 0,
                    "model": args.model, "parse_error": "",
                    "filter_applied": "cnn",
                }
                predictions.append(_row)
                _csv_writer.writerow(_row)
                _csv_file.flush()
                return
            # Skip confident destroyed (CNN handles this better than VLM)
            if prob_destroyed > args.cnn_dest_thresh:
                n_filtered_cnn += 1
                _row = {
                    "tile_id": tile_id, "uid": uid, "gt_label": gt_label,
                    "pred_label": "destroyed", "reasoning": "",
                    "latency_ms": 0, "tokens_used": 0,
                    "model": args.model, "parse_error": "",
                    "filter_applied": "cnn_dest",
                }
                predictions.append(_row)
                _csv_writer.writerow(_row)
                _csv_file.flush()
                return

        # ------------------------------------------------------------------
        # Pre-filter: ensemble (both minor and major score < threshold → skip)
        # ------------------------------------------------------------------
        if args.prefilter in ("ensemble", "cnn+ensemble"):
            bid = (tile_id, uid)
            minor_s = ens_minor_scores.get(bid, 1.0)
            major_s = ens_major_scores.get(bid, 1.0)
            if minor_s < args.ensemble_threshold and major_s < args.ensemble_threshold:
                n_filtered_ens += 1
                _row = {
                    "tile_id": tile_id, "uid": uid, "gt_label": gt_label,
                    "pred_label": "no-damage", "reasoning": "",
                    "latency_ms": 0, "tokens_used": 0,
                    "model": args.model, "parse_error": "",
                    "filter_applied": "ensemble",
                }
                predictions.append(_row)
                _csv_writer.writerow(_row)
                _csv_file.flush()
                return

        # ------------------------------------------------------------------
        # VLM call
        # ------------------------------------------------------------------
        # Generate masked crops (building-only, background blacked out) if requested
        pre_masked_crop  = None
        post_masked_crop = None
        if args.masked_crops:
            pre_masked_crop  = make_masked_crop(pre_tile,  wkt, img_w, img_h,
                                                json_w, json_h, args.pad_fraction)
            post_masked_crop = make_masked_crop(post_tile, wkt, img_w, img_h,
                                                json_w, json_h, args.pad_fraction)

        if args.vlm_size > 0:
            pre_crop  = resize_crop(pre_crop,  args.vlm_size)
            post_crop = resize_crop(post_crop, args.vlm_size)
            if pre_masked_crop is not None:
                pre_masked_crop  = resize_crop(pre_masked_crop,  args.vlm_size)
                post_masked_crop = resize_crop(post_masked_crop, args.vlm_size)

        if args.call_delay_s > 0:
            time.sleep(args.call_delay_s)

        _use_mm = (args.prompt_mode == "minor_major")
        try:
            result = vlm.classify(
                pre_crop, post_crop,
                pre_masked=pre_masked_crop,
                post_masked=post_masked_crop,
                grounded=False,
                geometry=None,
                minor_major=_use_mm,
            )
        except Exception as e:
            print(f"  WARN: VLM call failed for {uid}: {e}")
            n_errors += 1
            return

        total_tokens += result.get("tokens_used", 0)
        n_done += 1

        _row = {
            "tile_id":            tile_id,
            "uid":                uid,
            "gt_label":           gt_label,       # ground truth — eval only, never sent to VLM
            "pred_label":         result.get("damage_level", "no-damage"),
            "reasoning":          result.get("reasoning", "")[:200],
            "key_evidence":       result.get("key_evidence", ""),
            "considered_classes": result.get("considered_classes", ""),
            "destroyed_gate_passed": result.get("destroyed_gate_passed", ""),
            "why_not_destroyed":  result.get("why_not_destroyed", ""),
            "why_destroyed":      result.get("why_destroyed", ""),
            "major_gate_passed":  result.get("major_gate_passed", ""),
            "why_major":          result.get("why_major", ""),
            "why_not_major":      result.get("why_not_major", ""),
            "confidence":         result.get("confidence", ""),
            "latency_ms":         round(result.get("latency_ms", 0), 1),
            "tokens_used":        result.get("tokens_used", 0),
            "model":              args.model,
            "parse_error":        result.get("parse_error", ""),
            "filter_applied":     "",
        }
        predictions.append(_row)
        _csv_writer.writerow(_row)
        _csv_file.flush()

        if n_done % 10 == 0 or n_done == 1:
            print(f"  [{n_done}] {tile_id[-24:]}/{uid[:8]}  "
                  f"pred={result.get('damage_level'):<14} "
                  f"gt={gt_label:<14} "
                  f"lat={result.get('latency_ms', 0):.0f}ms  "
                  f"tokens={total_tokens}")

    if use_sample:
        # Sample mode: iterate tile-grouped to load each tile image once
        for tile_id, bldgs in tiles.items():
            if args.max_crops is not None and n_done >= args.max_crops:
                break

            row0 = bldgs[0]
            try:
                pre_tile  = load_image(row0["pre_path"])
                post_tile = load_image(row0["post_path"])
            except Exception as e:
                print(f"  WARN: could not load tile {tile_id}: {e}")
                continue

            img_h, img_w = post_tile.shape[:2]
            try:
                label_data = load_label_json(row0["label_json_path"])
                json_w, json_h = get_label_canvas_size(label_data)
                if json_w <= 0:
                    json_w, json_h = 1024, 1024
                wkt_by_uid = {b["uid"]: b["wkt"]
                              for b in get_buildings_from_label(label_data, use_xy=True)}
            except Exception as e:
                print(f"  WARN: could not load JSON for {tile_id}: {e}")
                continue

            for b in bldgs:
                if args.max_crops is not None and n_done >= args.max_crops:
                    break
                uid = b["uid"]
                wkt = wkt_by_uid.get(uid)
                if wkt is None:
                    n_errors += 1
                    continue
                _process_building(tile_id, uid, b["gt_label"], wkt,
                                  pre_tile, post_tile, img_w, img_h, json_w, json_h)

    else:
        # Full index mode
        for row in rows:
            if args.max_crops is not None and n_done >= args.max_crops:
                break

            tile_id    = row["tile_id"]
            pre_path   = row.get("pre_path", "")
            post_path  = row.get("post_path", "")
            label_path = row.get("label_json_path", "")
            if not label_path or not pre_path or not post_path:
                continue

            try:
                pre_tile  = load_image(pre_path)
                post_tile = load_image(post_path)
            except Exception as e:
                print(f"  WARN: could not load tile {tile_id}: {e}")
                continue

            img_h, img_w = post_tile.shape[:2]
            try:
                label_data = load_label_json(label_path)
                buildings  = get_buildings_from_label(label_data, use_xy=True)
            except Exception as e:
                print(f"  WARN: could not load JSON for {tile_id}: {e}")
                continue

            json_w, json_h = get_label_canvas_size(label_data)
            if json_w <= 0:
                json_w, json_h = 1024, 1024

            gt_by_uid = {b["uid"]: b.get("subtype", "no-damage") for b in buildings}

            for b in buildings:
                if args.max_crops is not None and n_done >= args.max_crops:
                    break
                _process_building(tile_id, b["uid"], gt_by_uid[b["uid"]], b["wkt"],
                                  pre_tile, post_tile, img_w, img_h, json_w, json_h)

    _csv_file.close()

    elapsed = time.perf_counter() - t_start
    n_total = n_done + n_filtered_cnn + n_filtered_ens + n_errors
    print(f"\nDone: {n_done} VLM calls in {elapsed:.1f}s "
          f"({elapsed/max(n_done,1)*1000:.0f} ms/building) | "
          f"errors={n_errors} | tokens={total_tokens}")
    if args.prefilter != "none":
        print(f"Filtered: cnn={n_filtered_cnn}  ensemble={n_filtered_ens}  "
              f"(total skipped={n_filtered_cnn+n_filtered_ens}/{n_total})")
    print(f"Saved -> {out_path}")

    # Quick accuracy summary (VLM rows only, excluding filtered)
    vlm_rows = [r for r in predictions if not r.get("filter_applied")]
    if vlm_rows:
        correct = sum(1 for r in vlm_rows if r["pred_label"] == r["gt_label"])
        print(f"VLM exact match accuracy: {correct}/{len(vlm_rows)} = {correct/len(vlm_rows):.1%}")


if __name__ == "__main__":
    main()
