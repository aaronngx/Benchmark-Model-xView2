"""
Track 2A — Unsupervised numeric baseline: change signals within footprint -> thresholds -> damage.
Uses only pre/post images and GT footprints (no GT damage at inference).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from disaster_bench.data.io import (
    get_buildings_from_label,
    get_label_canvas_size,
    load_image,
    load_label_json,
    read_index_csv,
)
from disaster_bench.data.polygons import parse_and_scale_building, scale_factors
from disaster_bench.data.rasterize import rasterize_polygon
from disaster_bench.geometry.features import mask_area
from disaster_bench.geometry.thresholds import thresholds_to_damage


def run_track2a(
    index_csv: str | Path,
    run_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Compute abs-diff change within each building footprint; map to damage via thresholds.
    """
    config = config or {}
    rows = read_index_csv(index_csv)
    all_predictions: list[dict[str, Any]] = []
    for row in rows:
        tile_id = row["tile_id"]
        pre_path = row.get("pre_path")
        post_path = row.get("post_path")
        label_path = row.get("label_json_path")
        if not label_path:
            continue
        try:
            label_data = load_label_json(label_path)
            buildings = get_buildings_from_label(label_data, use_xy=True)
        except Exception:
            continue
        json_w, json_h = get_label_canvas_size(label_data)
        if json_w <= 0:
            json_w, json_h = 1024, 1024
        try:
            pre_img = load_image(pre_path) if pre_path else None
            post_img = load_image(post_path) if post_path else None
        except Exception:
            pre_img = post_img = None
        if post_img is None:
            for b in buildings:
                all_predictions.append(_row(tile_id, b["uid"], "no-damage", 0.0, "2A"))
            continue
        img_h, img_w = post_img.shape[:2]
        sx, sy = scale_factors(img_w, img_h, json_w, json_h)
        # Simple change: abs difference (grayscale)
        if pre_img is not None and pre_img.shape[:2] == (img_h, img_w):
            diff = np.abs(
                post_img.astype(np.float32).mean(axis=2)
                - pre_img.astype(np.float32).mean(axis=2),
            )
            diff_binary = (diff > config.get("diff_threshold", 25.0)).astype(np.uint8)
        else:
            diff_binary = np.zeros((img_h, img_w), dtype=np.uint8)
        for b in buildings:
            uid = b["uid"]
            poly, _ = parse_and_scale_building(b["wkt"], sx, sy)
            if poly is None:
                all_predictions.append(_row(tile_id, uid, "no-damage", 0.0, "2A"))
                continue
            mask = rasterize_polygon(poly, (img_h, img_w))
            building_pixels = mask_area(mask)
            if building_pixels == 0:
                all_predictions.append(_row(tile_id, uid, "no-damage", 0.0, "2A"))
                continue
            changed_in_building = np.logical_and(mask > 0, diff_binary > 0).sum()
            pct = 100.0 * float(changed_in_building) / float(building_pixels)
            pred = thresholds_to_damage(pct_changed=pct, pct_damaged=None)
            all_predictions.append(_row(tile_id, uid, pred, pct / 100.0, "2A"))
    return all_predictions


def _row(
    tile_id: str,
    uid: str,
    pred_damage: str,
    conf: float,
    track: str,
) -> dict[str, Any]:
    return {
        "tile_id": tile_id,
        "pred_instance_id": uid,
        "matched_gt_uid": None,
        "iou": None,
        "pred_damage": pred_damage,
        "pred_conf": round(conf, 4),
        "track": track,
        "notes": "",
    }


def run_track2a_and_save(
    index_csv: str | Path,
    run_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> None:
    from disaster_bench.eval.report import write_predictions_csv, write_metrics_json
    preds = run_track2a(index_csv, run_dir, config)
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    write_predictions_csv(preds, Path(run_dir) / "predictions.csv")
    write_metrics_json({
        "macro_f1": None,
        "per_class_f1": {},
        "coverage": None,
        "note": "Run eval-run to compute metrics",
    }, Path(run_dir) / "metrics.json")
