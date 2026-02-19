"""
Track 1 — End-to-End ML: predicted footprints -> crops -> damage classifier -> per-building label.
This runner uses GT footprints as placeholder when no footprint/damage model is loaded;
produces standardized predictions.csv.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from disaster_bench.data.io import (
    get_buildings_from_label,
    get_label_canvas_size,
    load_image,
    load_label_json,
    read_index_csv,
)
from disaster_bench.data.polygons import parse_and_scale_building, scale_factors


def run_track1(
    index_csv: str | Path,
    run_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Run Track 1 over index; write predictions to run_dir/predictions.csv and metrics placeholder.
    Uses GT footprints and a placeholder damage prediction (no-damage) when no model is set.
    """
    config = config or {}
    rows = read_index_csv(index_csv)
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
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
        except Exception:
            continue
        buildings = get_buildings_from_label(label_data, use_xy=True)
        json_w, json_h = get_label_canvas_size(label_data)
        if json_w <= 0:
            json_w, json_h = 1024, 1024
        try:
            post_img = load_image(post_path) if post_path else None
        except Exception:
            post_img = None
        img_h, img_w = (post_img.shape[:2] if post_img is not None else (json_h, json_w))
        sx, sy = scale_factors(img_w, img_h, json_w, json_h)
        for b in buildings:
            uid = b["uid"]
            _, bbox = parse_and_scale_building(b["wkt"], sx, sy)
            if bbox is None:
                continue
            # Placeholder: no model -> predict no-damage
            pred_damage = config.get("placeholder_damage", "no-damage")
            all_predictions.append({
                "tile_id": tile_id,
                "pred_instance_id": uid,
                "matched_gt_uid": None,
                "iou": None,
                "pred_damage": pred_damage,
                "pred_conf": 0.5,
                "track": 1,
                "notes": "",
            })
    return all_predictions


def run_track1_and_save(
    index_csv: str | Path,
    run_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> None:
    """Run Track 1 and write predictions.csv + metrics.json to run_dir."""
    from disaster_bench.eval.report import write_predictions_csv, write_metrics_json
    preds = run_track1(index_csv, run_dir, config)
    write_predictions_csv(preds, Path(run_dir) / "predictions.csv")
    write_metrics_json({
        "macro_f1": None,
        "per_class_f1": {},
        "coverage": None,
        "note": "Run eval-run to compute metrics",
    }, Path(run_dir) / "metrics.json")
