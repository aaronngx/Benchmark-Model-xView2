"""
Track 3 — Hybrid: ML/VLM prediction + geometry guardrails; flag when inconsistent.
Uses Track 1-style predictions then applies geometry checks and sets flagged_inconsistent.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from disaster_bench.pipelines.track1 import run_track1
from disaster_bench.pipelines.track2a import run_track2a


def run_track3(
    index_csv: str | Path,
    run_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Get base prediction (Track 1 placeholder or 2A-style change); add geometry evidence;
    set notes to 'flagged_inconsistent' when confidence low or conflict.
    """
    config = config or {}
    # Base predictions: use Track 1 format (per-building) then enrich with geometry
    base = run_track1(index_csv, run_dir, config)
    # Geometry-based damage for comparison (no GT damage used)
    geom_preds = run_track2a(index_csv, run_dir, config)
    geom_by_key: dict[tuple[str, str], str] = {}
    for g in geom_preds:
        geom_by_key[(g["tile_id"], g["pred_instance_id"])] = g["pred_damage"]
    for p in base:
        key = (p["tile_id"], p["pred_instance_id"])
        geom_damage = geom_by_key.get(key, "no-damage")
        ml_damage = p["pred_damage"]
        conf = p.get("pred_conf", 0.5)
        notes = []
        if conf < config.get("min_confidence", 0.3):
            notes.append("low_confidence")
        if ml_damage != geom_damage:
            notes.append("flagged_inconsistent")
        p["notes"] = ";".join(notes) if notes else ""
        p["track"] = 3
    return base


def run_track3_and_save(
    index_csv: str | Path,
    run_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> None:
    from disaster_bench.eval.report import write_predictions_csv, write_metrics_json
    preds = run_track3(index_csv, run_dir, config)
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    write_predictions_csv(preds, Path(run_dir) / "predictions.csv")
    write_metrics_json({
        "macro_f1": None,
        "per_class_f1": {},
        "coverage": None,
        "note": "Run eval-run to compute metrics",
    }, Path(run_dir) / "metrics.json")
