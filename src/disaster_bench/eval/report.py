"""
Write runs/<run_id>/predictions.csv and metrics.json.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_predictions_csv(
    rows: list[dict[str, Any]],
    out_path: str | Path,
    *,
    columns: list[str] | None = None,
) -> None:
    """Write predictions table to CSV."""
    if not rows:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            f.write("tile_id,pred_instance_id,matched_gt_uid,iou,pred_damage,pred_conf,track,notes\n")
        return
    if columns is None:
        columns = [
            "tile_id", "pred_instance_id", "matched_gt_uid", "iou",
            "pred_damage", "pred_conf", "track", "notes",
            "gt_damage",  # optional, added at eval time
        ]
    # Only include columns that appear in rows
    seen = set()
    for r in rows:
        seen.update(r.keys())
    cols = [c for c in columns if c in seen]
    if not cols:
        cols = list(rows[0].keys())
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_metrics_json(metrics: dict[str, Any], out_path: str | Path) -> None:
    """Write metrics dict to JSON."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
