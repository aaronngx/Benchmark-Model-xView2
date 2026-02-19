"""
Track 2B — Damage-map aggregation: pixel-wise change/severity -> aggregate within footprint -> thresholds.
Placeholder: use same change signal as 2A until a damage map model is plugged in.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from disaster_bench.pipelines.track2a import run_track2a, run_track2a_and_save


def run_track2b(
    index_csv: str | Path,
    run_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Same as 2A for now; can plug in learned damage map and %% damaged pixels."""
    config = config or {}
    # Delegate to 2a logic but set track=2B
    preds = run_track2a(index_csv, run_dir, config)
    for p in preds:
        p["track"] = "2B"
    return preds


def run_track2b_and_save(
    index_csv: str | Path,
    run_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> None:
    from disaster_bench.eval.report import write_predictions_csv, write_metrics_json
    preds = run_track2b(index_csv, run_dir, config)
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    write_predictions_csv(preds, Path(run_dir) / "predictions.csv")
    write_metrics_json({
        "macro_f1": None,
        "per_class_f1": {},
        "coverage": None,
        "note": "Run eval-run to compute metrics",
    }, Path(run_dir) / "metrics.json")
