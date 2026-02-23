#!/usr/bin/env python3
"""
Side-by-side metrics comparison across all benchmark runs.
Ref §B deliverable: comparison table across tracks.

Usage:
    python scripts/compare_metrics.py [--runs_dir runs] [--json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RUNS_DEFAULT = [
    # Track 1 — oracle bbox, 6-channel CNN variants
    "runs/track1_placeholder",
    "runs/track1_6ch_v1",
    "runs/track1_six_channel",
    "runs/track1_pre_post_diff",
    "runs/track1_siamese",
    "runs/track1_centroid_patch",
    "runs/track1_final",
    # Track 1 — real deployment (predicted footprints)
    "runs/track1_deploy",
    # Track 2 — heuristic baselines
    "runs/track2a_v1",
    "runs/track2a_v2",
    "runs/track2b_v1",
    # Track 3 — hybrid
    "runs/track3_v1",
    "runs/track3_6ch_v1",
    "runs/track3_final",
    # Track 4 — VLM
    "runs/track4_vlm_ungrounded",
    "runs/track4_vlm_grounded",
    "runs/track4_vlm_gemini",
    "runs/track4_vlm_claude",
    "runs/track4_vlm_local",
]

DAMAGE_ABBR = {
    "no-damage":    "no",
    "minor-damage": "min",
    "major-damage": "maj",
    "destroyed":    "dst",
}


def _fmt(v, width=8, decimals=4):
    if v is None:
        return f"{'N/A':>{width}}"
    if isinstance(v, float):
        return f"{v:.{decimals}f}".rjust(width)
    return str(v).rjust(width)


def load_run(run_dir: Path) -> dict | None:
    p = run_dir / "metrics.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def print_table(runs_dirs: list[Path]) -> None:
    col_run  = 32
    col_f1   = 10
    col_cov  = 9
    col_fema = 10
    col_lat  = 10

    hdr = (f"{'Run':<{col_run}}"
           f"{'macro_F1':>{col_f1}}"
           f"{'coverage':>{col_cov}}"
           f"{'FEMA_F1':>{col_fema}}"
           f"{'lat_ms':>{col_lat}}"
           f"  per_class_f1 (no/min/maj/dst)"
           f"  is_dmg_f1  is_dest_f1  sev_mae")
    print(hdr)
    print("-" * len(hdr))

    for rd in runs_dirs:
        m = load_run(rd)
        run_label = rd.name if rd.is_dir() else str(rd)
        if m is None:
            print(f"{run_label:<{col_run}}  (no metrics.json)")
            continue

        macro  = m.get("macro_f1")
        cov    = m.get("coverage")
        fema   = m.get("fema_macro_f1")
        lat    = m.get("avg_latency_ms")
        pcf    = m.get("per_class_f1", {})
        pcf_str = "  ".join(
            f"{DAMAGE_ABBR.get(k, k[:3])}={v:.3f}"
            for k, v in pcf.items()
        )
        is_dmg  = m.get("is_damaged_f1")
        is_dest = m.get("is_destroyed_f1")
        sev_mae = m.get("severity_mae")

        row = (f"{run_label:<{col_run}}"
               f"{_fmt(macro, col_f1)}"
               f"{_fmt(cov,   col_cov)}"
               f"{_fmt(fema,  col_fema)}"
               f"{_fmt(lat,   col_lat)}"
               f"  {pcf_str}")
        if is_dmg is not None:
            row += f"  {is_dmg:.3f}  {is_dest:.3f}  {sev_mae:.3f}"
        print(row)

    print()


def print_confusion_tables(runs_dirs: list[Path]) -> None:
    """Print confusion matrices for runs that have them."""
    from disaster_bench.eval.report import print_confusion_matrix
    for rd in runs_dirs:
        m = load_run(rd)
        if m is None or "confusion_matrix" not in m:
            continue
        print(f"\n  [{rd.name}] Confusion Matrix (rows=GT, cols=Pred):")
        print_confusion_matrix(m["confusion_matrix"])


def main() -> None:
    p = argparse.ArgumentParser(description="Compare benchmark run metrics")
    p.add_argument("--runs_dir", default="runs",
                   help="Root directory containing run subdirectories")
    p.add_argument("--runs", nargs="*",
                   help="Explicit run directory names (relative to --runs_dir)")
    p.add_argument("--json", action="store_true", help="Also dump JSON comparison")
    p.add_argument("--confusion", action="store_true", help="Print confusion matrices")
    args = p.parse_args()

    runs_root = Path(args.runs_dir)
    if args.runs:
        run_dirs = [runs_root / r for r in args.runs]
    else:
        # Auto-discover: all subdirs with metrics.json, PLUS the default list
        discovered = sorted(runs_root.glob("*/metrics.json"))
        discovered_dirs = {p.parent for p in discovered}
        default_dirs    = {runs_root / Path(r).name for r in RUNS_DEFAULT}
        all_dirs = sorted(discovered_dirs | default_dirs,
                          key=lambda d: d.name)
        run_dirs = all_dirs

    print_table(run_dirs)

    if args.confusion:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        print_confusion_tables(run_dirs)

    if args.json:
        out = {}
        for rd in run_dirs:
            m = load_run(rd)
            if m:
                out[rd.name] = m
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
