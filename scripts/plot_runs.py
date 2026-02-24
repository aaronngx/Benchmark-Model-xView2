#!/usr/bin/env python3
"""
Generate per-epoch benchmark plots comparing baseline, sampler_capped, sampler_noweights.

Outputs (reports/plots/):
  fp_per_1k_no_minor_vs_epoch.png
  fp_per_1k_no_major_vs_epoch.png
  f1_minor_vs_epoch.png
  f1_major_vs_epoch.png
  macro_f1_vs_epoch.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not found. Run: pip install matplotlib", file=sys.stderr)
    sys.exit(1)

RUNS_DIR = Path("runs")
OUT_DIR  = Path("reports/plots")

# Methods and their seeds — colors/labels fixed per method
METHODS = {
    "baseline": {
        "seeds": ["baseline_s1", "baseline_s2", "baseline_s3"],
        "color": "#4878CF",
        "label": "Baseline (norm_invfreq, no sampler)",
    },
    "sampler_capped": {
        "seeds": ["sampler_capped_s1", "sampler_capped_s2", "sampler_capped_s3"],
        "color": "#D65F5F",
        "label": "Sampler + capped_floored [0.25, 5.0]",
    },
    "sampler_noweights": {
        "seeds": ["sampler_noweights_s1", "sampler_noweights_s2", "sampler_noweights_s3"],
        "color": "#6ACC65",
        "label": "Sampler + no weights (weight_mode=none)",
    },
}

PLOTS = [
    ("fp_per_1000_no_minor", "FP per 1000 no-damage (→ minor)",
     "fp_per_1k_no_minor_vs_epoch.png"),
    ("fp_per_1000_no_major", "FP per 1000 no-damage (→ major)",
     "fp_per_1k_no_major_vs_epoch.png"),
    ("f1_minor",   "F1 minor-damage",  "f1_minor_vs_epoch.png"),
    ("f1_major",   "F1 major-damage",  "f1_major_vs_epoch.png"),
    ("macro_f1",   "Macro F1 (4-class)", "macro_f1_vs_epoch.png"),
]


def load_run(run_id: str) -> list[dict]:
    path = RUNS_DIR / run_id / "val_metrics.jsonl"
    if not path.exists():
        print(f"WARNING: {path} not found — skipping", file=sys.stderr)
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return sorted(rows, key=lambda r: r["epoch"])


def make_plot(metric: str, ylabel: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    for method_key, cfg in METHODS.items():
        all_series = []
        max_len = 0
        for run_id in cfg["seeds"]:
            rows = load_run(run_id)
            if not rows:
                continue
            vals = [r.get(metric, float("nan")) for r in rows]
            all_series.append(vals)
            max_len = max(max_len, len(vals))

        if not all_series:
            continue

        # Pad shorter series with nan
        arr = np.full((len(all_series), max_len), np.nan)
        for i, s in enumerate(all_series):
            arr[i, :len(s)] = s

        epochs = np.arange(1, max_len + 1)
        mean = np.nanmean(arr, axis=0)
        std  = np.nanstd(arr, axis=0, ddof=1) if len(all_series) > 1 else np.zeros(max_len)

        ax.plot(epochs, mean, color=cfg["color"], label=cfg["label"], linewidth=2)
        ax.fill_between(epochs, mean - std, mean + std,
                        color=cfg["color"], alpha=0.15)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{ylabel} vs Epoch", fontsize=13)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    for metric, ylabel, filename in PLOTS:
        make_plot(metric, ylabel, filename)
    print("All plots written to", OUT_DIR)


if __name__ == "__main__":
    main()
