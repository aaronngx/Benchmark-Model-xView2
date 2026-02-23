#!/usr/bin/env python3
"""
Benchmark visualization suite.
Generates four charts from runs/ metrics:
  1. Macro-F1 bar chart          → plots/macro_f1_bar.png
  2. Per-class F1 grouped bars   → plots/per_class_f1.png
  3. Confusion matrix heatmap    → plots/confusion_<run>.png
  4. Coverage vs Macro-F1 scatter → plots/coverage_vs_f1.png

Usage:
    python scripts/visualize_benchmark.py [--runs_dir runs] [--out_dir plots]
    python scripts/visualize_benchmark.py --confusion_run track1_final
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# ── colour palette (colourblind-safe) ─────────────────────────────────────────
TRACK_COLORS = {
    "placeholder": "#b0b0b0",
    "2a":          "#4e9af1",
    "2b":          "#29b09d",
    "3_baseline":  "#f4a259",
    "3_final":     "#e07b39",
    "1_baseline":  "#7b5ea7",
    "1_final":     "#d62728",
    "default":     "#1f77b4",
}

CLASS_COLORS = {
    "no-damage":    "#2ca02c",
    "minor-damage": "#ff7f0e",
    "major-damage": "#d62728",
    "destroyed":    "#9467bd",
}

DAMAGE_CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]

# Canonical display order for runs (best-to-worst within a tier)
RUN_ORDER = [
    "track1_placeholder",
    "track2a_run1",
    "track2a_v2",
    "track2b_v1",
    "track3_v1",
    "track3_6ch_v1",
    "track3_final",
    "track1_6ch_v1",
    "track1_final",
]

# Friendly labels for bar charts
RUN_LABELS = {
    "track1_placeholder":  "T1 — Placeholder\n(always no-damage)",
    "track2a_run1":        "T2A — Heuristic v1\n(abs-diff only)",
    "track2a_v2":          "T2A — Heuristic v2\n(diff+SSIM+edge)",
    "track2b_v1":          "T2B — Severity map\n(pixel heuristic)",
    "track3_v1":           "T3 — Hybrid\n(no model)",
    "track3_6ch_v1":       "T3 — Hybrid\n(6ch CNN early)",
    "track3_final":        "T3 — Hybrid\n(6ch CNN final)",
    "track1_6ch_v1":       "T1 — Oracle\n(6ch CNN early)",
    "track1_final":        "T1 — Oracle\n(6ch CNN final) ★",
}


def _color_for(run_name: str) -> str:
    n = run_name.lower()
    if "placeholder" in n:  return TRACK_COLORS["placeholder"]
    if "2a" in n:           return TRACK_COLORS["2a"]
    if "2b" in n:           return TRACK_COLORS["2b"]
    if "3_final" in n:      return TRACK_COLORS["3_final"]
    if "3_" in n:           return TRACK_COLORS["3_baseline"]
    if "1_final" in n or "track1_final" in n: return TRACK_COLORS["1_final"]
    if "1_" in n or "track1_" in n:           return TRACK_COLORS["1_baseline"]
    return TRACK_COLORS["default"]


def load_metrics(runs_dir: Path) -> dict[str, dict]:
    out = {}
    for p in runs_dir.glob("*/metrics.json"):
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
            if m.get("macro_f1") is not None:
                out[p.parent.name] = m
        except Exception:
            pass
    return out


def _ordered_runs(metrics: dict) -> list[str]:
    ordered = [r for r in RUN_ORDER if r in metrics]
    extra   = sorted(r for r in metrics if r not in RUN_ORDER)
    return ordered + extra


# ── Chart 1: Macro-F1 bar chart ────────────────────────────────────────────────

def chart_macro_f1(metrics: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    runs   = _ordered_runs(metrics)
    values = [metrics[r]["macro_f1"] for r in runs]
    labels = [RUN_LABELS.get(r, r) for r in runs]
    colors = [_color_for(r) for r in runs]

    fig, ax = plt.subplots(figsize=(max(10, len(runs) * 1.35), 5.5))
    bars = ax.bar(range(len(runs)), values, color=colors, width=0.65,
                  edgecolor="white", linewidth=0.8)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Random-classifier baseline (0.25 for 4-class)
    ax.axhline(0.25, color="#888", linestyle="--", linewidth=1.1, label="Random (0.250)")

    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("Macro-F1 (4-class, unweighted)", fontsize=10)
    ax.set_title("Benchmark Macro-F1 by Track & Variant", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylim(0, min(1.0, max(values) * 1.18))
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9)

    # Legend patches for track tiers
    patches = [
        mpatches.Patch(color=TRACK_COLORS["placeholder"], label="Track 1 placeholder"),
        mpatches.Patch(color=TRACK_COLORS["2a"],          label="Track 2A (heuristic)"),
        mpatches.Patch(color=TRACK_COLORS["2b"],          label="Track 2B (pixel map)"),
        mpatches.Patch(color=TRACK_COLORS["3_baseline"],  label="Track 3 (hybrid)"),
        mpatches.Patch(color=TRACK_COLORS["1_baseline"],  label="Track 1 (6ch CNN)"),
    ]
    ax.legend(handles=patches, loc="upper left", fontsize=8.5, framealpha=0.85)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Chart 2: Per-class F1 grouped bars ────────────────────────────────────────

def chart_per_class_f1(metrics: dict, out_path: Path,
                       highlight_runs: list[str] | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Show a curated subset to avoid clutter
    if highlight_runs is None:
        priority = ["track1_placeholder", "track2a_v2", "track2b_v1",
                    "track3_final", "track1_final"]
        highlight_runs = [r for r in priority if r in metrics]
        # Also include any deploy/siamese/vlm runs if present
        for extra in sorted(metrics):
            if any(x in extra for x in ("deploy", "siamese", "pre_post", "centroid", "vlm")):
                if extra not in highlight_runs:
                    highlight_runs.append(extra)

    n_runs    = len(highlight_runs)
    n_classes = len(DAMAGE_CLASSES)
    x         = np.arange(n_classes)
    width     = 0.75 / max(n_runs, 1)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, run in enumerate(highlight_runs):
        pf     = metrics[run].get("per_class_f1", {})
        vals   = [pf.get(c, 0.0) for c in DAMAGE_CLASSES]
        offset = (i - n_runs / 2 + 0.5) * width
        label  = RUN_LABELS.get(run, run).replace("\n", " ")
        bars   = ax.bar(x + offset, vals, width * 0.9,
                        label=label, color=_color_for(run),
                        alpha=0.85, edgecolor="white", linewidth=0.6)
        for bar, val in zip(bars, vals):
            if val > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.008,
                        f"{val:.2f}", ha="center", va="bottom",
                        fontsize=6.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["No-damage\n(majority)", "Minor-damage\n(very rare)",
         "Major-damage\n(very rare)", "Destroyed\n(common)"],
        fontsize=9,
    )
    ax.set_ylabel("F1 Score", fontsize=10)
    ax.set_title("Per-class F1 by Track (class imbalance visible)", fontsize=12,
                 fontweight="bold", pad=12)
    ax.set_ylim(0, 1.12)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85, ncol=2)

    # Annotate imbalance note
    ax.text(1.0, 0.95, "← minor & major are rare:\nonly 47 + 44 samples each",
            transform=ax.transAxes, fontsize=7.5, va="top", ha="right",
            color="#666", style="italic")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Chart 3: Confusion matrix heatmap ─────────────────────────────────────────

def chart_confusion_matrix(metrics: dict, run_name: str, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m  = metrics.get(run_name, {})
    cm = m.get("confusion_matrix")
    if cm is None:
        print(f"  No confusion_matrix in {run_name}/metrics.json — run eval-run first")
        return

    labels = DAMAGE_CLASSES
    abbr   = ["no-dmg", "minor", "major", "destr"]
    n      = len(labels)
    mat    = np.array([[cm.get(r, {}).get(c, 0) for c in labels] for r in labels],
                      dtype=float)

    # Row-normalise to show recall per class
    row_sums = mat.sum(axis=1, keepdims=True)
    mat_norm = np.where(row_sums > 0, mat / row_sums, 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    for ax, data, title, fmt in [
        (axes[0], mat,      "Counts",               ".0f"),
        (axes[1], mat_norm, "Row-normalised (recall)", ".2f"),
    ]:
        im = ax.imshow(data, cmap="Blues", vmin=0,
                       vmax=data.max() if data.max() > 0 else 1)
        plt.colorbar(im, ax=ax, shrink=0.82)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(abbr, fontsize=9)
        ax.set_yticklabels(abbr, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Ground Truth", fontsize=9)
        ax.set_title(f"Confusion Matrix — {title}", fontsize=10, pad=8)
        thresh = data.max() * 0.5
        for i in range(n):
            for j in range(n):
                val = data[i, j]
                ax.text(j, i, format(val, fmt),
                        ha="center", va="center", fontsize=9,
                        color="white" if val > thresh else "black")

    fig.suptitle(f"Run: {run_name}  |  macro-F1 = {m.get('macro_f1', '?')}",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Chart 4: Coverage vs Macro-F1 scatter ─────────────────────────────────────

def chart_coverage_vs_f1(metrics: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for run, m in sorted(metrics.items()):
        cov   = m.get("coverage")
        macro = m.get("macro_f1")
        lat   = m.get("avg_latency_ms")
        if cov is None or macro is None:
            continue
        size = max(40, min(300, (lat or 50) * 2))
        ax.scatter(cov, macro, s=size, color=_color_for(run),
                   alpha=0.85, edgecolors="white", linewidth=0.8, zorder=3)
        label = RUN_LABELS.get(run, run).split("\n")[0]
        ax.annotate(label, (cov, macro),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=7.5, color="#333")

    ax.set_xlabel("Coverage (% GT buildings with prediction)", fontsize=10)
    ax.set_ylabel("Macro-F1", fontsize=10)
    ax.set_title("Coverage vs Macro-F1\n(bubble size ∝ avg latency ms/instance)",
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(0, 0.60)
    ax.yaxis.grid(True, alpha=0.3)
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    note = ("Oracle tracks (T1/T2/T3) have coverage=1.0 by design.\n"
            "Track 1-deploy will appear left of 1.0 once footprint model is trained.")
    ax.text(0.01, 0.01, note, transform=ax.transAxes,
            fontsize=7.5, color="#777", va="bottom", style="italic")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Chart 5: Class imbalance bar (training set) ────────────────────────────────

def chart_class_imbalance(out_path: Path) -> None:
    """Visualise label distribution in train vs val splits."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
    try:
        from disaster_bench.data.dataset import build_crop_records, train_val_split, DAMAGE_CLASSES as DC
        records = build_crop_records("data/processed/index.csv",
                                     "data/processed/crops_oracle")
        train_r, val_r = train_val_split(records, val_fraction=0.2, seed=42)
    except Exception as e:
        print(f"  Skipping imbalance chart: {e}")
        return

    import collections
    tc = collections.Counter(r["label"] for r in train_r)
    vc = collections.Counter(r["label"] for r in val_r)

    x     = np.arange(len(DC))
    width = 0.38
    tv    = [tc[c] for c in DC]
    vv    = [vc[c] for c in DC]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x - width/2, tv, width, label="Train (6612)",
                color=[CLASS_COLORS[c] for c in DC], alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + width/2, vv, width, label="Val (1886)",
                color=[CLASS_COLORS[c] for c in DC], alpha=0.5,
                edgecolor="white", linewidth=0.7, hatch="///")

    for bar, val in list(zip(b1, tv)) + list(zip(b2, vv)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                str(val), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(DC, fontsize=9)
    ax.set_ylabel("Building count", fontsize=10)
    ax.set_title("Dataset Class Distribution — Train vs Val\n"
                 "(tile-level split, seed=42, val_fraction=0.2)",
                 fontsize=11, fontweight="bold", pad=10)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9)

    note = "minor-damage (47 total) and major-damage (44 total) are severely under-represented."
    ax.text(0.5, -0.20, note, transform=ax.transAxes,
            ha="center", fontsize=8, color="#555", style="italic")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Generate benchmark visualizations")
    p.add_argument("--runs_dir",       default="runs",  help="Root of run directories")
    p.add_argument("--out_dir",        default="plots", help="Output directory for PNGs")
    p.add_argument("--confusion_run",  default=None,
                   help="Run name for confusion matrix (default: best macro-F1)")
    p.add_argument("--highlight_runs", nargs="*", default=None,
                   help="Run names to show in per-class chart (default: curated set)")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir  = Path(args.out_dir)
    metrics  = load_metrics(runs_dir)

    if not metrics:
        print(f"No metrics.json files found under {runs_dir}/")
        return

    print(f"Loaded {len(metrics)} runs: {sorted(metrics)}")
    print()

    # Chart 1 — Macro-F1 bar
    print("Generating Chart 1: Macro-F1 bar chart...")
    chart_macro_f1(metrics, out_dir / "macro_f1_bar.png")

    # Chart 2 — Per-class F1
    print("Generating Chart 2: Per-class F1 grouped bars...")
    chart_per_class_f1(metrics, out_dir / "per_class_f1.png",
                       highlight_runs=args.highlight_runs)

    # Chart 3 — Confusion matrix
    best_run = args.confusion_run or max(metrics, key=lambda r: metrics[r].get("macro_f1", 0))
    print(f"Generating Chart 3: Confusion matrix ({best_run})...")
    chart_confusion_matrix(metrics, best_run, out_dir / f"confusion_{best_run}.png")

    # Chart 4 — Coverage vs F1
    print("Generating Chart 4: Coverage vs Macro-F1...")
    chart_coverage_vs_f1(metrics, out_dir / "coverage_vs_f1.png")

    # Chart 5 — Class imbalance
    print("Generating Chart 5: Class imbalance (train/val)...")
    chart_class_imbalance(out_dir / "class_imbalance.png")

    print()
    print(f"All charts saved to: {out_dir}/")
    print("Include in report/slides: macro_f1_bar.png, per_class_f1.png,")
    print(f"  confusion_{best_run}.png, coverage_vs_f1.png, class_imbalance.png")


if __name__ == "__main__":
    main()
