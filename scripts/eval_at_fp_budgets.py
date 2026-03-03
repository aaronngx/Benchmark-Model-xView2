#!/usr/bin/env python3
"""
Evaluate recall at fixed FP/1k-no-damage budgets for minor and major classes.

Supports two source modes:
  checkpoint  — softmax scores from a 4-class CNN/ViT checkpoint
                (scores are lower than ensemble; different scoring pipeline)
  ensemble    — LR ensemble mean probability from binary_ensemble models

Split reconstruction (checkpoint mode):
  Loads run_summary.json; uses val_tile_ids if present (exact).
  Falls back to train_val_split(seed=42) with a [split: reconstructed] label.

Output:
  Console table + reports/tables/fp_budget_recall.csv

Usage:
  # Checkpoint mode
  python scripts/eval_at_fp_budgets.py \\
      --run_ids sampler_noweights_s1,sampler_noweights_s2,mae80_s1 \\
      --source checkpoint \\
      --index_csv data/processed/index.csv \\
      --crops_dir data/processed/crops_oracle

  # Ensemble mode
  python scripts/eval_at_fp_budgets.py \\
      --source ensemble \\
      --ensemble_minor_dir models/binary_ensemble/cv_minor \\
      --ensemble_major_dir models/binary_ensemble/cv_major \\
      --embeddings_npz data/processed/embeddings_mae80_s1.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

RUNS_DIR   = Path("runs")
MODELS_DIR = Path("models")
OUT_DIR    = Path("reports/tables")

FP1K_TARGETS = [5, 10, 20]
TOP_KS       = [50, 100]


# ---------------------------------------------------------------------------
# Metric computation (same logic as train_binary_ensemble.py)
# ---------------------------------------------------------------------------

def compute_fp_budget_metrics(
    scores: np.ndarray,    # (N,) higher = more likely positive
    y_true: np.ndarray,    # (N,) 4-class labels
    cls: int,              # 1=minor or 2=major
) -> dict:
    """
    Compute Top-K recall and recall at fixed FP/1k-no-damage budgets.

    FP/1k definition: FP from no-damage buildings only as denominator.
    This is consistent with all existing project code.
    """
    y_bin   = (y_true == cls).astype(int)
    n_pos   = int(y_bin.sum())
    n_nodmg = int((y_true == 0).sum())

    if n_pos == 0:
        return {"n_pos": 0, "n_nodmg": n_nodmg, "top_k": {}, "fp1k": {}, "max_f1": 0.0}

    order        = np.argsort(-scores)
    y_sorted     = y_bin[y_true[order] != -999]   # all
    y_sorted     = y_bin[order]
    nodmg_sorted = (y_true[order] == 0)

    results: dict = {"n_pos": n_pos, "n_nodmg": n_nodmg}

    # --- Top-K recall ---
    top_k = {}
    for k in sorted(TOP_KS):
        if k > len(y_sorted):
            top_k[k] = None
            continue
        tp = int(y_sorted[:k].sum())
        top_k[k] = {"tp": tp, "recall": round(tp / n_pos, 4)}
    results["top_k"] = top_k

    # --- FP/1k recall ---
    cumtp  = np.cumsum(y_sorted).astype(float)
    cumfp  = np.cumsum(nodmg_sorted).astype(float)
    fp1k   = cumfp / n_nodmg * 1000 if n_nodmg > 0 else np.zeros_like(cumfp)

    fp1k_res = {}
    for target in FP1K_TARGETS:
        idx = int(np.searchsorted(fp1k, target, side="left"))
        idx = min(idx, len(cumtp) - 1)
        tp_here  = int(cumtp[idx])
        fp_here  = int(cumfp[idx])
        rec      = tp_here / n_pos
        actual   = round(float(fp1k[idx]), 1)
        fp1k_res[target] = {"recall": round(rec, 4), "tp": tp_here,
                             "fp": fp_here, "actual_fp1k": actual}
    results["fp1k"] = fp1k_res

    # --- Max F1 over threshold sweep ---
    # At rank i: predicted positives = i+1, TP = cumtp[i]
    n_pred_arr = np.arange(1, len(y_sorted) + 1, dtype=float)
    prec_arr   = cumtp / n_pred_arr
    rec_arr    = cumtp / n_pos
    denom      = prec_arr + rec_arr
    f1_arr     = np.where(denom > 0, 2 * prec_arr * rec_arr / denom, 0.0)
    results["max_f1"] = round(float(f1_arr.max()), 4)

    return results


# ---------------------------------------------------------------------------
# Split reconstruction
# ---------------------------------------------------------------------------

def get_val_records(run_id: str, all_records: list[dict]) -> tuple[list[dict], str]:
    """
    Return (val_records, split_label).
    Prefers val_tile_ids from run_summary.json (exact).
    Falls back to train_val_split(seed=42) with a warning label.
    """
    summary_path = RUNS_DIR / run_id / "run_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        val_tile_ids = summary.get("val_tile_ids")
        if val_tile_ids:
            val_set = set(val_tile_ids)
            val_recs = [r for r in all_records if r["tile_id"] in val_set]
            return val_recs, "exact"

    # Fallback: fixed seed=42 tile split (the tile split seed is 42, not args.seed)
    from disaster_bench.data.dataset import train_val_split
    _, val_recs = train_val_split(all_records, val_fraction=0.2, seed=42)
    return val_recs, "reconstructed(seed=42)"


# ---------------------------------------------------------------------------
# Source: checkpoint
# ---------------------------------------------------------------------------

def eval_checkpoint(
    run_id: str,
    all_records: list[dict],
    crops_dir: str,
    size: int,
) -> list[dict]:
    """Load best.pt, run forward pass on val set, return rows."""
    import torch
    from torch.utils.data import DataLoader
    from disaster_bench.data.dataset import CropDataset
    from disaster_bench.models.damage.classifiers import build_classifier

    # Find checkpoint — try several candidate paths
    summary_path = RUNS_DIR / run_id / "run_summary.json"
    summary = json.load(open(summary_path)) if summary_path.exists() else {}
    model_type = summary.get("model_type", "six_channel")

    # Candidate locations in priority order
    ckpt_candidates = [
        MODELS_DIR / run_id / "best.pt",                     # models/<run_id>/best.pt
        MODELS_DIR / model_type / "best.pt",                 # models/<model_type>/best.pt
    ]
    # For seed-suffixed runs like mae80_s1 → models/mae80/s1/best.pt
    parts = run_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].startswith("s") and parts[1][1:].isdigit():
        ckpt_candidates.append(MODELS_DIR / parts[0] / parts[1] / "best.pt")

    ckpt_path = None
    for c in ckpt_candidates:
        if c.exists():
            ckpt_path = c
            break

    if ckpt_path is None:

        print(f"  WARNING: checkpoint not found for {run_id} "
              f"(tried: {[str(c) for c in ckpt_candidates]})")
        return []

    val_recs, split_label = get_val_records(run_id, all_records)
    if not val_recs:
        print(f"  WARNING: no val records for {run_id}")
        return []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model_type_ckpt = ckpt.get("model_type", "six_channel")
    model = build_classifier(model_type_ckpt, num_classes=4).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    def collate(batch):
        xs = torch.from_numpy(np.stack([b[0] for b in batch])).float()
        ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return xs, ys

    ds = CropDataset(val_recs, size=size, augment=False, preload=True)
    loader = DataLoader(ds, batch_size=64, shuffle=False,
                        collate_fn=collate, num_workers=0)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.extend(y.tolist())

    probs  = np.concatenate(all_probs, axis=0)   # (N, 4)
    labels = np.array(all_labels)

    rows = []
    for cls, cls_name in [(1, "minor"), (2, "major")]:
        m = compute_fp_budget_metrics(probs[:, cls], labels, cls)
        row = _format_row(run_id, "checkpoint", split_label, cls_name, m)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Source: ensemble
# ---------------------------------------------------------------------------

def eval_ensemble(
    ensemble_dir: Path,
    embeddings_npz: str,
    cls: int,
    cls_name: str,
) -> list[dict]:
    """Load LR ensemble, score all val buildings, return rows."""
    import pickle
    from sklearn.preprocessing import StandardScaler

    npz = np.load(embeddings_npz, allow_pickle=True)
    Z          = npz["Z"].astype(np.float32)
    labels     = npz["label_idx"].astype(int)
    splits     = npz["split"]

    val_mask = (splits == "val")
    Z_val    = Z[val_mask]
    y_val    = labels[val_mask]

    # Load ensemble members
    member_files = sorted(ensemble_dir.glob("member_*.pkl"))
    if not member_files:
        print(f"  WARNING: no member_*.pkl in {ensemble_dir}")
        return []

    all_probs = []
    for mf in member_files:
        with open(mf, "rb") as f:
            obj = pickle.load(f)
        # obj may be (scaler, clf) tuple or just clf
        if isinstance(obj, tuple):
            scaler, clf = obj
            X = scaler.transform(Z_val)
        else:
            clf = obj
            X = Z_val
        prob = clf.predict_proba(X)[:, 1]
        all_probs.append(prob)

    scores = np.mean(all_probs, axis=0)

    m = compute_fp_budget_metrics(scores, y_val, cls)
    run_id = ensemble_dir.name
    row = _format_row(run_id, "ensemble", "val_split_from_npz", cls_name, m)
    return [row]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_row(run_id, source, split_label, cls_name, m) -> dict:
    row = {
        "run_id":      run_id,
        "source":      source,
        "split":       split_label,
        "cls":         cls_name,
        "n_pos":       m.get("n_pos", 0),
        "n_nodmg":     m.get("n_nodmg", 0),
        "max_f1":      m.get("max_f1", float("nan")),
    }
    for k in TOP_KS:
        entry = m.get("top_k", {}).get(k)
        row[f"top{k}_rec"] = entry["recall"] if entry else float("nan")
    for t in FP1K_TARGETS:
        entry = m.get("fp1k", {}).get(t)
        row[f"@FP{t}_rec"] = entry["recall"] if entry else float("nan")
    return row


def print_table(rows: list[dict]) -> None:
    hdr = (f"{'run_id':<32} {'src':<12} {'cls':<6} {'n_pos':>5}  "
           f"{'top50':>6} {'top100':>7}  "
           f"{'@FP5':>6} {'@FP10':>6} {'@FP20':>6}  {'maxF1':>6}  split")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        top50  = f"{r['top50_rec']:.3f}"  if not np.isnan(r.get('top50_rec',  float('nan'))) else "  n/a"
        top100 = f"{r['top100_rec']:.3f}" if not np.isnan(r.get('top100_rec', float('nan'))) else "  n/a"
        fp5    = f"{r['@FP5_rec']:.3f}"   if not np.isnan(r.get('@FP5_rec',   float('nan'))) else "  n/a"
        fp10   = f"{r['@FP10_rec']:.3f}"  if not np.isnan(r.get('@FP10_rec',  float('nan'))) else "  n/a"
        fp20   = f"{r['@FP20_rec']:.3f}"  if not np.isnan(r.get('@FP20_rec',  float('nan'))) else "  n/a"
        mf1    = f"{r['max_f1']:.3f}"     if not np.isnan(r.get('max_f1',     float('nan'))) else "  n/a"
        print(f"{r['run_id']:<32} {r['source']:<12} {r['cls']:<6} {r['n_pos']:>5}  "
              f"{top50:>6} {top100:>7}  "
              f"{fp5:>6} {fp10:>6} {fp20:>6}  {mf1:>6}  [{r['split']}]")


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = (["run_id", "source", "split", "cls", "n_pos", "n_nodmg", "max_f1"]
            + [f"top{k}_rec" for k in TOP_KS]
            + [f"@FP{t}_rec" for t in FP1K_TARGETS])
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"\nWrote {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Recall at FP/1k budgets for minor/major")
    p.add_argument("--source", choices=["checkpoint", "ensemble"], default="checkpoint")

    # Checkpoint mode
    p.add_argument("--run_ids",    type=str, default=None,
                   help="Comma-separated run IDs (checkpoint mode)")
    p.add_argument("--index_csv",  default="data/processed/index.csv")
    p.add_argument("--crops_dir",  default="data/processed/crops_oracle")
    p.add_argument("--size",       type=int, default=128)

    # Ensemble mode
    p.add_argument("--ensemble_minor_dir", default="models/binary_ensemble/cv_minor")
    p.add_argument("--ensemble_major_dir", default="models/binary_ensemble/cv_major")
    p.add_argument("--embeddings_npz",     default="data/processed/embeddings_mae80_s1.npz")

    # Output
    p.add_argument("--out_csv", default="reports/tables/fp_budget_recall.csv")

    args = p.parse_args()

    all_rows = []

    if args.source == "checkpoint":
        from disaster_bench.data.dataset import build_crop_records
        print("Building crop records...")
        all_records = build_crop_records(args.index_csv, args.crops_dir)
        print(f"  {len(all_records)} buildings total")

        run_ids = [r.strip() for r in args.run_ids.split(",") if r.strip()]
        for rid in run_ids:
            print(f"\nEvaluating {rid} (checkpoint mode)...")
            rows = eval_checkpoint(rid, all_records, args.crops_dir, args.size)
            all_rows.extend(rows)

    else:  # ensemble
        for cls, cls_name, edir_arg in [
            (1, "minor", args.ensemble_minor_dir),
            (2, "major", args.ensemble_major_dir),
        ]:
            edir = Path(edir_arg)
            if not edir.exists():
                print(f"  WARNING: {edir} not found — skipping {cls_name}")
                continue
            print(f"\nEvaluating {cls_name} ensemble from {edir} ...")
            rows = eval_ensemble(edir, args.embeddings_npz, cls, cls_name)
            all_rows.extend(rows)

    if not all_rows:
        print("No results to report.")
        return

    print_table(all_rows)
    write_csv(all_rows, Path(args.out_csv))


if __name__ == "__main__":
    main()
