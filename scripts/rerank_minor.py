#!/usr/bin/env python3
"""
Minor cascade reranker: re-score top-K minor candidates at higher crop resolution.

Strategy (Option B — CNN rerank):
  The ViT encoder uses fixed positional embeddings (64 tokens for 128px/16px-patch);
  256px input fails with a shape mismatch. Instead, we use SixChannelCNN which uses
  global average pooling (GAP) and is resolution-agnostic at inference time.

Pipeline:
  1. Load binary ensemble scores for minor (from cv_minor ensemble)
  2. Take top-K buildings by ensemble score (default K=200)
  3. Reload each candidate's crop at rerank_size (default 256px)
  4. Run SixChannelCNN forward -> P(minor) = softmax(logits)[1]
  5. Re-rank top-K by CNN score and report Top-100 recall vs baseline

Note: CNN at 256px is a heuristic ablation. The CNN was trained at 128px, so the
absolute score values differ from training. The re-ranking uses relative ordering
of CNN scores among candidates, not the absolute threshold.

Usage:
  python scripts/rerank_minor.py \\
      --ensemble_dir models/binary_ensemble/cv_minor \\
      --cnn_ckpt     models/sampler_noweights/best.pt \\
      --buildings_csv data/processed/buildings_v2.csv \\
      --crops_dir    data/processed/crops_oracle \\
      --embeddings_npz data/processed/embeddings_mae80_s1.npz \\
      --topk 200 \\
      --rerank_size 256
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def load_ensemble_scores(
    ensemble_dir: Path,
    Z: np.ndarray,
    building_ids: np.ndarray,
) -> np.ndarray:
    """Return ensemble mean score for each building (N,)."""
    member_files = sorted(ensemble_dir.glob("member_*.pkl"))
    if not member_files:
        print(f"ERROR: no member_*.pkl in {ensemble_dir}", file=sys.stderr)
        sys.exit(1)

    all_probs = []
    for mf in member_files:
        with open(mf, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, tuple):
            scaler, clf = obj
            X = scaler.transform(Z)
        else:
            clf = obj
            X = Z
        prob = clf.predict_proba(X)[:, 1]
        all_probs.append(prob)

    return np.mean(all_probs, axis=0)   # (N,)


def cnn_scores_at_size(
    candidate_indices: np.ndarray,
    records_all: list[dict],
    cnn_ckpt: Path,
    rerank_size: int,
) -> np.ndarray:
    """
    Run SixChannelCNN at rerank_size on each candidate; return P(minor) scores.
    CNN uses GAP so it accepts any spatial resolution at inference.
    """
    import torch
    from disaster_bench.data.dataset import load_crop_pair
    from disaster_bench.models.damage.classifiers import build_classifier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(str(cnn_ckpt), map_location=device, weights_only=False)
    model_type = ckpt.get("model_type", "six_channel")
    model = build_classifier(model_type, num_classes=4).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    scores = np.zeros(len(candidate_indices), dtype=np.float32)
    with torch.no_grad():
        for out_idx, rec_idx in enumerate(candidate_indices):
            r = records_all[rec_idx]
            try:
                x = load_crop_pair(r["pre_path"], r["post_path"], size=rerank_size)
                x_t = torch.from_numpy(x[np.newaxis]).float().to(device)
                logits = model(x_t)
                prob_minor = torch.softmax(logits, dim=1)[0, 1].item()
                scores[out_idx] = prob_minor
            except Exception as e:
                print(f"  WARNING: failed for {r['tile_id']}:{r['uid']}: {e}",
                      file=sys.stderr)
                scores[out_idx] = 0.0

    return scores


def recall_at_k(scores: np.ndarray, labels: np.ndarray, cls: int, k: int) -> float:
    """Recall of top-k scored items for class cls."""
    n_pos = int((labels == cls).sum())
    if n_pos == 0:
        return 0.0
    order  = np.argsort(-scores)
    k_real = min(k, len(scores))
    tp     = int((labels[order[:k_real]] == cls).sum())
    return round(tp / n_pos, 4)


def main() -> None:
    p = argparse.ArgumentParser(description="Rerank top-K minor candidates at higher resolution")
    p.add_argument("--ensemble_dir",   default="models/binary_ensemble/cv_minor",
                   help="Path to cv_minor ensemble directory (member_*.pkl)")
    p.add_argument("--cnn_ckpt",       default="models/sampler_noweights/best.pt",
                   help="SixChannelCNN checkpoint for CNN reranking")
    p.add_argument("--buildings_csv",  default="data/processed/buildings_v2.csv")
    p.add_argument("--crops_dir",      default="data/processed/crops_oracle")
    p.add_argument("--embeddings_npz", default="data/processed/embeddings_mae80_s1.npz")
    p.add_argument("--topk",           type=int, default=200,
                   help="Number of top candidates from ensemble to re-score (default 200)")
    p.add_argument("--rerank_size",    type=int, default=256,
                   help="Crop resolution for CNN re-scoring (default 256)")
    p.add_argument("--report_k",       type=int, default=100,
                   help="Report recall at this K (default 100)")
    args = p.parse_args()

    # ----------------------------------------------------------------- load
    try:
        import torch  # noqa: F401
    except ImportError:
        print("torch not installed", file=sys.stderr)
        sys.exit(1)

    from disaster_bench.data.dataset import build_crop_records, train_val_split

    print("Loading embeddings...")
    npz = np.load(args.embeddings_npz, allow_pickle=True)
    Z          = npz["Z"].astype(np.float32)
    label_idx  = npz["label_idx"].astype(int)
    splits     = npz["split"]
    tile_id    = npz["tile_id"]
    uid_arr    = npz["uid"]

    if "building_id" in npz:
        building_ids = npz["building_id"]
    else:
        building_ids = np.array([f"{t}:{u}" for t, u in zip(tile_id, uid_arr)])

    val_mask = (splits == "val")
    Z_val    = Z[val_mask]
    y_val    = label_idx[val_mask]
    bids_val = building_ids[val_mask]
    tid_val  = tile_id[val_mask]
    uid_val  = uid_arr[val_mask]

    n_minor_val = int((y_val == 1).sum())
    print(f"Val buildings: {val_mask.sum()}  (minor={n_minor_val})")

    # ---------------------------------------------------------------- ensemble scores
    ensemble_dir = Path(args.ensemble_dir)
    print(f"\nScoring with ensemble ({len(list(ensemble_dir.glob('member_*.pkl')))} members)...")
    ens_scores = load_ensemble_scores(ensemble_dir, Z_val, bids_val)

    # Baseline recall before reranking
    baseline_top100 = recall_at_k(ens_scores, y_val, cls=1, k=args.report_k)
    baseline_topK   = recall_at_k(ens_scores, y_val, cls=1, k=args.topk)
    print(f"\nBaseline (ensemble, 128px):")
    print(f"  Top-{args.report_k} recall = {baseline_top100:.4f}")
    print(f"  Top-{args.topk}  recall = {baseline_topK:.4f}  (candidates for rerank)")

    # ---------------------------------------------------------------- get top-K candidates
    top_k_order = np.argsort(-ens_scores)[:args.topk]    # indices into val arrays
    top_k_bids  = bids_val[top_k_order]
    top_k_labels = y_val[top_k_order]
    n_minor_in_topk = int((top_k_labels == 1).sum())
    print(f"\nTop-{args.topk} candidates: {args.topk} buildings, "
          f"{n_minor_in_topk} are true minor ({100*n_minor_in_topk/max(n_minor_val,1):.1f}% of all val minor)")

    # Build records lookup for path retrieval
    crops_dir = args.crops_dir
    bid_to_paths: dict[str, dict] = {}
    import csv
    with open(args.buildings_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            bid_to_paths[row["building_id"]] = {
                "pre_path":  row["pre_path"],
                "post_path": row["post_path"],
                "label_idx": int(row["label_idx"]),
            }

    # Build ordered candidate record list
    cand_records = []
    for bid in top_k_bids:
        info = bid_to_paths.get(str(bid))
        if info:
            cand_records.append(info)
        else:
            # Fallback: construct path from tile_id / uid
            parts = str(bid).split(":")
            if len(parts) == 2:
                t_id, u_id = parts
                pre_p  = Path(crops_dir) / t_id / u_id / "pre_bbox.png"
                post_p = Path(crops_dir) / t_id / u_id / "post_bbox.png"
                cand_records.append({
                    "pre_path": str(pre_p), "post_path": str(post_p),
                    "label_idx": int(y_val[top_k_order[len(cand_records)]]),
                })
            else:
                cand_records.append({"pre_path": "", "post_path": "", "label_idx": 0})

    # ---------------------------------------------------------------- CNN rerank
    cnn_ckpt = Path(args.cnn_ckpt)
    if not cnn_ckpt.exists():
        print(f"\nWARNING: CNN checkpoint not found at {cnn_ckpt}. Skipping CNN rerank.")
        return

    print(f"\nCNN reranking {args.topk} candidates at {args.rerank_size}px "
          f"(using {cnn_ckpt.name})...")
    print("  Note: CNN was trained at 128px. 256px rerank is a heuristic ablation.")

    # Build a dummy all_records list for cnn_scores_at_size
    all_records_compat = cand_records
    candidate_indices  = np.arange(len(cand_records))
    cnn_scores_cands   = cnn_scores_at_size(
        candidate_indices, all_records_compat, cnn_ckpt, args.rerank_size
    )

    # Re-rank by CNN score within the top-K candidates
    cand_labels = np.array([r["label_idx"] for r in cand_records])
    rerank_order = np.argsort(-cnn_scores_cands)

    # To compute final Top-report_k recall: use CNN-reranked order within candidates,
    # then append remaining val buildings (not in candidates) at the end
    final_labels = np.concatenate([
        cand_labels[rerank_order],       # CNN-reranked top-K
        y_val[np.argsort(-ens_scores)[args.topk:]],   # rest by ensemble score
    ])
    after_topk   = recall_at_k(
        np.concatenate([cnn_scores_cands[rerank_order],
                        ens_scores[np.argsort(-ens_scores)[args.topk:]] - 1e6]),
        final_labels, cls=1, k=args.report_k,
    )

    # Simpler view: recall within the candidate pool itself
    cand_top100_before = recall_at_k(np.ones(len(cand_labels)), cand_labels, 1, args.report_k)
    cand_top100_cnn    = recall_at_k(cnn_scores_cands, cand_labels, 1, args.report_k)

    print(f"\n{'='*60}")
    print(f"Results (minor, val split)")
    print(f"{'='*60}")
    print(f"  Baseline ensemble top-{args.report_k} recall (full val):  {baseline_top100:.4f}")
    print(f"  Among top-{args.topk} candidates:")
    print(f"    Before CNN rerank  (all equally ranked):  {cand_top100_before:.4f}")
    print(f"    After  CNN rerank  (sorted by CNN score): {cand_top100_cnn:.4f}")
    print(f"  Full val top-{args.report_k} recall after CNN rerank:     {after_topk:.4f}")
    print(f"\n  n_minor in top-{args.topk}: {n_minor_in_topk}/{n_minor_val} val positives")
    print(f"\n  Note: CNN rerank at 256px is a heuristic; if cand_top100_cnn < "
          f"cand_top100_before, CNN score is not helpful at this resolution.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
