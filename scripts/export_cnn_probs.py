#!/usr/bin/env python3
"""
Export 4-class CNN softmax probabilities for all val-split buildings.
Averages softmax across N checkpoint seeds, writes one CSV row per building.

Output CSV columns:
  building_id, uid, tile_id, gt_label, gt_label_idx,
  p_nodmg, p_minor, p_major, p_dest,
  pred_argmax, confidence_max, margin_top1_top2, entropy

Usage:
  python scripts/export_cnn_probs.py \\
    --ckpts models/sampler_noweights_s1/best.pt \\
            models/sampler_noweights_s2/best.pt \\
            models/sampler_noweights_s3/best.pt \\
    --out_csv reports/fusion/cnn_probs.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _load_model(ckpt_path: str, device):
    import torch
    from disaster_bench.models.damage.classifiers import build_classifier
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_type = ckpt.get("model_type", "six_channel")
    num_classes = ckpt.get("num_classes", 4)
    model = build_classifier(model_type, num_classes=num_classes).to(device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def _infer_probs(model, loader, device) -> np.ndarray:
    """Run inference; return softmax probabilities (N, 4)."""
    import torch
    all_probs = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def main() -> None:
    p = argparse.ArgumentParser(description="Export CNN softmax probs for val split")
    p.add_argument("--ckpts", nargs="+", required=True,
                   help="One or more checkpoint paths (averaged)")
    p.add_argument("--out_csv", default="reports/fusion/cnn_probs.csv")
    p.add_argument("--index_csv",  default="data/processed/index.csv")
    p.add_argument("--crops_dir",  default="data/processed/crops_oracle")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--size",       type=int, default=128)
    p.add_argument("--batch",      type=int, default=64)
    p.add_argument("--device",     default=None)
    args = p.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from disaster_bench.data.dataset import (
        build_crop_records, train_val_split, CropDataset,
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build val split
    print("Building crop records...")
    records = build_crop_records(args.index_csv, args.crops_dir)
    print(f"  Total: {len(records)} buildings")
    _, val_recs = train_val_split(records, val_fraction=args.val_fraction, seed=args.seed)
    print(f"  Val:   {len(val_recs)} buildings")

    # DataLoader
    ds = CropDataset(val_recs, size=args.size, augment=False, preload=True)
    loader = DataLoader(
        ds, batch_size=args.batch, shuffle=False, num_workers=0,
        collate_fn=lambda batch: (
            torch.from_numpy(np.stack([b[0] for b in batch])).float(),
            torch.tensor([b[1] for b in batch], dtype=torch.long),
        ),
    )

    # Run inference across all checkpoints, average softmax
    all_seed_probs: list[np.ndarray] = []
    for ckpt_path in args.ckpts:
        print(f"  Loading: {ckpt_path}")
        model = _load_model(ckpt_path, device)
        probs = _infer_probs(model, loader, device)   # (N, 4)
        all_seed_probs.append(probs)
        del model

    avg_probs = np.mean(all_seed_probs, axis=0)  # (N, 4)
    print(f"Averaged softmax across {len(args.ckpts)} checkpoint(s).")

    # Compute uncertainty metrics
    pred_argmax       = avg_probs.argmax(axis=1)
    confidence_max    = avg_probs.max(axis=1)
    sorted_probs      = np.sort(avg_probs, axis=1)[:, ::-1]
    margin_top1_top2  = sorted_probs[:, 0] - sorted_probs[:, 1]
    eps               = 1e-8
    entropy           = -(avg_probs * np.log(avg_probs + eps)).sum(axis=1)

    # Ground-truth labels
    LABEL_NAMES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
    labels     = [r["label"]     for r in val_recs]
    label_idxs = [r["label_idx"] for r in val_recs]

    # Write CSV
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "building_id", "uid", "tile_id",
            "gt_label", "gt_label_idx",
            "p_nodmg", "p_minor", "p_major", "p_dest",
            "pred_argmax", "confidence_max", "margin_top1_top2", "entropy",
        ])
        for i, rec in enumerate(val_recs):
            building_id = f"{rec['tile_id']}:{rec['uid']}"
            writer.writerow([
                building_id,
                rec["uid"],
                rec["tile_id"],
                labels[i],
                label_idxs[i],
                f"{avg_probs[i,0]:.6f}",
                f"{avg_probs[i,1]:.6f}",
                f"{avg_probs[i,2]:.6f}",
                f"{avg_probs[i,3]:.6f}",
                int(pred_argmax[i]),
                f"{confidence_max[i]:.6f}",
                f"{margin_top1_top2[i]:.6f}",
                f"{entropy[i]:.6f}",
            ])

    print(f"\nWrote {len(val_recs)} rows to {out_path}")

    # Quick accuracy summary
    labels_arr = np.array(label_idxs)
    correct = (pred_argmax == labels_arr).mean()
    print(f"Accuracy (argmax): {correct:.4f}")
    from collections import Counter
    dist = Counter(int(p) for p in pred_argmax)
    print("Prediction distribution:", dict(sorted(dist.items())))
    from collections import defaultdict
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for pr, gt in zip(pred_argmax, labels_arr):
        if pr == gt: tp[gt] += 1
        else: fp[pr] += 1; fn[gt] += 1
    f1s = []
    for c in range(4):
        prec = tp[c]/(tp[c]+fp[c]) if tp[c]+fp[c] > 0 else 0.0
        rec  = tp[c]/(tp[c]+fn[c]) if tp[c]+fn[c] > 0 else 0.0
        f1   = 2*prec*rec/(prec+rec) if prec+rec > 0 else 0.0
        f1s.append(f1)
    print(f"Per-class F1: no={f1s[0]:.3f}  minor={f1s[1]:.3f}  major={f1s[2]:.3f}  dest={f1s[3]:.3f}")
    print(f"Macro F1: {np.mean(f1s):.4f}")


if __name__ == "__main__":
    main()
