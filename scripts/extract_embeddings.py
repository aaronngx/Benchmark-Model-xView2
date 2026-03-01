#!/usr/bin/env python3
"""
Extract 1536-dim fusion embeddings from a frozen ViT encoder for every building.

The fusion vector z = cat(f_pre, f_post, |f_post-f_pre|, f_pre*f_post) matches
the internal representation used by ViTDamageClassifier before its head.

Output .npz contains:
  Z         : (N, 1536) float32 — fusion embeddings
  label_idx : (N,)      int32   — 0=no-damage 1=minor 2=major 3=destroyed
  split     : (N,)      str     — "train" or "val" (tile-level 80/20, seed=42)
  tile_id   : (N,)      str
  uid       : (N,)      str

Usage:
  python scripts/extract_embeddings.py \
    --ckpt_path models/mae80/s1/best.pt \
    --out_npz   data/processed/embeddings_mae80_s1.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> None:
    p = argparse.ArgumentParser(description="Extract ViT fusion embeddings for all buildings")
    p.add_argument("--ckpt_path",  default="models/mae80/s1/best.pt",
                   help="Path to vit_finetune best.pt checkpoint")
    p.add_argument("--out_npz",    default="data/processed/embeddings_mae80_s1.npz",
                   help="Output .npz path")
    p.add_argument("--index_csv",  default="data/processed/index.csv")
    p.add_argument("--crops_dir",  default="data/processed/crops_oracle")
    p.add_argument("--size",       type=int, default=128)
    p.add_argument("--batch",      type=int, default=64)
    args = p.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        print("torch not installed", file=sys.stderr)
        sys.exit(1)

    from disaster_bench.data.dataset import build_crop_records, CropDataset, train_val_split
    from disaster_bench.models.damage.classifiers import build_classifier

    # ------------------------------------------------------------------ data
    all_records = build_crop_records(args.index_csv, args.crops_dir)
    train_recs, val_recs = train_val_split(all_records, val_fraction=0.2)  # seed=42 default

    train_set = {(r["tile_id"], r["uid"]) for r in train_recs}

    # Ordered list matching dataset iteration order
    ordered = all_records
    splits = ["train" if (r["tile_id"], r["uid"]) in train_set else "val"
              for r in ordered]

    def collate(batch):
        xs = torch.from_numpy(np.stack([b[0] for b in batch])).float()
        ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return xs, ys

    ds = CropDataset(ordered, size=args.size, augment=False, preload=True)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        collate_fn=collate, num_workers=0)

    # ----------------------------------------------------------------- model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_classifier("vit_finetune", num_classes=4).to(device)
    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    print(f"Loaded: {args.ckpt_path}  (epoch {ckpt.get('epoch', '?')}, "
          f"val_macro_f1={ckpt.get('val_macro_f1', '?')})")
    print(f"Buildings: {len(ordered)}  train={sum(s=='train' for s in splits)}  "
          f"val={sum(s=='val' for s in splits)}")

    # -------------------------------------------------------- embedding loop
    all_z: list[np.ndarray] = []

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(loader):
            x = x.to(device)
            pre  = x[:, :3]
            post = x[:, 3:]
            f_pre, f_post = model.encoder.forward_features(pre, post)
            z = torch.cat(
                [f_pre, f_post, (f_post - f_pre).abs(), f_pre * f_post],
                dim=1,
            )  # (B, 1536)
            all_z.append(z.cpu().numpy())

            if (batch_idx + 1) % 20 == 0:
                done = min((batch_idx + 1) * args.batch, len(ordered))
                print(f"  {done}/{len(ordered)} buildings embedded...")

    Z = np.concatenate(all_z, axis=0).astype(np.float32)  # (N, 1536)

    # ------------------------------------------------------------- save
    out_path = Path(args.out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_path,
        Z=Z,
        label_idx=np.array([r["label_idx"] for r in ordered], dtype=np.int32),
        split=np.array(splits),
        tile_id=np.array([r["tile_id"] for r in ordered]),
        uid=np.array([r["uid"] for r in ordered]),
    )

    print(f"\nSaved {Z.shape} embeddings -> {out_path}")

    # Quick label summary
    labels = np.array([r["label_idx"] for r in ordered])
    names = ["no-damage", "minor", "major", "destroyed"]
    for cls, name in enumerate(names):
        n = (labels == cls).sum()
        tr = sum(1 for r, s in zip(ordered, splits) if r["label_idx"] == cls and s == "train")
        va = n - tr
        print(f"  {name:12s}: {n:5d} total  (train={tr}, val={va})")


if __name__ == "__main__":
    main()
