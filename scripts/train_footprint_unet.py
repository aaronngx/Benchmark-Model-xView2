#!/usr/bin/env python3
"""
Train U-Net building footprint detector on rasterized polygon masks.
Ref §A2 footprint stage: "predict building footprints from post-disaster imagery"

Approach:
  - Read post-disaster (fallback: pre) satellite tiles from index.csv
  - Rasterize ALL building polygons (including un-classified) into binary masks
  - Patchify 1024x1024 tiles into patch_size x patch_size crops with stride overlap
  - Train lightweight U-Net with BCEWithLogitsLoss + computed pos_weight

Usage:
    python scripts/train_footprint_unet.py \\
        --index_csv data/processed/index.csv \\
        --out_dir   models/footprint_unet \\
        [--epochs 30] [--batch 4] [--patch_size 512] [--stride 256]

After training, set track1_deploy.yaml:
    footprint_ckpt: models/footprint_unet/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np


def collate_footprint(batch):
    import torch
    imgs  = torch.from_numpy(np.stack([b[0] for b in batch])).float()
    masks = torch.from_numpy(np.stack([b[1] for b in batch])).float()
    return imgs, masks


def run(args: argparse.Namespace) -> None:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        print("torch not installed. Run: pip install torch")
        sys.exit(1)

    from disaster_bench.models.footprints.semantic_seg import (
        build_unet,
        build_footprint_records,
        footprint_tile_split,
        FootprintTileDataset,
        save_footprint_checkpoint,
    )

    print("Building records...")
    records = build_footprint_records(args.index_csv)
    if not records:
        print(f"No valid tiles found in {args.index_csv}", file=sys.stderr)
        sys.exit(1)

    train_recs, val_recs = footprint_tile_split(records, val_fraction=args.val_fraction)
    print(f"  Train tiles: {len(train_recs)}  Val tiles: {len(val_recs)}")

    train_ds = FootprintTileDataset(
        train_recs,
        patch_size=args.patch_size,
        stride=args.stride,
        augment=True,
        preload=not args.no_preload,
    )
    val_ds = FootprintTileDataset(
        val_recs,
        patch_size=args.patch_size,
        stride=args.stride,
        augment=False,
        preload=not args.no_preload,
    )
    print(f"  Train patches: {len(train_ds)}  Val patches: {len(val_ds)}")

    # Compute pos_weight = bg_pixels / building_pixels for class imbalance
    print("  Computing pos_weight from training data...")
    pos_weight_val = train_ds.compute_pos_weight()
    print(f"  pos_weight = {pos_weight_val:.1f}  (bg/fg pixel ratio)")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        collate_fn=collate_footprint, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        collate_fn=collate_footprint, num_workers=0,
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_unet(in_ch=3, out_ch=1, base=args.base).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    pw_tensor = torch.tensor([pos_weight_val], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_iou  = -1.0
    best_path = out_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        # --- train ---
        model.train()
        total_loss, n_batches = 0.0, 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)          # (B, 1, H, W)
            loss   = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1
        scheduler.step()

        # --- val: binary IoU for building class ---
        model.eval()
        intersect = 0.0
        union     = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                prob = torch.sigmoid(model(imgs)).cpu().numpy()[:, 0]  # (B,H,W)
                pred = (prob > 0.5).astype(np.uint8)
                gt   = masks[:, 0].numpy().astype(np.uint8)
                intersect += float((pred & gt).sum())
                union     += float((pred | gt).sum())

        val_iou = intersect / max(union, 1.0)
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={total_loss / max(n_batches, 1):.4f} | "
            f"val_iou(bldg)={val_iou:.4f}",
            flush=True,
        )

        if val_iou > best_iou:
            best_iou = val_iou
            save_footprint_checkpoint(
                model, best_path,
                metadata={
                    "epoch":      epoch,
                    "val_iou":    val_iou,
                    "patch_size": args.patch_size,
                    "base":       args.base,
                },
            )
            print(f"  -> saved best.pt (iou={val_iou:.4f})")

    print(f"\nDone. Best val IoU(bldg)={best_iou:.4f} -> {out_dir}/best.pt")
    print(f"Update track1_deploy.yaml: footprint_ckpt: {out_dir}/best.pt")


def main() -> None:
    p = argparse.ArgumentParser(description="Train U-Net building footprint detector")
    p.add_argument("--index_csv",    default="data/processed/index.csv",
                   help="Index CSV from build-index")
    p.add_argument("--out_dir",      default="models/footprint_unet",
                   help="Output directory for checkpoints")
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch",        type=int,   default=8,
                   help="Patches per batch (default 8 for 256px patches; use 4 for 512px)")
    p.add_argument("--patch_size",   type=int,   default=256,
                   help="Patch size for patchifying tiles (default 256; use 512 for more context)")
    p.add_argument("--stride",       type=int,   default=256,
                   help="Stride between patches — smaller = more overlap (default 256)")
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--base",         type=int,   default=32,
                   help="U-Net base channel count (32 = ~7.8M params)")
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--device",       default=None,
                   help="cuda | cpu (default: auto-detect)")
    p.add_argument("--no_preload",   action="store_true",
                   help="Disable tile preloading (saves RAM but slower training)")
    run(p.parse_args())


if __name__ == "__main__":
    main()
