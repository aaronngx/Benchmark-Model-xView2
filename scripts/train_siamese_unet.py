#!/usr/bin/env python3
"""
Train Siamese U-Net for pixel-wise damage severity.
Ref §2B learned models: "Siamese U-Net pixel-wise damage severity {0..4}"

Usage:
    python scripts/train_siamese_unet.py \\
        --index_csv data/processed/index.csv \\
        --crops_dir data/processed/crops_oracle \\
        --out_dir   models/siamese_unet \\
        [--post_only] [--epochs 30] [--batch 8] [--size 256]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np


def collate_pixel(batch):
    import torch
    xs  = torch.from_numpy(np.stack([b[0] for b in batch])).float()
    ys  = torch.from_numpy(np.stack([b[1] for b in batch])).long()
    return xs, ys


def run(args: argparse.Namespace) -> None:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        print("torch not installed. Run: pip install torch")
        sys.exit(1)

    from disaster_bench.data.dataset import build_crop_records, train_val_split, DAMAGE_CLASSES
    from disaster_bench.models.damage.siamese_unet import (
        build_siamese_unet, SEVERITY_CLASSES, PixelDamageDataset,
        build_pixel_records, save_siamese_unet_checkpoint,
    )

    print("Building records...")
    records = build_crop_records(args.index_csv, args.crops_dir)
    train_recs, val_recs = train_val_split(records, val_fraction=args.val_fraction)
    print(f"  Train: {len(train_recs)}  Val: {len(val_recs)}")

    pixel_train = build_pixel_records(train_recs)
    pixel_val   = build_pixel_records(val_recs)

    train_ds = PixelDamageDataset(pixel_train, size=args.size, augment=True)
    val_ds   = PixelDamageDataset(pixel_val,   size=args.size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              collate_fn=collate_pixel, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              collate_fn=collate_pixel, num_workers=0)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  post_only={args.post_only}")

    model = build_siamese_unet(
        num_classes=SEVERITY_CLASSES, post_only=args.post_only
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Class weights (severity 0-4, 0=background is common)
    counts = np.zeros(SEVERITY_CLASSES, dtype=np.float32)
    for r in pixel_train:
        counts[r["severity"]] += args.size * args.size
    counts[0] = counts[1:].mean()  # background weight = mean of building classes
    counts    = np.where(counts == 0, 1, counts)
    cw        = torch.from_numpy(1.0 / counts * SEVERITY_CLASSES).float().to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=cw, ignore_index=-1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_miou   = -1.0
    best_path   = out_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        # --- train ---
        model.train()
        total_loss, n_batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)                  # (B, C, H, W)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1
        scheduler.step()

        # --- val: compute per-class pixel IoU ---
        model.eval()
        intersect = np.zeros(SEVERITY_CLASSES, dtype=np.float64)
        union_arr = np.zeros(SEVERITY_CLASSES, dtype=np.float64)
        with torch.no_grad():
            for x, y in val_loader:
                x    = x.to(device)
                pred = model(x).argmax(dim=1).cpu().numpy()   # (B, H, W)
                gt   = y.numpy()
                for c in range(SEVERITY_CLASSES):
                    p_c = pred == c; g_c = gt == c
                    intersect[c] += (p_c & g_c).sum()
                    union_arr[c] += (p_c | g_c).sum()

        iou = np.where(union_arr > 0, intersect / union_arr, 0.0)
        miou_all  = iou.mean()
        miou_bldg = iou[1:].mean()  # exclude background

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={total_loss/max(n_batches,1):.4f} | "
              f"mIoU={miou_all:.4f} mIoU(bldg)={miou_bldg:.4f} "
              f"[{' '.join(f'{v:.3f}' for v in iou)}]", flush=True)

        if miou_bldg > best_miou:
            best_miou = miou_bldg
            save_siamese_unet_checkpoint(model, best_path, post_only=args.post_only,
                                         epoch=epoch, val_miou=miou_bldg)
            print(f"  -> saved best.pt (mIoU_bldg={miou_bldg:.4f})")

    print(f"\nDone. Best mIoU(bldg)={best_miou:.4f} -> {out_dir}/best.pt")


def main() -> None:
    p = argparse.ArgumentParser(description="Train Siamese U-Net pixel damage model")
    p.add_argument("--index_csv",    default="data/processed/index.csv")
    p.add_argument("--crops_dir",    default="data/processed/crops_oracle")
    p.add_argument("--out_dir",      default="models/siamese_unet")
    p.add_argument("--post_only",    action="store_true",
                   help="Ablation: use only post image (no pre) — single stream")
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch",        type=int,   default=8)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--size",         type=int,   default=256)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--device",       default=None, help="cuda | cpu | auto")
    run(p.parse_args())


if __name__ == "__main__":
    main()
