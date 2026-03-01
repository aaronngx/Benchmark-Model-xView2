#!/usr/bin/env python3
"""
Temporal MAE pretraining on paired pre/post building crops.

Trains a TemporalMAEEncoder to reconstruct masked patches from both frames.
Saves encoder weights for downstream fine-tuning via train_damage.py.

Key flags:
  --buildings_csv   Path to buildings_v2.csv (default: data/processed/buildings_v2.csv)
  --crops_dir       (unused — paths are in buildings_v2.csv, kept for CLI parity)
  --out_dir         Output directory (default: models/mae/pretrain)
  --backbone        vit_small | vit_tiny | vit_base  (default: vit_small)
  --epochs          Pretraining epochs (default: 200)
  --batch           Batch size (default: 64)
  --lr              Learning rate (default: 1e-4)
  --weight_decay    AdamW weight decay (default: 0.05)
  --mask_ratio_pre  Masking ratio for pre frame (default: 0.75)
  --mask_ratio_post Masking ratio for post frame (default: 0.75)
  --train_disasters Comma-sep disaster IDs to include (LOWO: train side only)
  --crop_size       Crop size (default: 128)
  --patch_size      Patch size (default: 16)
  --seed            Random seed (default: 42)
  --num_workers     DataLoader workers (default: 0)
  --run_id          Optional run ID for logging

Output:
  <out_dir>/
    pretrain_config.json   — all hyperparameters
    encoder.pt             — full TemporalMAE state dict (load with _load_encoder_weights)
    train_curves.jsonl     — one JSON per epoch: {epoch, loss, lr}

Usage (LOWO Run A — pretrain on SoCal):
  python scripts/train_mae.py \\
      --train_disasters socal-fire \\
      --out_dir models/mae/lowo_a \\
      --epochs 200 --batch 64

Usage (full dataset):
  python scripts/train_mae.py --out_dir models/mae/pretrain_all

Ref: prompt.md §Build §3 Pretraining run
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np


def seed_all(seed: int) -> None:
    import torch as _torch
    random.seed(seed)
    np.random.seed(seed)
    _torch.manual_seed(seed)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(seed)
    _torch.backends.cudnn.benchmark = False
    _torch.backends.cudnn.deterministic = True


def collate_mae(batch):
    import torch
    pres = torch.from_numpy(np.stack([b[0] for b in batch])).float()
    posts = torch.from_numpy(np.stack([b[1] for b in batch])).float()
    return pres, posts


def run(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader
    from disaster_bench.data.mae_dataset import MAECropDataset
    from disaster_bench.models.mae.temporal_mae import build_temporal_mae

    seed_all(args.seed)

    # Device
    if args.device == "auto" or args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    # Output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Disasters filter
    train_disasters = (
        [d.strip() for d in args.train_disasters.split(",")]
        if args.train_disasters else None
    )
    print(f"Train disasters: {train_disasters if train_disasters else 'ALL'}")

    # Dataset
    ds = MAECropDataset(
        buildings_csv=args.buildings_csv,
        size=args.crop_size,
        train_disasters=train_disasters,
        augment=True,
        preload=False,
        seed=args.seed,
    )
    print(f"SSL dataset: {len(ds)} crop pairs")

    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_mae,
        drop_last=True,
        pin_memory=(device == "cuda"),
    )

    # Model
    model = build_temporal_mae(
        backbone=args.backbone,
        img_size=args.crop_size,
        patch_size=args.patch_size,
        mask_ratio_pre=args.mask_ratio_pre,
        mask_ratio_post=args.mask_ratio_post,
        norm_pix_loss=True,
    ).to(device)

    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    print(f"Encoder params: {n_enc:,}   Decoder params: {n_dec:,}")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # Save config
    config = vars(args)
    config["n_train"] = len(ds)
    config["train_disasters"] = train_disasters
    (out_dir / "pretrain_config.json").write_text(json.dumps(config, indent=2))

    # Training loop
    curves_path = out_dir / "train_curves.jsonl"
    best_loss = float("inf")
    print(f"\nStarting pretraining: {args.epochs} epochs, batch={args.batch}")
    print("=" * 60)

    with open(curves_path, "w") as f_curves:
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            t0 = time.time()

            for pre, post in loader:
                pre = pre.to(device)
                post = post.to(device)

                optimizer.zero_grad()
                loss, _, _ = model(pre, post, objective=args.mae_objective)
                loss.backward()
                # Gradient clipping for training stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            avg_loss = epoch_loss / max(n_batches, 1)
            current_lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0

            print(f"Epoch [{epoch:3d}/{args.epochs}]  loss={avg_loss:.4f}  "
                  f"lr={current_lr:.2e}  time={elapsed:.1f}s")

            row = {"epoch": epoch, "loss": round(avg_loss, 6), "lr": current_lr}
            f_curves.write(json.dumps(row) + "\n")
            f_curves.flush()

            # Save best checkpoint
            if avg_loss < best_loss:
                best_loss = avg_loss
                _save_encoder(model, out_dir / "encoder.pt", args, epoch, avg_loss,
                              train_disasters=train_disasters)

    # Always save final checkpoint
    _save_encoder(model, out_dir / "encoder_last.pt", args, args.epochs, avg_loss,
                  train_disasters=train_disasters)

    print(f"\nPretraining done. Best loss: {best_loss:.4f}")
    print(f"Encoder saved: {out_dir / 'encoder.pt'}")
    print(f"Load for fine-tuning with:")
    print(f"  python scripts/train_damage.py \\")
    print(f"    --model_type vit_finetune \\")
    print(f"    --init_mode pretrained \\")
    print(f"    --pretrained_ckpt_path {out_dir / 'encoder.pt'}")


def _save_encoder(
    model,
    path: Path,
    args,
    epoch: int,
    loss: float,
    train_disasters: list[str] | None = None,
) -> None:
    """Save full model state dict (includes encoder.* and decoder.* keys)."""
    import torch
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "backbone": args.backbone,
        "crop_size": args.crop_size,
        "patch_size": args.patch_size,
        "mask_ratio_pre": args.mask_ratio_pre,
        "mask_ratio_post": args.mask_ratio_post,
        "train_disasters": train_disasters,
    }, path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Temporal MAE pretraining")
    p.add_argument("--buildings_csv", default="data/processed/buildings_v2.csv",
                   help="Path to buildings_v2.csv")
    p.add_argument("--out_dir",       default="models/mae/pretrain_all",
                   help="Directory to save encoder checkpoint and logs")
    p.add_argument("--backbone",      default="vit_small",
                   choices=["vit_tiny", "vit_small", "vit_base"],
                   help="Encoder backbone (default: vit_small)")
    p.add_argument("--epochs",        type=int,   default=200,
                   help="Pretraining epochs (default: 200)")
    p.add_argument("--batch",         type=int,   default=64,
                   help="Batch size (default: 64)")
    p.add_argument("--lr",            type=float, default=1e-4,
                   help="Peak learning rate (default: 1e-4)")
    p.add_argument("--weight_decay",  type=float, default=0.05,
                   help="AdamW weight decay (default: 0.05)")
    p.add_argument("--mask_ratio_pre",  type=float, default=0.75,
                   help="Masking ratio for pre frame (default: 0.75)")
    p.add_argument("--mask_ratio_post", type=float, default=0.75,
                   help="Masking ratio for post frame (default: 0.75)")
    p.add_argument("--mae_objective",   type=str, default="v1",
                   choices=["v1", "v2"],
                   help="v1: reconstruct both frames (default). "
                        "v2: reconstruct post only (change-aligned; "
                        "use with mask_ratio_pre=0.5, mask_ratio_post=0.8)")
    p.add_argument("--train_disasters", type=str, default=None,
                   help="Comma-sep disaster IDs for LOWO (e.g. 'socal-fire'). "
                        "If None, uses all disasters.")
    p.add_argument("--crop_size",     type=int,   default=128,
                   help="Input crop size in pixels (default: 128)")
    p.add_argument("--patch_size",    type=int,   default=16,
                   help="MAE patch size (default: 16)")
    p.add_argument("--seed",          type=int,   default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--num_workers",   type=int,   default=0,
                   help="DataLoader workers (default: 0)")
    p.add_argument("--device",        type=str,   default=None,
                   choices=["cuda", "cpu", "auto"],
                   help="Device (default: auto)")
    p.add_argument("--run_id",        type=str,   default=None,
                   help="Optional run identifier (for logging only)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
