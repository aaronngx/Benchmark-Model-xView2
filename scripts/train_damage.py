#!/usr/bin/env python3
"""
Train damage classifier on oracle crops.
Ref §2.1: Supervised (non-LLM) baselines.

Supports model types:
  six_channel   — [preRGB || postRGB] 6-ch CNN (default)
  pre_post_diff — [preRGB || postRGB || |diff|] 9-ch CNN
  siamese       — dual-stream encoder + fusion
  centroid_patch— fixed-size patch from polygon centroid (6-ch CNN input)

Usage:
    python scripts/train_damage.py \\
        --index_csv data/processed/index.csv \\
        --crops_dir data/processed/crops_oracle \\
        --out_dir   models/six_channel \\
        --model_type six_channel \\
        [--epochs 30] [--batch 32] [--lr 3e-4] [--size 128]
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np


def collate(batch):
    import torch
    xs = torch.from_numpy(np.stack([b[0] for b in batch])).float()
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return xs, ys


def _compute_val_f1(all_preds, all_labels):
    from collections import defaultdict
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for p, t in zip(all_preds, all_labels):
        if p == t:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    f1s = []
    for c in range(4):
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        rec  = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return f1s, float(np.mean(f1s))


def run(args: argparse.Namespace) -> None:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        print("torch not installed. Run: pip install torch")
        sys.exit(1)

    from disaster_bench.data.dataset import (
        DAMAGE_CLASSES, build_crop_records, train_val_split, CropDataset,
        NineChannelDataset,
    )
    from disaster_bench.models.damage.classifiers import build_classifier, save_checkpoint

    print(f"Model type: {args.model_type}")

    # --- Records ---
    if args.model_type == "centroid_patch":
        from disaster_bench.data.dataset import build_centroid_records
        print("Building centroid records from index...")
        records = build_centroid_records(args.index_csv)
    else:
        print("Building crop records...")
        records = build_crop_records(args.index_csv, args.crops_dir)

    label_dist = Counter(r["label"] for r in records)
    print(f"  Total buildings: {len(records)}")
    for cls in DAMAGE_CLASSES:
        print(f"  {cls:20s}: {label_dist.get(cls, 0)}")

    train_recs, val_recs = train_val_split(records, val_fraction=args.val_fraction)
    print(f"\nTrain: {len(train_recs)}  Val: {len(val_recs)}")

    # --- Datasets ---
    if args.model_type == "centroid_patch":
        # CentroidPatchDataset: loads full tile image and extracts centroid crops
        from disaster_bench.data.dataset import load_centroid_patch, LABEL2IDX
        import random

        class CentroidDataset:
            def __init__(self, recs, size=128, augment=False):
                self.recs    = recs
                self.size    = size
                self.augment = augment

            def __len__(self):
                return len(self.recs)

            def __getitem__(self, idx):
                from PIL import Image
                r = self.recs[idx]
                pre_img  = np.array(Image.open(r["pre_path"]).convert("RGB"))
                post_img = np.array(Image.open(r["post_path"]).convert("RGB"))
                x = load_centroid_patch(pre_img, post_img, r["cx"], r["cy"], self.size)
                if self.augment:
                    if random.random() > 0.5:
                        x = x[:, :, ::-1].copy()
                    if random.random() > 0.5:
                        x = x[:, ::-1, :].copy()
                return x, r["label_idx"]

            def class_weights(self):
                counts = np.zeros(len(DAMAGE_CLASSES), dtype=np.float32)
                for r in self.recs:
                    counts[r["label_idx"]] += 1
                counts = np.where(counts == 0, 1, counts)
                w = 1.0 / counts
                w /= w.sum()
                return w * len(DAMAGE_CLASSES)

        train_ds = CentroidDataset(train_recs, size=args.size, augment=True)
        val_ds   = CentroidDataset(val_recs,   size=args.size, augment=False)
        weights  = torch.from_numpy(train_ds.class_weights()).float()

    else:
        train_ds_base = CropDataset(train_recs, size=args.size, augment=True,  preload=True)
        val_ds_base   = CropDataset(val_recs,   size=args.size, augment=False, preload=True)
        weights       = torch.from_numpy(train_ds_base.class_weights()).float()

        if args.model_type == "pre_post_diff":
            train_ds = NineChannelDataset(train_ds_base)
            val_ds   = NineChannelDataset(val_ds_base)
        else:
            train_ds = train_ds_base
            val_ds   = val_ds_base

    print(f"Class weights: { {c: round(float(w),3) for c,w in zip(DAMAGE_CLASSES, weights)} }")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              collate_fn=collate, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              collate_fn=collate, num_workers=0)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = build_classifier(args.model_type, num_classes=4, dropout=args.dropout).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    weights     = weights.to(device)
    criterion   = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    best_val_f1 = -1.0
    best_path   = out_dir / "best.pt"

    if best_path.exists() and args.resume:
        print(f"Resuming from {best_path} ...")
        ckpt_r = torch.load(str(best_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt_r["model_state_dict"])
        best_val_f1 = ckpt_r.get("val_macro_f1", -1.0)
        start_epoch = ckpt_r.get("epoch", 0) + 1
        print(f"  Resuming at epoch {start_epoch}, best_f1={best_val_f1:.4f}")

    f1s, macro_f1 = [0.0]*4, 0.0  # for final save if 0 epochs

    for epoch in range(start_epoch, args.epochs + 1):
        # --- train ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        t0 = time.perf_counter()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += len(y)
        scheduler.step()
        ep_time = time.perf_counter() - t0

        # --- val ---
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                all_preds.extend(model(x).argmax(1).cpu().tolist())
                all_labels.extend(y.tolist())

        f1s, macro_f1 = _compute_val_f1(all_preds, all_labels)
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss/train_total:.4f} "
              f"train_acc={train_correct/train_total:.3f} | "
              f"val_macro_f1={macro_f1:.4f} "
              f"[{' '.join(f'{f:.3f}' for f in f1s)}]  {ep_time:.1f}s",
              flush=True)

        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            save_checkpoint(
                model, str(best_path),
                model_type=args.model_type,
                epoch=epoch,
                val_macro_f1=macro_f1,
                per_class_f1={DAMAGE_CLASSES[i]: round(f1s[i], 4) for i in range(4)},
                input_size=args.size,
            )
            print(f"  -> saved best.pt (macro_f1={macro_f1:.4f})")

    # Save final
    save_checkpoint(
        model, str(out_dir / "last.pt"),
        model_type=args.model_type,
        epoch=args.epochs,
        val_macro_f1=macro_f1,
        per_class_f1={DAMAGE_CLASSES[i]: round(f1s[i], 4) for i in range(4)},
        input_size=args.size,
    )
    print(f"\nDone. Best val macro_f1={best_val_f1:.4f}  -> {out_dir}/best.pt")


def main() -> None:
    p = argparse.ArgumentParser(description="Train damage classifier")
    p.add_argument("--index_csv",    default="data/processed/index.csv")
    p.add_argument("--crops_dir",    default="data/processed/crops_oracle")
    p.add_argument("--out_dir",      default="models/six_channel")
    p.add_argument("--model_type",   default="six_channel",
                   choices=["six_channel", "pre_post_diff", "siamese", "centroid_patch"],
                   help="Classifier architecture (Ref §2.1)")
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch",        type=int,   default=32)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--size",         type=int,   default=128)
    p.add_argument("--dropout",      type=float, default=0.4)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--device",       default=None, help="cuda | cpu | auto")
    p.add_argument("--resume",       action="store_true", help="Resume from best.pt if present")
    run(p.parse_args())


if __name__ == "__main__":
    main()
