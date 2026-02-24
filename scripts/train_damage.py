#!/usr/bin/env python3
"""
Train damage classifier on oracle crops.
Ref §2.1: Supervised (non-LLM) baselines.

Supports model types:
  six_channel   — [preRGB || postRGB] 6-ch CNN (default)
  pre_post_diff — [preRGB || postRGB || |diff|] 9-ch CNN
  siamese       — dual-stream encoder + fusion
  centroid_patch— fixed-size patch from polygon centroid (6-ch CNN input)

New flags (all default to off/scratch — existing behavior is fully preserved):
  --seed                     global RNG seed for reproducibility (default: 42)
  --run_id                   write val_metrics.jsonl to runs/<run_id>/
  --use_sampler              WeightedRandomSampler for train loader (Step 1)
  --log_batch_class_counts   log first 50 train-batch label histograms, epoch 1 (Step 2B)
  --weight_mode              normalized_invfreq (default) | capped_floored | none (Step 3)
  --w_min / --w_max          clipping bounds for capped_floored (Step 3)
  --use_hard_negative_mining boost no-dmg hard negatives each epoch (Step 4)
  --hnm_mult                 boost multiplier for HNM (Step 4)
  --two_stage                Stage1=3-class + Stage2=2-class minor/major (Step 5)
  --init_mode                scratch (default) | pretrained (Step 6)
  --pretrained_ckpt_path     encoder checkpoint for pretrained init (Step 6)
  --cv_folds_path            path to cv_folds JSON (make_cv_folds.py output)
  --cv_fold                  fold to use as val set (0..k-1); requires cv_folds_path

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
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import random

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def seed_all(seed: int) -> None:
    """Pin all RNG sources for reproducible runs."""
    import torch as _torch
    random.seed(seed)
    np.random.seed(seed)
    _torch.manual_seed(seed)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(seed)
    _torch.backends.cudnn.benchmark     = False
    _torch.backends.cudnn.deterministic = True


def collate(batch):
    import torch
    xs = torch.from_numpy(np.stack([b[0] for b in batch])).float()
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return xs, ys


def _compute_confusion_matrix(all_preds, all_labels, num_classes=4):
    """cm[true_class][pred_class] = count."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(all_labels, all_preds):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t][p] += 1
    return cm


def _compute_val_metrics(all_preds, all_labels, num_classes=4):
    """Returns (f1s, macro_f1, precs, recs) — one value per class."""
    from collections import defaultdict
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for p, t in zip(all_preds, all_labels):
        if p == t:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    f1s, precs, recs = [], [], []
    for c in range(num_classes):
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        rec  = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
        precs.append(prec)
        recs.append(rec)
    return f1s, float(np.mean(f1s)), precs, recs


def _compute_class_weights(records, num_classes, weight_mode, w_min, w_max):
    """Normalized inverse-frequency weights, optionally capped/floored, or all-ones."""
    if weight_mode == "none":
        return np.ones(num_classes, dtype=np.float32)
    counts = np.zeros(num_classes, dtype=np.float32)
    for r in records:
        counts[r["label_idx"]] += 1
    counts = np.where(counts == 0, 1, counts)
    w = 1.0 / counts
    w /= w.sum()
    w *= num_classes
    if weight_mode == "capped_floored":
        w = np.clip(w, w_min, w_max)
    return w


def _get_train_records(ds):
    """Return the underlying record list from any dataset variant."""
    if hasattr(ds, "records"):          # CropDataset
        return ds.records
    if hasattr(ds, "_base"):            # NineChannelDataset
        return ds._base.records
    if hasattr(ds, "recs"):             # inline CentroidDataset
        return ds.recs
    raise AttributeError(f"Cannot find records on {type(ds)}")


def _base_sample_weights(records, class_weights_np):
    """Per-sample weight array from per-class weights (indexed by label_idx)."""
    return np.array(
        [float(class_weights_np[r["label_idx"]]) for r in records],
        dtype=np.float32,
    )


def _build_sampler(sample_weights_np, num_samples):
    """Build WeightedRandomSampler with replacement."""
    import torch
    from torch.utils.data import WeightedRandomSampler
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights_np).float(),
        num_samples=num_samples,
        replacement=True,
    )


def _load_encoder_weights(model, ckpt_path: str, device):
    """Load encoder submodule weights from checkpoint. Head stays random."""
    import torch
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_state = ckpt.get("model_state_dict", ckpt)

    if hasattr(model, "encoder"):
        enc_attr = "encoder"
    elif hasattr(model, "stream"):
        enc_attr = "stream"
    else:
        print(f"  [init] WARNING: no 'encoder' or 'stream' found; skipping pretrained init.")
        return

    encoder = getattr(model, enc_attr)
    prefix = enc_attr + "."
    enc_state = {k[len(prefix):]: v for k, v in ckpt_state.items() if k.startswith(prefix)}

    if not enc_state:
        avail = list(ckpt_state.keys())[:5]
        print(f"  [init] WARNING: no keys with prefix '{prefix}'. "
              f"Available (first 5): {avail}")
        return

    missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
    print(f"  [init] Loaded encoder ('{enc_attr}') from {ckpt_path} | keys={len(enc_state)}")
    if missing:
        print(f"  [init] Missing keys  : {missing}")
    if unexpected:
        print(f"  [init] Unexpected keys: {unexpected}")
    print(f"  [init] Head layers remain randomly initialized.")


def _two_stage_predict(x_batch, model1, model2, device):
    """
    Stage 1: 3-class (0=no-dmg, 1=damaged, 2=destroyed).
    Stage 2: 2-class (0=minor, 1=major), applied only where Stage 1 predicts 'damaged'.
    Returns a list of 4-class predictions (0=no-dmg,1=minor,2=major,3=destroyed).
    """
    s1_preds = model1(x_batch).argmax(1).cpu().tolist()
    s2_idxs = [i for i, p in enumerate(s1_preds) if p == 1]
    s2_map = {}
    if s2_idxs:
        import torch
        x_s2 = x_batch[torch.tensor(s2_idxs)]
        s2_out = model2(x_s2).argmax(1).cpu().tolist()  # 0=minor, 1=major
        for j, orig_i in enumerate(s2_idxs):
            s2_map[orig_i] = s2_out[j] + 1  # minor→1, major→2
    final = []
    for i, p1 in enumerate(s1_preds):
        if p1 == 0:
            final.append(0)       # no-damage
        elif p1 == 2:
            final.append(3)       # destroyed
        else:
            final.append(s2_map[i])  # minor or major
    return final


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

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

    # Validate pretrained args
    if args.init_mode == "pretrained" and not args.pretrained_ckpt_path:
        print("Error: --pretrained_ckpt_path is required when --init_mode pretrained",
              file=sys.stderr)
        sys.exit(1)

    # Seed for reproducibility (sampler, weight init, augmentation)
    seed_all(args.seed)
    print(f"Seed: {args.seed}")

    # Step 0: run output directory
    run_dir = Path("runs") / args.run_id if args.run_id else None
    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model type: {args.model_type}")

    # -----------------------------------------------------------------------
    # Records
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Train / val split: CV fold or default 80/20
    # -----------------------------------------------------------------------
    if args.cv_folds_path and args.cv_fold is not None:
        with open(args.cv_folds_path) as _f:
            _fold_data = json.load(_f)
        _tile_to_fold = _fold_data["tile_to_fold"]
        _val_fold     = args.cv_fold
        _val_tiles    = {t for t, fi in _tile_to_fold.items() if fi == _val_fold}
        train_recs = [r for r in records if r["tile_id"] not in _val_tiles]
        val_recs   = [r for r in records if r["tile_id"] in _val_tiles]
        print(f"\nCV fold {_val_fold}/{_fold_data['k']}: "
              f"train={len(train_recs)}  val={len(val_recs)}")
        tr_dist = Counter(r["label"] for r in train_recs)
        va_dist = Counter(r["label"] for r in val_recs)
        print("  Train:", {cls: tr_dist.get(cls, 0) for cls in DAMAGE_CLASSES})
        print("  Val:  ", {cls: va_dist.get(cls, 0) for cls in DAMAGE_CLASSES})
    else:
        train_recs, val_recs = train_val_split(records, val_fraction=args.val_fraction)
        print(f"\nTrain: {len(train_recs)}  Val: {len(val_recs)}")

    # -----------------------------------------------------------------------
    # Step 5: two-stage label remapping
    # -----------------------------------------------------------------------
    S1_REMAP  = {0: 0, 1: 1, 2: 1, 3: 2}   # no-dmg=0, damaged(minor+major)=1, dest=2
    S2_REMAP  = {1: 0, 2: 1}                # minor=0, major=1
    S2_FILTER = {1, 2}

    if args.two_stage:
        s1_train_recs = [dict(r, label_idx=S1_REMAP[r["label_idx"]]) for r in train_recs]
        s2_train_recs = [dict(r, label_idx=S2_REMAP[r["label_idx"]])
                         for r in train_recs if r["label_idx"] in S2_FILTER]
        print(f"  Stage1 train: {len(s1_train_recs)} (3-class)  "
              f"Stage2 train: {len(s2_train_recs)} (2-class minor/major only)")

    # -----------------------------------------------------------------------
    # Datasets
    # -----------------------------------------------------------------------
    if args.model_type == "centroid_patch":
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

        if args.two_stage:
            train_ds    = CentroidDataset(s1_train_recs, size=args.size, augment=True)
            s2_train_ds = CentroidDataset(s2_train_recs, size=args.size, augment=True)
        else:
            train_ds    = CentroidDataset(train_recs, size=args.size, augment=True)
            s2_train_ds = None
        val_ds      = CentroidDataset(val_recs, size=args.size, augment=False)
        train_ds_base = None

    else:
        if args.two_stage:
            train_ds_base    = CropDataset(s1_train_recs, size=args.size, augment=True,  preload=True)
            s2_train_ds_base = CropDataset(s2_train_recs, size=args.size, augment=True,  preload=True)
        else:
            train_ds_base    = CropDataset(train_recs, size=args.size, augment=True,  preload=True)
            s2_train_ds_base = None
        val_ds_base = CropDataset(val_recs, size=args.size, augment=False, preload=True)

        if args.model_type == "pre_post_diff":
            train_ds    = NineChannelDataset(train_ds_base)
            s2_train_ds = NineChannelDataset(s2_train_ds_base) if s2_train_ds_base else None
            val_ds      = NineChannelDataset(val_ds_base)
        else:
            train_ds    = train_ds_base
            s2_train_ds = s2_train_ds_base
            val_ds      = val_ds_base

    # -----------------------------------------------------------------------
    # Step 3: loss weights
    # -----------------------------------------------------------------------
    if args.two_stage:
        # Stage 1: 3-class weights from s1_train_recs
        s1_w_np = _compute_class_weights(s1_train_recs, 3,
                                          args.weight_mode, args.w_min, args.w_max)
        weights = torch.from_numpy(s1_w_np).float()
        print(f"Stage1 weights (3-class, {args.weight_mode}): "
              f"no-dmg={weights[0]:.4f}  damaged={weights[1]:.4f}  dest={weights[2]:.4f}")
        # Stage 2: 2-class weights from s2_train_recs
        s2_w_np = _compute_class_weights(s2_train_recs, 2,
                                          args.weight_mode, args.w_min, args.w_max)
        s2_weights = torch.from_numpy(s2_w_np).float()
        print(f"Stage2 weights (2-class, {args.weight_mode}): "
              f"minor={s2_weights[0]:.4f}  major={s2_weights[1]:.4f}")
        # Base (unclipped) s1 weights for sampler
        s1_base_w_np = _compute_class_weights(s1_train_recs, 3,
                                               "normalized_invfreq", 0.0, 1e9)
    else:
        # 4-class weights (existing behavior)
        raw_w_np = _compute_class_weights(train_recs, 4,
                                           args.weight_mode, args.w_min, args.w_max)
        weights  = torch.from_numpy(raw_w_np).float()
        print(f"Class weights ({args.weight_mode}): "
              f"{ {c: round(float(w), 4) for c, w in zip(DAMAGE_CLASSES, weights)} }")
        # Base (unclipped) weights for sampler
        base_w_np = _compute_class_weights(train_recs, 4, "normalized_invfreq", 0.0, 1e9)

    # -----------------------------------------------------------------------
    # Step 1: DataLoaders (optional WeightedRandomSampler)
    # -----------------------------------------------------------------------
    sample_weights_np = None   # kept in scope for HNM (Step 4)
    train_records_list = _get_train_records(train_ds)

    # Auto-enable sampler if HNM requested without sampler
    need_sampler = bool(args.use_sampler) or bool(args.use_hard_negative_mining)
    if args.use_hard_negative_mining and not args.use_sampler:
        print("[HNM] use_sampler=0 but use_hard_negative_mining=1; "
              "switching to sampler automatically.")

    if need_sampler:
        base_w = s1_base_w_np if args.two_stage else base_w_np
        sample_weights_np = _base_sample_weights(train_records_list, base_w)
        sampler = _build_sampler(sample_weights_np, len(train_records_list))
        train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                                  collate_fn=collate, num_workers=0)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                  collate_fn=collate, num_workers=0)

    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            collate_fn=collate, num_workers=0)

    # Stage 2 DataLoader
    if args.two_stage and s2_train_ds is not None:
        s2_records_list = _get_train_records(s2_train_ds)
        if args.use_sampler:
            s2_base_w = _compute_class_weights(s2_train_recs, 2, "normalized_invfreq", 0.0, 1e9)
            s2_sample_w = _base_sample_weights(s2_records_list, s2_base_w)
            s2_sampler = _build_sampler(s2_sample_w, len(s2_records_list))
            s2_train_loader = DataLoader(s2_train_ds, batch_size=args.batch, sampler=s2_sampler,
                                         collate_fn=collate, num_workers=0)
        else:
            s2_train_loader = DataLoader(s2_train_ds, batch_size=args.batch, shuffle=True,
                                         collate_fn=collate, num_workers=0)
    else:
        s2_train_loader = None

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    if args.two_stage:
        model  = build_classifier(args.model_type, num_classes=3, dropout=args.dropout).to(device)
        model2 = build_classifier(args.model_type, num_classes=2, dropout=args.dropout).to(device)
        num_classes_s1 = 3
        num_classes_s2 = 2
    else:
        model  = build_classifier(args.model_type, num_classes=4, dropout=args.dropout).to(device)
        model2 = None
        num_classes_s1 = 4
        num_classes_s2 = None

    print(f"Parameters (Stage1): {sum(p.numel() for p in model.parameters()):,}")
    if model2 is not None:
        print(f"Parameters (Stage2): {sum(p.numel() for p in model2.parameters()):,}")

    # Step 6: pretrained encoder init
    if args.init_mode == "pretrained":
        _load_encoder_weights(model, args.pretrained_ckpt_path, device)
        if model2 is not None:
            _load_encoder_weights(model2, args.pretrained_ckpt_path, device)

    # -----------------------------------------------------------------------
    # Loss / optimizer / scheduler
    # -----------------------------------------------------------------------
    if args.weight_mode == "none":
        criterion = torch.nn.CrossEntropyLoss()
        print("Criterion: unweighted CrossEntropyLoss (weight_mode=none)")
    else:
        weights = weights.to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.two_stage and model2 is not None:
        if args.weight_mode == "none":
            criterion2 = torch.nn.CrossEntropyLoss()
        else:
            s2_weights = s2_weights.to(device)
            criterion2 = torch.nn.CrossEntropyLoss(weight=s2_weights)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=args.epochs)

    # -----------------------------------------------------------------------
    # Checkpoint paths
    # -----------------------------------------------------------------------
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path  = out_dir / "best.pt"
    last_path  = out_dir / "last.pt"
    best2_path = out_dir / "best_s2.pt"
    last2_path = out_dir / "last_s2.pt"

    start_epoch = 1
    best_val_f1 = -1.0

    if best_path.exists() and args.resume:
        print(f"Resuming from {best_path} ...")
        ckpt_r = torch.load(str(best_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt_r["model_state_dict"])
        best_val_f1 = ckpt_r.get("val_macro_f1", -1.0)
        start_epoch = ckpt_r.get("epoch", 0) + 1
        print(f"  Resuming at epoch {start_epoch}, best_f1={best_val_f1:.4f}")
        if args.two_stage and model2 is not None and best2_path.exists():
            ckpt_r2 = torch.load(str(best2_path), map_location=device, weights_only=False)
            model2.load_state_dict(ckpt_r2["model_state_dict"])
            print(f"  Loaded Stage2 from {best2_path}")

    f1s, macro_f1 = [0.0] * 4, 0.0

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs + 1):

        # --- Stage 1 train ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        t0 = time.perf_counter()

        # Step 2B: batch histogram logging (epoch 1 only, first 50 batches)
        _do_log = (args.log_batch_class_counts and epoch == 1 and run_dir is not None)
        _batch_log = run_dir / "train_batch_label_hist_epoch1.jsonl" if run_dir else None

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item() * len(y)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total   += len(y)

            if _do_log and batch_idx < 50:
                counts = [int((y == c).sum().item()) for c in range(4)]
                with open(_batch_log, "a", encoding="utf-8") as bf:
                    bf.write(json.dumps({"batch_idx": batch_idx, "counts": counts}) + "\n")

        # --- Stage 2 train (two_stage only) ---
        s2_loss_avg = None
        if args.two_stage and model2 is not None and s2_train_loader is not None:
            model2.train()
            s2_loss_sum, s2_total = 0.0, 0
            for x2, y2 in s2_train_loader:
                x2, y2 = x2.to(device), y2.to(device)
                optimizer2.zero_grad()
                logits2 = model2(x2)
                loss2   = criterion2(logits2, y2)
                loss2.backward()
                optimizer2.step()
                s2_loss_sum += loss2.item() * len(y2)
                s2_total    += len(y2)
            s2_loss_avg = s2_loss_sum / max(s2_total, 1)
            scheduler2.step()

        scheduler.step()
        ep_time = time.perf_counter() - t0

        # --- Val ---
        model.eval()
        if model2 is not None:
            model2.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                if args.two_stage and model2 is not None:
                    all_preds.extend(_two_stage_predict(x, model, model2, device))
                else:
                    all_preds.extend(model(x).argmax(1).cpu().tolist())
                all_labels.extend(y.tolist())

        f1s, macro_f1, precs, recs_pc = _compute_val_metrics(all_preds, all_labels, num_classes=4)

        # --- Print epoch summary ---
        s2_info = f" s2_loss={s2_loss_avg:.4f}" if s2_loss_avg is not None else ""
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss/max(train_total,1):.4f} "
            f"train_acc={train_correct/max(train_total,1):.3f}"
            f"{s2_info} | "
            f"val_macro_f1={macro_f1:.4f} "
            f"[{' '.join(f'{f:.3f}' for f in f1s)}]  {ep_time:.1f}s",
            flush=True,
        )

        # --- Step 0: extended val metrics block ---
        cm = _compute_confusion_matrix(all_preds, all_labels, num_classes=4)
        NO, MINOR, MAJOR = 0, 1, 2
        N_no             = int(cm[NO].sum())
        FP_minor_from_no = int(cm[NO][MINOR])
        FP_major_from_no = int(cm[NO][MAJOR])
        rate_minor       = 1000.0 * FP_minor_from_no / max(N_no, 1)
        rate_major       = 1000.0 * FP_major_from_no / max(N_no, 1)
        pred_minor       = int(cm[:, MINOR].sum())
        pred_major       = int(cm[:, MAJOR].sum())

        print(f"  FP/1k-no: minor={rate_minor:.1f}  major={rate_major:.1f} "
              f"| pred_minor={pred_minor}  pred_major={pred_major}")
        cls_names = ["no", "minor", "major", "dest"]
        print("  " + "  ".join(
            f"{n}: P={precs[i]:.3f} R={recs_pc[i]:.3f} F1={f1s[i]:.3f}"
            for i, n in enumerate(cls_names)
        ), flush=True)

        # --- Step 0: write JSONL ---
        if run_dir is not None:
            entry = {
                "epoch":                 epoch,
                "fp_per_1000_no_minor":  round(rate_minor, 3),
                "fp_per_1000_no_major":  round(rate_major, 3),
                "pred_minor":            pred_minor,
                "pred_major":            pred_major,
                "macro_f1":              round(macro_f1, 4),
                "f1_no":                 round(f1s[0], 4),
                "f1_minor":              round(f1s[1], 4),
                "f1_major":              round(f1s[2], 4),
                "f1_dest":               round(f1s[3], 4),
                "prec_no":               round(precs[0], 4),
                "prec_minor":            round(precs[1], 4),
                "prec_major":            round(precs[2], 4),
                "prec_dest":             round(precs[3], 4),
                "rec_no":                round(recs_pc[0], 4),
                "rec_minor":             round(recs_pc[1], 4),
                "rec_major":             round(recs_pc[2], 4),
                "rec_dest":              round(recs_pc[3], 4),
            }
            with open(run_dir / "val_metrics.jsonl", "a", encoding="utf-8") as jf:
                jf.write(json.dumps(entry) + "\n")

        # --- Save best checkpoint ---
        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            save_checkpoint(
                model, str(best_path),
                model_type=args.model_type,
                epoch=epoch,
                val_macro_f1=macro_f1,
                per_class_f1={DAMAGE_CLASSES[i]: round(f1s[i], 4) for i in range(4)},
                input_size=args.size,
                num_classes=num_classes_s1,
            )
            print(f"  -> saved best.pt (macro_f1={macro_f1:.4f})")
            if args.two_stage and model2 is not None:
                save_checkpoint(
                    model2, str(best2_path),
                    model_type=args.model_type,
                    epoch=epoch,
                    val_macro_f1=macro_f1,
                    per_class_f1={},
                    input_size=args.size,
                    num_classes=num_classes_s2,
                )

        # --- Step 4: Hard Negative Mining ---
        if args.use_hard_negative_mining:
            model.eval()
            hnm_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=False,
                                    collate_fn=collate, num_workers=0)
            all_train_preds = []
            with torch.no_grad():
                for xb, _ in hnm_loader:
                    all_train_preds.extend(model(xb.to(device)).argmax(1).cpu().tolist())

            # Reset to base weights each epoch, then boost hard negatives
            base_w = s1_base_w_np if args.two_stage else base_w_np
            sample_weights_np = _base_sample_weights(train_records_list, base_w)
            hard_neg_count = 0
            for i, (r, pred) in enumerate(zip(train_records_list, all_train_preds)):
                lbl = r["label_idx"]
                if args.two_stage:
                    # Stage1 labels: 0=no-dmg, 1=damaged, 2=dest
                    # Hard negative: GT=no-dmg (0) predicted as damaged (1)
                    if lbl == 0 and pred == 1:
                        sample_weights_np[i] *= args.hnm_mult
                        hard_neg_count += 1
                else:
                    # 4-class: GT=0 (no-dmg) predicted as minor(1) or major(2)
                    if lbl == 0 and pred in (1, 2):
                        sample_weights_np[i] *= args.hnm_mult
                        hard_neg_count += 1

            print(f"  [HNM] boosted {hard_neg_count} hard negatives "
                  f"(mult={args.hnm_mult:.1f})")
            new_sampler = _build_sampler(sample_weights_np, len(train_records_list))
            train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=new_sampler,
                                      collate_fn=collate, num_workers=0)

    # -----------------------------------------------------------------------
    # Save final checkpoints
    # -----------------------------------------------------------------------
    save_checkpoint(
        model, str(last_path),
        model_type=args.model_type,
        epoch=args.epochs,
        val_macro_f1=macro_f1,
        per_class_f1={DAMAGE_CLASSES[i]: round(f1s[i], 4) for i in range(4)},
        input_size=args.size,
        num_classes=num_classes_s1,
    )
    if args.two_stage and model2 is not None:
        save_checkpoint(
            model2, str(last2_path),
            model_type=args.model_type,
            epoch=args.epochs,
            val_macro_f1=macro_f1,
            per_class_f1={},
            input_size=args.size,
            num_classes=num_classes_s2,
        )
    print(f"\nDone. Best val macro_f1={best_val_f1:.4f}  -> {out_dir}/best.pt")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Train damage classifier")

    # Existing args (unchanged defaults)
    p.add_argument("--index_csv",    default="data/processed/index.csv")
    p.add_argument("--crops_dir",    default="data/processed/crops_oracle")
    p.add_argument("--out_dir",      default="models/six_channel")
    p.add_argument("--model_type",   default="six_channel",
                   choices=["six_channel", "pre_post_diff", "siamese", "centroid_patch"])
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch",        type=int,   default=32)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--size",         type=int,   default=128)
    p.add_argument("--dropout",      type=float, default=0.4)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--device",       default=None, help="cuda | cpu | auto")
    p.add_argument("--resume",       action="store_true")

    # Reproducibility
    p.add_argument("--seed",         type=int, default=42,
                   help="Global RNG seed (random, numpy, torch, cuda)")

    # Step 0 — tracking
    p.add_argument("--run_id",       type=str, default=None,
                   help="Write val_metrics.jsonl to runs/<run_id>/")

    # Step 1 — sampler
    p.add_argument("--use_sampler",  type=int, choices=[0, 1], default=0,
                   help="Use WeightedRandomSampler (replaces shuffle=True)")

    # Step 2B — batch histogram
    p.add_argument("--log_batch_class_counts", type=int, choices=[0, 1], default=0,
                   help="Log first 50 train-batch label histograms epoch 1 (needs --run_id)")

    # Step 3 — weight mode
    p.add_argument("--weight_mode",  type=str, default="normalized_invfreq",
                   choices=["normalized_invfreq", "capped_floored", "none"])
    p.add_argument("--w_min",        type=float, default=0.25,
                   help="Min loss weight for capped_floored")
    p.add_argument("--w_max",        type=float, default=5.0,
                   help="Max loss weight for capped_floored")

    # Step 4 — hard negative mining
    p.add_argument("--use_hard_negative_mining", type=int, choices=[0, 1], default=0)
    p.add_argument("--hnm_mult",     type=float, default=5.0,
                   help="Sampling weight multiplier for hard negatives")

    # Step 5 — two-stage
    p.add_argument("--two_stage",    type=int, choices=[0, 1], default=0,
                   help="Stage1=3-class then Stage2=2-class (minor/major)")

    # Step 6 — pretrained encoder
    p.add_argument("--init_mode",    type=str, default="scratch",
                   choices=["scratch", "pretrained"])
    p.add_argument("--pretrained_ckpt_path", type=str, default=None,
                   help="Encoder checkpoint path (required if init_mode=pretrained)")

    # CV — fold-based train/val split (overrides default 80/20 when both are given)
    p.add_argument("--cv_folds_path", type=str, default=None,
                   help="Path to cv_folds JSON (output of make_cv_folds.py)")
    p.add_argument("--cv_fold",       type=int, default=None,
                   help="Which fold to hold out as val (0..k-1)")

    run(p.parse_args())


if __name__ == "__main__":
    main()
