#!/usr/bin/env python3
"""
FEMA 3-class evaluation for benchmark checkpoints.

Loads best.pt for each method, runs inference on the val split,
applies FEMA remapping (intact / damaged / destroyed), and saves:
  reports/fema3/<method>_fema3.json

Usage:
  python scripts/eval_fema_3class.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

METHODS = {
    "baseline":          "models/baseline_s1/best.pt",
    "sampler_capped":    "models/sampler_capped_s1/best.pt",
    "sampler_noweights": "models/sampler_noweights_s1/best.pt",
}

INDEX_CSV  = "data/processed/index.csv"
CROPS_DIR  = "data/processed/crops_oracle"
MODEL_TYPE = "six_channel"
BATCH_SIZE = 64
IMG_SIZE   = 128
OUT_DIR    = Path("reports/fema3")

# FEMA 3-class: integer index → FEMA label
# 4-class mapping: 0=no-damage, 1=minor-damage, 2=major-damage, 3=destroyed
FEMA_IDX = {0: "intact", 1: "damaged", 2: "damaged", 3: "destroyed"}
FEMA_CLASSES = ["intact", "damaged", "destroyed"]
DAMAGE_CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]


def collate(batch):
    import torch
    xs = torch.from_numpy(np.stack([b[0] for b in batch])).float()
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return xs, ys


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return round(prec, 4), round(rec, 4), round(f1, 4)


def eval_method(name: str, ckpt_path: str) -> dict:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        print("torch not installed", file=sys.stderr)
        sys.exit(1)

    from disaster_bench.data.dataset import build_crop_records, train_val_split, CropDataset
    from disaster_bench.models.damage.classifiers import build_classifier

    ckpt_p = Path(ckpt_path)
    if not ckpt_p.exists():
        print(f"ERROR: checkpoint not found: {ckpt_p}", file=sys.stderr)
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load val split (same split as training)
    records = build_crop_records(INDEX_CSV, CROPS_DIR)
    _, val_recs = train_val_split(records)
    val_ds = CropDataset(val_recs, size=IMG_SIZE, augment=False, preload=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate, num_workers=0)

    # Load model
    model = build_classifier(MODEL_TYPE, num_classes=4).to(device)
    ckpt  = torch.load(str(ckpt_p), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    ckpt_epoch = ckpt.get("epoch", "?")
    model.eval()

    # Inference
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            preds = model(x).argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(y.tolist())

    # Apply FEMA remap (integer → fema string)
    fema_pred  = [FEMA_IDX[p] for p in all_preds]
    fema_true  = [FEMA_IDX[t] for t in all_labels]

    # Confusion matrix (rows=true, cols=pred)
    cls_to_i = {c: i for i, c in enumerate(FEMA_CLASSES)}
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(fema_true, fema_pred):
        cm[cls_to_i[t], cls_to_i[p]] += 1

    # Per-class precision / recall / F1
    per_class = {}
    for i, cls in enumerate(FEMA_CLASSES):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum()) - tp
        fn = int(cm[i, :].sum()) - tp
        prec, rec, f1 = precision_recall_f1(tp, fp, fn)
        per_class[cls] = {"precision": prec, "recall": rec, "f1": f1}

    macro_f1 = round(sum(v["f1"] for v in per_class.values()) / len(FEMA_CLASSES), 4)

    result = {
        "method":        name,
        "checkpoint":    str(ckpt_p),
        "checkpoint_epoch": ckpt_epoch,
        "n_val":         len(all_labels),
        "fema_macro_f1": macro_f1,
        "fema_per_class": per_class,
        "confusion_matrix": {
            "classes": FEMA_CLASSES,
            "matrix":  cm.tolist(),
        },
    }

    print(f"\n[{name}] checkpoint epoch={ckpt_epoch}, val_n={len(all_labels)}")
    print(f"  FEMA macro F1 = {macro_f1:.4f}")
    for cls, m in per_class.items():
        print(f"  {cls:12s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")
    print("  Confusion matrix (rows=true, cols=pred):")
    print("  " + "  ".join(f"{c:10s}" for c in FEMA_CLASSES))
    for i, row in enumerate(cm.tolist()):
        print(f"  {FEMA_CLASSES[i]:10s}  " + "  ".join(f"{v:10d}" for v in row))

    return result


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for method_name, ckpt_path in METHODS.items():
        result = eval_method(method_name, ckpt_path)
        out_path = OUT_DIR / f"{method_name}_fema3.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  -> saved {out_path}")


if __name__ == "__main__":
    main()
