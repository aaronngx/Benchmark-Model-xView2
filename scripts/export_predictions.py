#!/usr/bin/env python3
"""
Export per-building predictions for a held-out disaster.

Loads a trained checkpoint, runs inference on all records belonging to the
specified disaster, and writes a CSV with columns:
  tile_id, building_id, y_true, y_pred

Used to feed predictions into bootstrap_tile_ci.py.

Usage:
  python scripts/export_predictions.py \
    --ckpt_path models/lowo/train_socal_test_santarosa/best.pt \
    --disaster santa-rosa-wildfire \
    --run_id lowo_train_socal_test_santarosa \
    --out_csv reports/lowo/preds_lowo_train_socal_test_santarosa.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def disaster_id(tile_id: str) -> str:
    return tile_id.rsplit("_", 1)[0]


def main() -> None:
    p = argparse.ArgumentParser(description="Export per-building predictions for a disaster")
    p.add_argument("--ckpt_path",  required=True,
                   help="Path to best.pt checkpoint")
    p.add_argument("--index_csv",  default="data/processed/index.csv")
    p.add_argument("--crops_dir",  default="data/processed/crops_oracle")
    p.add_argument("--disaster",   required=True,
                   help="Disaster ID to evaluate, e.g. santa-rosa-wildfire")
    p.add_argument("--run_id",     required=True,
                   help="Run ID used for the output filename label")
    p.add_argument("--out_csv",    default=None,
                   help="Output CSV path (default: reports/lowo/preds_<run_id>.csv)")
    p.add_argument("--model_type", default="six_channel")
    p.add_argument("--batch",      type=int, default=64)
    p.add_argument("--size",       type=int, default=128)
    args = p.parse_args()

    out_csv = Path(args.out_csv) if args.out_csv else \
              Path("reports/lowo") / f"preds_{args.run_id}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        print("torch not installed", file=sys.stderr)
        sys.exit(1)

    from disaster_bench.data.dataset import build_crop_records, CropDataset

    # Load all records and filter to the held-out disaster
    all_records = build_crop_records(args.index_csv, args.crops_dir)
    records = [r for r in all_records if disaster_id(r["tile_id"]) == args.disaster]

    if not records:
        print(f"ERROR: no records found for disaster '{args.disaster}'", file=sys.stderr)
        print(f"  Available disasters: {sorted({disaster_id(r['tile_id']) for r in all_records})}",
              file=sys.stderr)
        sys.exit(1)

    print(f"Disaster '{args.disaster}': {len(records)} buildings across "
          f"{len({r['tile_id'] for r in records})} tiles")

    # Dataset + loader
    def collate(batch):
        xs = torch.from_numpy(np.stack([b[0] for b in batch])).float()
        ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return xs, ys

    ds = CropDataset(records, size=args.size, augment=False, preload=True)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        collate_fn=collate, num_workers=0)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from disaster_bench.models.damage.classifiers import build_classifier
    model = build_classifier(args.model_type, num_classes=4).to(device)
    ckpt  = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {args.ckpt_path} (epoch {ckpt.get('epoch','?')})")

    # Inference
    all_preds = []
    with torch.no_grad():
        for x, _ in loader:
            all_preds.extend(model(x.to(device)).argmax(1).cpu().tolist())

    # Write CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["tile_id", "building_id", "y_true", "y_pred"])
        writer.writeheader()
        for i, (r, pred) in enumerate(zip(records, all_preds)):
            writer.writerow({
                "tile_id":     r["tile_id"],
                "building_id": r.get("uid", str(i)),
                "y_true":      r["label_idx"],
                "y_pred":      pred,
            })

    print(f"Saved {len(records)} predictions -> {out_csv}")

    # Quick summary
    correct = sum(r["label_idx"] == p for r, p in zip(records, all_preds))
    print(f"Overall accuracy: {correct}/{len(records)} = {correct/len(records):.3f}")


if __name__ == "__main__":
    main()
