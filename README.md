# Wildfire Damage Benchmark

SoCal wildfire building damage benchmark. Takes **pre + post** imagery and outputs **per-building damage** on:

`no-damage` | `minor-damage` | `major-damage` | `destroyed`

## Tracks

- **Track 1** — End-to-end ML: predicted footprints → crops → damage classifier → per-building label
- **Track 2A** — Unsupervised numeric baseline (change signals + thresholds)
- **Track 2B** — Damage-map aggregation (pixel map + rules)
- **Track 3** — Hybrid: ML/VLM + geometry guardrails, flag when inconsistent

**Rule:** GT damage labels (`subtype`) are used **only for training and evaluation**, never at prediction time.

## Setup

```bash
pip install -e .
# optional: pip install -e ".[ml,config]"
```

## Dataset

Place data under `data/raw/test_images_labels_targets/` (or point CLI to your path).  
If your dataset is at repo root as `test_images_labels_targets/`, use  
`--dataset_root test_images_labels_targets` when running `build-index`.  
The scanner builds `data/processed/index.csv` from whatever layout you have (e.g. `test/labels/*.json`).

## Commands

```bash
# Build dataset index
python -m disaster_bench.cli build-index --dataset_root data/raw/test_images_labels_targets --out_csv data/processed/index.csv

# Oracle paired crops (GT polygons)
python -m disaster_bench.cli make-oracle-crops --index_csv data/processed/index.csv --out_dir data/processed/crops_oracle

# Run a track
python -m disaster_bench.cli run --track track1 --config configs/tracks/track1.yaml
python -m disaster_bench.cli run --track track2a --config configs/tracks/track2a.yaml
python -m disaster_bench.cli run --track track3 --config configs/tracks/track3.yaml

# Evaluate a run (adds gt_damage, computes metrics)
python -m disaster_bench.cli eval-run --run_dir runs/<run_id> --index_csv data/processed/index.csv
```

## Outputs

- **Index:** `data/processed/index.csv` — tile_id, pre_path, post_path, label_json_path
- **Oracle crops:** `data/processed/crops_oracle/<tile_id>/<uid>/pre_bbox.png`, `post_bbox.png`
- **Runs:** `runs/<run_id>/predictions.csv`, `runs/<run_id>/metrics.json` (macro_f1, per_class_f1, coverage)

## Tech

Python 3.11; numpy, opencv-python, shapely, pillow. Optional: torch, torchvision, timm, pyyaml, pydantic.
