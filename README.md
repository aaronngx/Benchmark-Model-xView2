# Wildfire Building Damage Benchmark

Per-building damage classification from pre/post satellite imagery.  
Covers **socal-fire** + **santa-rosa-wildfire** (xView2 dataset).

**Output classes:** `no-damage` | `minor-damage` | `major-damage` | `destroyed`

> For full documentation see [`PROJECT_XRAY.md`](PROJECT_XRAY.md).

---

## Tracks

| Track | Method | Best macro-F1 |
|-------|--------|--------------|
| Track 1 (oracle) | Oracle crops → 6-channel CNN (GPU batch) | **0.477** |
| Track 1 (deploy) | U-Net footprints → IoU match → CNN | *(needs footprint model)* |
| Track 2A | Abs-diff + SSIM + edge → thresholds | 0.184 |
| Track 2B | Pixel severity map → polygon aggregation | 0.157 |
| Track 3 | Track 1 ML + geometry guardrails | 0.464 |
| Track 4 | VLM (OpenAI/Gemini/Claude, ±geometry context) | *(needs API keys)* |

**Rule:** GT labels (`subtype`) are used **only for training and evaluation**, never at inference.

---

## Setup

```bash
# Install (editable)
py -m pip install -e ".[ml,config]"

# PyTorch CUDA (Python 3.14 requires nightly)
py -m pip install --force-reinstall torch torchvision \
  --index-url https://download.pytorch.org/whl/nightly/cu128
```

---

## Data Setup

```bash
# 1. Index the dataset (produces data/processed/index.csv)
python -m disaster_bench.cli build-index \
  --dataset_root test_images_labels_targets \
  --out_csv data/processed/index.csv

# 2. Generate oracle crops (produces data/processed/crops_oracle/)
python -m disaster_bench.cli make-oracle-crops \
  --index_csv data/processed/index.csv \
  --out_dir data/processed/crops_oracle
```

---

## Train a Damage Classifier

```bash
# Six-channel CNN (default, already trained → models/six_channel/best.pt)
python scripts/train_damage.py --model_type six_channel --out_dir models/six_channel

# Pre/Post/Diff CNN (9-channel)
python scripts/train_damage.py --model_type pre_post_diff --out_dir models/pre_post_diff

# Siamese CNN (dual-stream)
python scripts/train_damage.py --model_type siamese --out_dir models/siamese

# Centroid-patch CNN
python scripts/train_damage.py --model_type centroid_patch --out_dir models/centroid_patch

# Pixel-wise Siamese U-Net (Track 2B learned)
python scripts/train_siamese_unet.py --out_dir models/siamese_unet
```

---

## Run & Evaluate

```bash
# Run any track
python -m disaster_bench.cli run \
  --track track1 \
  --config configs/tracks/track1.yaml \
  --run_dir runs/my_run

# Tracks: track1 | track2a | track2b | track3 | vlm | deploy

# Evaluate (adds GT labels, computes all metrics)
python -m disaster_bench.cli eval-run \
  --run_dir runs/my_run \
  --index_csv data/processed/index.csv

# Compare all runs side-by-side
python scripts/compare_metrics.py
python scripts/compare_metrics.py --confusion   # + confusion matrices
```

---

## Outputs

| Path | Contents |
|------|----------|
| `data/processed/index.csv` | 381 tiles × (tile_id, pre_path, post_path, label_path) |
| `data/processed/crops_oracle/` | 8,498 buildings × (pre_bbox.png, post_bbox.png) |
| `data/processed/pred_instances/` | Predicted footprint JSONs (track1_deploy) |
| `runs/<run>/predictions.csv` | Per-building predictions + GT damage |
| `runs/<run>/metrics.json` | macro_F1, FEMA_F1, confusion_matrix, latency, coverage |
| `models/<name>/best.pt` | Best checkpoint (auto-saved during training) |

---

## Stack

Python 3.11+; `numpy`, `opencv-python`, `shapely`, `pillow`.  
Optional: `torch`, `torchvision`, `timm`, `pyyaml`.  
VLM optional: `openai`, `anthropic`, `google-generativeai`.
