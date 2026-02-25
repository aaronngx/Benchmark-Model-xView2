# Never-Miss Mode ("FN=0")

## What it means

"Never-miss" mode targets **zero false negatives (FN=0) for minor-damage and
major-damage buildings** on the held-out test fold.  Instead of predicting the
argmax class, the model applies a **per-class probability threshold**: any
building whose `p(minor) >= tau_minor` is flagged as minor, and any building
whose `p(major) >= tau_major` is flagged as major, even if no-damage has the
highest raw probability.

The thresholds `tau_minor` and `tau_major` are chosen so that **every positive
in the calibration set has probability above the threshold** — guaranteeing
recall=1.0 on calibration.

## Why nested calibration is needed (no leakage)

A naive approach fits the threshold on the same data used for checkpoint
selection (e.g. the outer CV fold), then reports recall=1.0 on that same fold.
This is **leakage**: the threshold was optimised on the test set.

Nested calibration avoids this:

```
All tiles
├── Outer test fold (tile-grouped, ~20% of tiles) — NEVER SEEN until eval
└── Outer train tiles
    ├── train_fit (~80% of outer-train tiles) — used for model training
    └── calib set (~20% of outer-train tiles) — used for threshold fitting only
```

- Checkpoint selection uses `calib` macro-F1.
- Thresholds are fit on `calib` predictions (recall=1.0 is verified there).
- Final metrics (argmax + thresholded) are reported on the **outer test fold**,
  which was never used for training, validation, or threshold fitting.

## The trade-off

Lowering thresholds increases recall but also increases **false positives**.
FP/1k-no (false minor/major predictions per 1000 no-damage buildings) will
rise.  The test metrics report both:

| | Argmax | Thresholded (never-miss) |
|---|---|---|
| FN_minor | higher | 0 (if recall=1.0 on calib) |
| FN_major | higher | 0 (if recall=1.0 on calib) |
| FP_per_1k_no_minor | lower | higher |
| FP_per_1k_no_major | lower | higher |

> Note: recall=1.0 on calib does **not** guarantee FN=0 on outer test — it
> depends on whether the model has learned the class boundary.  With very few
> minority training samples the threshold may be so low it catches everything.

## Example commands

### 1. Baseline CV (no nesting, argmax)

```bash
python scripts/run_cv5.py --folds 0 1 2 3 4
```

Results in `runs/cv5_fold{0..4}/val_metrics.jsonl`.

### 2. Nested + never-miss mode (threshold calibrated on inner calib)

```bash
python scripts/run_cv5.py --folds 0 1 2 3 4 -- \
    --nested_cv 1 \
    --calib_fraction 0.2 \
    --never_miss_mode 1 \
    --loss focal \
    --sampler_mode class_quota_batch \
    --use_sampler 1
```

Per-fold results:
- `runs/cv5_fold{f}/test_metrics_argmax.json` — argmax on outer fold
- `runs/cv5_fold{f}/test_metrics_thresholded.json` — thresholded on outer fold
- `runs/cv5_fold{f}/run_summary.json` — full config + best calib F1
- `models/cv5/fold{f}/thresholds.json` — fitted thresholds + calib summary

Console output per fold will print:
```
  [test/argmax]  macro_f1=...  FN_minor=N  FN_major=N
  [test/thresh]  macro_f1=...  FN_minor=0  FN_major=0
```

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--nested_cv 1` | 0 | Enable nested calibration split |
| `--calib_fraction 0.2` | 0.2 | Fraction of outer-train tiles for calib |
| `--calib_seed N` | same as `--seed` | RNG seed for calib split |
| `--never_miss_mode 1` | 0 | Force target_recall=1.0, threshold_policy=per_class_threshold |
| `--threshold_policy per_class_threshold` | argmax | Enable threshold policy |
| `--target_recall_minor 0.8` | 0.80 | Target recall for minor (ignored in never_miss) |
| `--target_recall_major 0.8` | 0.80 | Target recall for major (ignored in never_miss) |
