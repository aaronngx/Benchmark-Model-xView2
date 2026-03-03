#!/usr/bin/env python3
"""
LOWO (Leave-One-Wildfire-Out) orchestration for Temporal MAE.

Each LOWO direction has two phases:
  Phase 1 — Pretrain:  train_mae.py on training disasters only
  Phase 2 — Finetune:  train_damage.py with vit_finetune + pretrained encoder

Two runs:
  Run A: pretrain+finetune on socal-fire   → test on santa-rosa-wildfire
  Run B: pretrain+finetune on santa-rosa   → test on socal-fire

Usage:
  python scripts/run_lowo_mae.py           # run both
  python scripts/run_lowo_mae.py --dry_run # print commands only
  python scripts/run_lowo_mae.py --pretrain_only  # only Phase 1
  python scripts/run_lowo_mae.py --finetune_only  # only Phase 2 (encoder.pt must exist)

After completion, evaluate with:
  python scripts/export_predictions.py --run_id lowo_mae_train_socal_test_santarosa ...
  python scripts/bootstrap_tile_ci.py ...

Ref: prompt.md §Build §5 Evaluation, §6 Tile bootstrap CIs
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Run definitions
# ---------------------------------------------------------------------------

RUNS = [
    {
        "name":            "Run A (pretrain+finetune=socal, test=santarosa)",
        "train_disasters": "socal-fire",
        "val_disasters":   "santa-rosa-wildfire",
        "pretrain_out":    "models/mae/lowo_a",
        "finetune_out":    "models/mae_finetune/lowo_a",
        "pretrain_run_id": "mae_pretrain_lowo_a",
        "finetune_run_id": "lowo_mae_train_socal_test_santarosa",
    },
    {
        "name":            "Run B (pretrain+finetune=santarosa, test=socal)",
        "train_disasters": "santa-rosa-wildfire",
        "val_disasters":   "socal-fire",
        "pretrain_out":    "models/mae/lowo_b",
        "finetune_out":    "models/mae_finetune/lowo_b",
        "pretrain_run_id": "mae_pretrain_lowo_b",
        "finetune_run_id": "lowo_mae_train_santarosa_test_socal",
    },
]

# Phase 1: MAE pretraining hyperparameters
PRETRAIN_CONFIG = dict(
    buildings_csv="data/processed/buildings_v2.csv",
    backbone="vit_small",
    epochs=200,
    batch=64,
    lr="1e-4",
    weight_decay="0.05",
    mask_ratio_pre="0.75",
    mask_ratio_post="0.75",
    crop_size=128,
    patch_size=16,
    seed=42,
    num_workers=0,
)

# Phase 2: Fine-tuning hyperparameters (winner recipe adapted for ViT)
FINETUNE_CONFIG = dict(
    index_csv="data/processed/index.csv",
    crops_dir="data/processed/crops_oracle",
    model_type="vit_finetune",
    init_mode="pretrained",
    epochs=30,
    batch=32,
    lr="3e-4",
    size=128,
    use_sampler=1,
    weight_mode="none",
    seed=1,
)


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------

def build_pretrain_cmd(run: dict) -> list[str]:
    c = PRETRAIN_CONFIG
    return [
        sys.executable, "scripts/train_mae.py",
        "--buildings_csv",   c["buildings_csv"],
        "--out_dir",         run["pretrain_out"],
        "--backbone",        c["backbone"],
        "--epochs",          str(c["epochs"]),
        "--batch",           str(c["batch"]),
        "--lr",              str(c["lr"]),
        "--weight_decay",    str(c["weight_decay"]),
        "--mask_ratio_pre",  str(c["mask_ratio_pre"]),
        "--mask_ratio_post", str(c["mask_ratio_post"]),
        "--crop_size",       str(c["crop_size"]),
        "--patch_size",      str(c["patch_size"]),
        "--seed",            str(c["seed"]),
        "--num_workers",     str(c["num_workers"]),
        "--train_disasters", run["train_disasters"],
        "--run_id",          run["pretrain_run_id"],
    ]


def build_finetune_cmd(run: dict) -> list[str]:
    c = FINETUNE_CONFIG
    encoder_path = str(Path(run["pretrain_out"]) / "encoder.pt")
    return [
        sys.executable, "scripts/train_damage.py",
        "--index_csv",            c["index_csv"],
        "--crops_dir",            c["crops_dir"],
        "--model_type",           c["model_type"],
        "--init_mode",            c["init_mode"],
        "--pretrained_ckpt_path", encoder_path,
        "--epochs",               str(c["epochs"]),
        "--batch",                str(c["batch"]),
        "--lr",                   str(c["lr"]),
        "--size",                 str(c["size"]),
        "--use_sampler",          str(c["use_sampler"]),
        "--weight_mode",          c["weight_mode"],
        "--seed",                 str(c["seed"]),
        "--train_disasters",      run["train_disasters"],
        "--val_disasters",        run["val_disasters"],
        "--out_dir",              run["finetune_out"],
        "--run_id",               run["finetune_run_id"],
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="LOWO MAE training (pretrain + finetune)")
    p.add_argument("--dry_run",      action="store_true",
                   help="Print commands without executing")
    p.add_argument("--pretrain_only", action="store_true",
                   help="Only run Phase 1 (MAE pretraining)")
    p.add_argument("--finetune_only", action="store_true",
                   help="Only run Phase 2 (fine-tuning; encoder.pt must already exist)")
    args = p.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    print(f"Project root: {project_root}")

    for run in RUNS:
        print(f"\n{'='*70}")
        print(f"  {run['name']}")
        print(f"{'='*70}")

        # --- Phase 1: Pretrain ---
        if not args.finetune_only:
            pretrain_cmd = build_pretrain_cmd(run)
            print(f"\n[Phase 1] MAE Pretraining — train on {run['train_disasters']}")
            print("  " + " \\\n  ".join(pretrain_cmd))
            if not args.dry_run:
                result = subprocess.run(pretrain_cmd, cwd=str(project_root))
                if result.returncode != 0:
                    print(f"\nERROR: Pretrain phase exited with code {result.returncode}",
                          file=sys.stderr)
                    sys.exit(result.returncode)
                print(f"[Phase 1] Complete.")

        # --- Phase 2: Fine-tune ---
        if not args.pretrain_only:
            finetune_cmd = build_finetune_cmd(run)
            encoder_path = Path(run["pretrain_out"]) / "encoder.pt"
            print(f"\n[Phase 2] Fine-tuning — val/test on {run['val_disasters']}")
            print(f"  Encoder: {encoder_path}")
            if not args.dry_run and not encoder_path.exists():
                print(f"ERROR: encoder.pt not found at {encoder_path}. "
                      f"Run Phase 1 first (or remove --finetune_only).",
                      file=sys.stderr)
                sys.exit(1)
            print("  " + " \\\n  ".join(finetune_cmd))
            if not args.dry_run:
                result = subprocess.run(finetune_cmd, cwd=str(project_root))
                if result.returncode != 0:
                    print(f"\nERROR: Fine-tune phase exited with code {result.returncode}",
                          file=sys.stderr)
                    sys.exit(result.returncode)
                print(f"[Phase 2] Complete.")

    if args.dry_run:
        print("\n[dry_run] No commands executed.")
    else:
        what = ("Phase 1 only" if args.pretrain_only
                else "Phase 2 only" if args.finetune_only
                else "Both phases")
        print(f"\n{what} complete for both LOWO directions.")
        if not args.pretrain_only:
            print("\nNext steps:")
            for run in RUNS:
                print(f"  python scripts/export_predictions.py "
                      f"--run_id {run['finetune_run_id']} ...")
            print("  python scripts/bootstrap_tile_ci.py ...")


if __name__ == "__main__":
    main()
