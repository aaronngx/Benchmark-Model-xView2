#!/usr/bin/env python3
"""
Fixed LOWO MAE fine-tuning with proper ViT fine-tune settings.

Reuses the already-trained encoder.pt checkpoints from run_lowo_mae.py.
Runs only Phase 2 (fine-tuning) with corrected hyperparameters to prevent
catastrophic forgetting.

Three fine-tune strategies (pick with --strategy):

  staged   (default, recommended)
    FT0: freeze encoder, train head-only for freeze_epochs (default 10)
    FT1: unfreeze encoder with encoder_lr_scale=0.1 (10x smaller LR)
    Total: 30 epochs. Implements the 3-stage approach.

  low_lr
    Full fine-tune from epoch 1 with encoder_lr_scale=0.1
    + warmup_epochs=5 to prevent early catastrophic forgetting.

  linear_probe
    Freeze encoder permanently. Only trains fusion head.
    Fast diagnostic: tells you if pretrained features are task-relevant at all.

Usage:
  python scripts/run_lowo_mae_ft.py                    # staged (default)
  python scripts/run_lowo_mae_ft.py --strategy low_lr
  python scripts/run_lowo_mae_ft.py --strategy linear_probe
  python scripts/run_lowo_mae_ft.py --dry_run

Ref: prompt.md §Refine — Fix fine-tuning first
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

RUNS = [
    {
        "name":            "Run A (finetune=socal, test=santarosa)",
        "train_disasters": "socal-fire",
        "val_disasters":   "santa-rosa-wildfire",
        "encoder_path":    "models/mae/lowo_a/encoder.pt",
    },
    {
        "name":            "Run B (finetune=santarosa, test=socal)",
        "train_disasters": "santa-rosa-wildfire",
        "val_disasters":   "socal-fire",
        "encoder_path":    "models/mae/lowo_b/encoder.pt",
    },
]

# Shared fine-tune config (same data settings as winner recipe)
BASE_CONFIG = dict(
    index_csv="data/processed/index.csv",
    crops_dir="data/processed/crops_oracle",
    model_type="vit_finetune",
    init_mode="pretrained",
    epochs=30,
    batch=32,
    size=128,
    use_sampler=1,
    weight_mode="none",
    seed=1,
    # Light augmentation (was 0 in original run — added here)
    aug_rotate90=0.5,
    aug_color_jitter=0.3,
)

STRATEGIES = {
    "staged": dict(
        # FT0: head-only for 10 epochs, then FT1: full with 10x smaller enc LR
        lr="1e-4",
        freeze_epochs=10,
        encoder_lr_scale="0.1",
        warmup_epochs=0,
        suffix="staged",
    ),
    "low_lr": dict(
        # Full fine-tune from start, enc gets 10x smaller LR, 5-epoch warmup
        lr="1e-4",
        freeze_epochs=0,
        encoder_lr_scale="0.1",
        warmup_epochs=5,
        suffix="low_lr",
    ),
    "linear_probe": dict(
        # Encoder permanently frozen — pure linear/fusion probe
        lr="1e-3",
        freeze_encoder=1,
        encoder_lr_scale="1.0",
        warmup_epochs=0,
        suffix="linear_probe",
        epochs=20,  # fewer epochs needed for head-only
    ),
}


def build_cmd(run: dict, strategy_cfg: dict) -> list[str]:
    c = {**BASE_CONFIG, **strategy_cfg}
    out_dir = (f"models/mae_finetune_{c['suffix']}/"
               f"{'lowo_a' if 'socal' in run['train_disasters'] else 'lowo_b'}")
    run_id = (f"lowo_mae_{c['suffix']}_"
              f"{'train_socal_test_santarosa' if 'socal' in run['train_disasters'] else 'train_santarosa_test_socal'}")

    cmd = [
        sys.executable, "scripts/train_damage.py",
        "--index_csv",            c["index_csv"],
        "--crops_dir",            c["crops_dir"],
        "--model_type",           c["model_type"],
        "--init_mode",            c["init_mode"],
        "--pretrained_ckpt_path", run["encoder_path"],
        "--epochs",               str(c.get("epochs", BASE_CONFIG["epochs"])),
        "--batch",                str(c["batch"]),
        "--lr",                   str(c["lr"]),
        "--size",                 str(c["size"]),
        "--use_sampler",          str(c["use_sampler"]),
        "--weight_mode",          c["weight_mode"],
        "--seed",                 str(c["seed"]),
        "--train_disasters",      run["train_disasters"],
        "--val_disasters",        run["val_disasters"],
        "--out_dir",              out_dir,
        "--run_id",               run_id,
        "--aug_rotate90",         str(c["aug_rotate90"]),
        "--aug_color_jitter",     str(c["aug_color_jitter"]),
        "--encoder_lr_scale",     str(c["encoder_lr_scale"]),
    ]

    if c.get("freeze_encoder", 0):
        cmd += ["--freeze_encoder", "1"]
    if c.get("freeze_epochs", 0):
        cmd += ["--freeze_epochs", str(c["freeze_epochs"])]
    if c.get("warmup_epochs", 0):
        cmd += ["--warmup_epochs", str(c["warmup_epochs"])]

    return cmd


def main() -> None:
    p = argparse.ArgumentParser(description="Fixed LOWO MAE fine-tuning")
    p.add_argument("--strategy", default="staged",
                   choices=list(STRATEGIES.keys()),
                   help="Fine-tune strategy (default: staged)")
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands without executing")
    p.add_argument("--run_a_only", action="store_true",
                   help="Only run LOWO Run A")
    p.add_argument("--run_b_only", action="store_true",
                   help="Only run LOWO Run B")
    args = p.parse_args()

    strategy_cfg = STRATEGIES[args.strategy]
    project_root = Path(__file__).resolve().parent.parent

    runs_to_do = RUNS
    if args.run_a_only:
        runs_to_do = [RUNS[0]]
    elif args.run_b_only:
        runs_to_do = [RUNS[1]]

    print(f"Strategy: {args.strategy}")
    print(f"  LR={strategy_cfg['lr']}, "
          f"encoder_lr_scale={strategy_cfg['encoder_lr_scale']}, "
          f"freeze_epochs={strategy_cfg.get('freeze_epochs', 0)}, "
          f"freeze_encoder={strategy_cfg.get('freeze_encoder', 0)}, "
          f"warmup={strategy_cfg.get('warmup_epochs', 0)}")

    for run in runs_to_do:
        print(f"\n{'='*70}")
        print(f"  {run['name']}")
        print(f"{'='*70}")

        encoder_path = Path(run["encoder_path"])
        if not args.dry_run and not encoder_path.exists():
            print(f"ERROR: {encoder_path} not found. Run run_lowo_mae.py --pretrain_only first.",
                  file=sys.stderr)
            sys.exit(1)

        cmd = build_cmd(run, strategy_cfg)
        print("  " + " \\\n  ".join(cmd))

        if not args.dry_run:
            result = subprocess.run(cmd, cwd=str(project_root))
            if result.returncode != 0:
                print(f"\nERROR: exited with code {result.returncode}", file=sys.stderr)
                sys.exit(result.returncode)
            print(f"  Done.")

    if args.dry_run:
        print("\n[dry_run] No commands executed.")
    else:
        print(f"\nAll runs complete ({args.strategy} strategy).")
        print("Compare results:")
        print("  python scripts/summarize_runs.py  (or check runs/ dirs directly)")


if __name__ == "__main__":
    main()
