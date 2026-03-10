#!/usr/bin/env python3
"""
Orchestrate 3-stage binary cascade classifier training.

Hierarchy (follows difficulty order):
  Stage 1: no-damage vs any-damage (all 8316 buildings, easiest boundary)
  Stage 2: partial-damage vs destroyed (1511 buildings, medium difficulty)
  Stage 3: minor vs major (91 buildings, hardest boundary — kept for last)

Each stage trains a binary SixChannelCNN using train_damage.py with
--label_remap and --selection_metric tuned per stage.

Usage:
  python scripts/run_binary_cascade.py --dry_run
  python scripts/run_binary_cascade.py --seed 1
  python scripts/run_binary_cascade.py --seed 1 --stage_only 3
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BASE = dict(
    index_csv  = "data/processed/index.csv",
    crops_dir  = "data/processed/crops_oracle",
    model_type = "six_channel",
    batch      = 32,
    weight_mode= "none",
    use_sampler= 1,
    skip_dataset_check = 0,
)

STAGES = [
    # Stage 1: no-damage vs any-damage (all 8316 buildings)
    # High recall_pos goal: minimize missed damaged buildings passing to Stage 2.
    dict(
        stage_num           = 1,
        label_remap         = "s1_nodmg_vs_dmg",
        aug_rotate90        = 0.5,
        aug_color_jitter_indep = 0.3,
        selection_metric    = "recall_pos",
        epochs              = 30,
        size                = 128,
        lr                  = 3e-4,
    ),
    # Stage 2: partial-damage vs destroyed (1511 buildings: minor+major+destroyed)
    # macro_f1 selection + no weights = BEST config (partial recall=0.583, dest P=0.972).
    # High destroyed precision is critical: low-precision variants flood S3 with destroyed
    # buildings that S3 misclassifies as minor/major, killing f1_major.
    # Ablation (all worse than baseline):
    #   recall_neg: partial R=0.917 but P=0.135 → cascade f1_major 0.240→0.021
    #   normalized_invfreq + macro_f1: partial R=0.833 but P=0.160 → cascade f1_major 0.240→0.000
    dict(
        stage_num           = 2,
        label_remap         = "s2_partial_vs_dest",
        aug_rotate90        = 0.5,
        aug_color_jitter_indep = 0.0,
        selection_metric    = "macro_f1",
        epochs              = 30,
        size                = 128,
        lr                  = 3e-4,
    ),
    # Stage 3: minor vs major (91 buildings only)
    # Higher resolution (256px) + multi-scale aug for subtle detail.
    # More epochs because each epoch is fast (91 samples) and the boundary is hard.
    dict(
        stage_num           = 3,
        label_remap         = "s3_minor_vs_major",
        aug_rotate90        = 0.5,
        aug_color_jitter_indep = 0.3,
        aug_multiscale      = 1,
        selection_metric    = "macro_f1",
        epochs              = 50,
        size                = 256,
        lr                  = 3e-4,
    ),
]


def build_cmd(stage: dict, seed: int) -> list[str]:
    sn   = stage["stage_num"]
    out_dir  = f"models/binary_cascade/stage{sn}"
    run_id   = f"binary_cascade_s{sn}_seed{seed}"

    cmd = [
        sys.executable, "scripts/train_damage.py",
        "--index_csv",        BASE["index_csv"],
        "--crops_dir",        BASE["crops_dir"],
        "--model_type",       BASE["model_type"],
        "--out_dir",          out_dir,
        "--run_id",           run_id,
        "--epochs",           str(stage["epochs"]),
        "--batch",            str(BASE["batch"]),
        "--lr",               str(stage["lr"]),
        "--size",             str(stage["size"]),
        "--weight_mode",      stage.get("weight_mode", BASE["weight_mode"]),
        "--use_sampler",      str(BASE["use_sampler"]),
        "--seed",             str(seed),
        "--label_remap",      stage["label_remap"],
        "--selection_metric", stage["selection_metric"],
        "--aug_rotate90",     str(stage["aug_rotate90"]),
        "--aug_color_jitter_indep", str(stage.get("aug_color_jitter_indep", 0.0)),
        "--skip_dataset_check", str(BASE["skip_dataset_check"]),
    ]

    if stage.get("aug_multiscale", 0):
        cmd += ["--aug_multiscale", "1"]

    return cmd


def main() -> None:
    p = argparse.ArgumentParser(description="3-stage binary cascade training")
    p.add_argument("--seed",       type=int, default=1,
                   help="Global RNG seed (default 1)")
    p.add_argument("--stage_only", type=int, choices=[1, 2, 3], default=None,
                   help="Train only this stage number (default: all 3)")
    p.add_argument("--dry_run",    action="store_true",
                   help="Print commands without executing")
    p.add_argument("--skip_dataset_check", type=int, choices=[0, 1], default=0,
                   help="Pass --skip_dataset_check to train_damage.py (default 0=strict)")
    args = p.parse_args()
    BASE["skip_dataset_check"] = args.skip_dataset_check

    project_root = Path(__file__).resolve().parent.parent

    stages_to_run = [s for s in STAGES
                     if args.stage_only is None or s["stage_num"] == args.stage_only]

    for stage in stages_to_run:
        sn  = stage["stage_num"]
        cmd = build_cmd(stage, args.seed)

        print(f"\n{'='*70}")
        print(f"  Stage {sn}: {stage['label_remap']}  "
              f"(epochs={stage['epochs']}, size={stage['size']}px, "
              f"selection={stage['selection_metric']})")
        print(f"{'='*70}")
        print("  " + " \\\n    ".join(cmd))

        if not args.dry_run:
            result = subprocess.run(cmd, cwd=str(project_root))
            if result.returncode != 0:
                print(f"\nERROR: Stage {sn} exited with code {result.returncode}",
                      file=sys.stderr)
                sys.exit(result.returncode)
            print(f"  Stage {sn} done -> models/binary_cascade/stage{sn}/best.pt")

    if args.dry_run:
        print("\n[dry_run] No commands executed.")
    else:
        print(f"\nAll {len(stages_to_run)} stage(s) complete.")
        print("Next: python scripts/eval_binary_cascade.py "
              "--s1_ckpt models/binary_cascade/stage1/best.pt "
              "--s2_ckpt models/binary_cascade/stage2/best.pt "
              "--s3_ckpt models/binary_cascade/stage3/best.pt "
              "--sweep_thresholds")


if __name__ == "__main__":
    main()
