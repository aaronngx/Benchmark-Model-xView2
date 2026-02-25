#!/usr/bin/env python3
"""
Run 5-fold tile-level CV training sequentially.

Each fold uses the winner recipe: --use_sampler 1 --weight_mode none
Checkpoints: models/cv5/fold{f}/best.pt
Logs:        runs/cv5_fold{f}/val_metrics.jsonl

Usage:
  python scripts/run_cv5.py                        # run all 5 folds (baseline)
  python scripts/run_cv5.py --folds 0 2 4          # run specific folds
  python scripts/run_cv5.py --dry_run              # print commands only

  # Pass extra args to train_damage.py for every fold (use '--' separator):
  python scripts/run_cv5.py --folds 0 1 2 3 4 -- `
      --nested_cv 1 --calib_fraction 0.2 --never_miss_mode 1 `
      --loss ldam_drw --sampler_mode class_quota_batch `
      --logit_adjust train_prior --aug_rotate90 0.5

  # Ordinal "always detect" never-miss run with crossfit_pool calibration
  # (two-pass: train all folds first, then eval with pooled calib):
  python scripts/run_cv5.py -- --nested_cv 1 --calib_mode crossfit_pool `
      --threshold_policy ordinal_threshold --never_miss_mode 1

  Everything after '--' is forwarded verbatim to train_damage.py.

crossfit_pool two-pass flow (automatic when --calib_mode crossfit_pool is present):
  Pass 1: Train all requested folds → each fold saves models/cv5/fold{f}/calib_preds.npz
  Pass 2: Re-run outer-test evaluation for each fold with --eval_only 1, using the
          pooled calib predictions from all other folds.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULTS = dict(
    index_csv="data/processed/index.csv",
    crops_dir="data/processed/crops_oracle",
    cv_folds_path="data/processed/cv_folds_k5_seed42.json",
    model_type="six_channel",
    epochs=30,
    batch=32,
    lr="3e-4",
    size=128,
    use_sampler=1,
    weight_mode="none",
    seed=1,
)


def build_cmd(fold: int, extra_args: list[str]) -> list[str]:
    d = DEFAULTS
    cmd = [
        sys.executable, "scripts/train_damage.py",
        "--index_csv",    d["index_csv"],
        "--crops_dir",    d["crops_dir"],
        "--model_type",   d["model_type"],
        "--epochs",       str(d["epochs"]),
        "--batch",        str(d["batch"]),
        "--lr",           str(d["lr"]),
        "--size",         str(d["size"]),
        "--use_sampler",  str(d["use_sampler"]),
        "--weight_mode",  d["weight_mode"],
        "--cv_folds_path", d["cv_folds_path"],
        "--cv_fold",      str(fold),
        "--seed",         str(d["seed"]),
        "--out_dir",      f"models/cv5/fold{fold}",
        "--run_id",       f"cv5_fold{fold}",
    ]
    cmd.extend(extra_args)
    return cmd


def _run_fold(fold: int, extra_args: list[str], project_root: Path,
              dry_run: bool, label: str = "") -> None:
    cmd = build_cmd(fold, extra_args)
    cmd_str = " \\\n  ".join(cmd)
    tag = f" [{label}]" if label else ""
    print(f"\n{'='*60}")
    print(f"FOLD {fold}{tag}")
    print(f"{'='*60}")
    print(cmd_str)
    if not dry_run:
        result = subprocess.run(cmd, cwd=str(project_root))
        if result.returncode != 0:
            print(f"\nERROR: fold {fold}{tag} exited with code {result.returncode}",
                  file=sys.stderr)
            sys.exit(result.returncode)
        print(f"\nFold {fold}{tag} complete.")


def main() -> None:
    # Split sys.argv on '--' to separate run_cv5 args from train_damage passthrough args
    argv = sys.argv[1:]
    try:
        sep_idx   = argv.index("--")
        my_argv   = argv[:sep_idx]
        extra_args = argv[sep_idx + 1:]
    except ValueError:
        my_argv   = argv
        extra_args = []

    p = argparse.ArgumentParser(description="Run all 5 CV folds sequentially")
    p.add_argument("--folds",   type=int, nargs="+", default=list(range(5)),
                   help="Which folds to run (default: 0 1 2 3 4)")
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands without executing")
    args = p.parse_args(my_argv)

    # Ensure we run from the project root
    project_root = Path(__file__).resolve().parent.parent
    print(f"Project root: {project_root}")
    if extra_args:
        print(f"Extra args forwarded to train_damage.py: {extra_args}")

    # Detect crossfit_pool mode → two-pass execution
    _crossfit = "crossfit_pool" in extra_args

    if _crossfit:
        print("\n[crossfit_pool] Two-pass mode:")
        print("  Pass 1 — train all folds + save calib_preds.npz")
        print("  Pass 2 — eval_only with pooled sibling calib\n")

    # -----------------------------------------------------------------------
    # Pass 1: train all folds (always runs)
    # -----------------------------------------------------------------------
    for fold in sorted(args.folds):
        _run_fold(fold, extra_args, project_root, args.dry_run,
                  label="train" if _crossfit else "")

    # -----------------------------------------------------------------------
    # Pass 2 (crossfit_pool only): re-run outer-test eval with pooled calib
    # -----------------------------------------------------------------------
    if _crossfit:
        print(f"\n{'='*60}")
        print("Pass 2: eval_only with crossfit pooled calibration")
        print(f"{'='*60}")
        # Pass 2 keeps all original extra_args (so calib_mode crossfit_pool is preserved)
        # and adds --eval_only 1 if not already present.
        eval_extra = list(extra_args)
        if "--eval_only" not in eval_extra:
            eval_extra = ["--eval_only", "1"] + eval_extra
        for fold in sorted(args.folds):
            _run_fold(fold, eval_extra, project_root, args.dry_run, label="eval")

    if args.dry_run:
        print("\n[dry_run] No commands executed.")
    else:
        print(f"\nAll folds done. Logs: runs/cv5_fold{{0..{max(args.folds)}}}/val_metrics.jsonl")
        if _crossfit:
            print("Ordinal metrics: runs/cv5_fold{f}/test_metrics_ordinal.json")
        print("Next: python scripts/summarize_cv.py")


if __name__ == "__main__":
    main()
