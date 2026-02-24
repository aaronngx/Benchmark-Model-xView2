#!/usr/bin/env python3
"""
Run 5-fold tile-level CV training sequentially.

Each fold uses the winner recipe: --use_sampler 1 --weight_mode none
Checkpoints: models/cv5/fold{f}/best.pt
Logs:        runs/cv5_fold{f}/val_metrics.jsonl

Usage:
  python scripts/run_cv5.py                        # run all 5 folds
  python scripts/run_cv5.py --folds 0 2 4          # run specific folds
  python scripts/run_cv5.py --dry_run              # print commands only
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


def build_cmd(fold: int) -> list[str]:
    d = DEFAULTS
    return [
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


def main() -> None:
    p = argparse.ArgumentParser(description="Run all 5 CV folds sequentially")
    p.add_argument("--folds",   type=int, nargs="+", default=list(range(5)),
                   help="Which folds to run (default: 0 1 2 3 4)")
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands without executing")
    args = p.parse_args()

    # Ensure we run from the project root
    project_root = Path(__file__).resolve().parent.parent
    print(f"Project root: {project_root}")

    for fold in sorted(args.folds):
        cmd = build_cmd(fold)
        cmd_str = " \\\n  ".join(cmd)
        print(f"\n{'='*60}")
        print(f"FOLD {fold}")
        print(f"{'='*60}")
        print(cmd_str)

        if not args.dry_run:
            result = subprocess.run(cmd, cwd=str(project_root))
            if result.returncode != 0:
                print(f"\nERROR: fold {fold} exited with code {result.returncode}",
                      file=sys.stderr)
                sys.exit(result.returncode)
            print(f"\nFold {fold} complete.")

    if args.dry_run:
        print("\n[dry_run] No commands executed.")
    else:
        print(f"\nAll folds done. Logs: runs/cv5_fold{{0..{max(args.folds)}}}/val_metrics.jsonl")
        print("Next: python scripts/summarize_cv.py")


if __name__ == "__main__":
    main()
