#!/usr/bin/env python3
"""
Leave-One-Wildfire-Out (LOWO) training runner.

Two runs:
  Run A: train=socal-fire         test=santa-rosa-wildfire
  Run B: train=santa-rosa-wildfire test=socal-fire

Usage:
  python scripts/run_lowo.py           # run both
  python scripts/run_lowo.py --dry_run # print commands only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

RUNS = [
    {
        "name":            "Run A (train=socal, test=santarosa)",
        "run_id":          "lowo_train_socal_test_santarosa",
        "out_dir":         "models/lowo/train_socal_test_santarosa",
        "train_disasters": "socal-fire",
        "val_disasters":   "santa-rosa-wildfire",
    },
    {
        "name":            "Run B (train=santarosa, test=socal)",
        "run_id":          "lowo_train_santarosa_test_socal",
        "out_dir":         "models/lowo/train_santarosa_test_socal",
        "train_disasters": "santa-rosa-wildfire",
        "val_disasters":   "socal-fire",
    },
]

COMMON = dict(
    index_csv="data/processed/index.csv",
    crops_dir="data/processed/crops_oracle",
    model_type="six_channel",
    epochs=30,
    batch=32,
    lr="3e-4",
    size=128,
    use_sampler=1,
    weight_mode="none",
    seed=1,
)


def build_cmd(run: dict) -> list[str]:
    c = COMMON
    return [
        sys.executable, "scripts/train_damage.py",
        "--index_csv",       c["index_csv"],
        "--crops_dir",       c["crops_dir"],
        "--model_type",      c["model_type"],
        "--epochs",          str(c["epochs"]),
        "--batch",           str(c["batch"]),
        "--lr",              str(c["lr"]),
        "--size",            str(c["size"]),
        "--use_sampler",     str(c["use_sampler"]),
        "--weight_mode",     c["weight_mode"],
        "--seed",            str(c["seed"]),
        "--train_disasters", run["train_disasters"],
        "--val_disasters",   run["val_disasters"],
        "--out_dir",         run["out_dir"],
        "--run_id",          run["run_id"],
    ]


def main() -> None:
    p = argparse.ArgumentParser(description="Run LOWO training (2 runs)")
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands without executing")
    args = p.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    print(f"Project root: {project_root}")

    for run in RUNS:
        cmd = build_cmd(run)
        print(f"\n{'='*60}")
        print(run["name"])
        print(f"{'='*60}")
        print(" \\\n  ".join(cmd))

        if not args.dry_run:
            result = subprocess.run(cmd, cwd=str(project_root))
            if result.returncode != 0:
                print(f"\nERROR: {run['name']} exited with code {result.returncode}",
                      file=sys.stderr)
                sys.exit(result.returncode)
            print(f"\n{run['name']} complete.")

    if args.dry_run:
        print("\n[dry_run] No commands executed.")
    else:
        print("\nBoth LOWO runs done.")
        print("Next: python scripts/export_predictions.py  (for each run)")


if __name__ == "__main__":
    main()
