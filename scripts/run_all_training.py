#!/usr/bin/env python3
"""
Sequential training orchestrator — runs all pending model training + benchmark evaluation.
Each step logs to logs/<step>.txt. A failed step is reported but does not block later steps.

Steps:
  1.  Train footprint U-Net        → models/footprint_unet/best.pt
  2.  Benchmark: track1_deploy     → runs/track1_deploy/
  3.  Train PrePostDiffCNN         → models/pre_post_diff/best.pt
  4.  Benchmark: track1 (ppd)      → runs/track1_ppd/
  5.  Train SiameseCNN             → models/siamese/best.pt
  6.  Benchmark: track1 (siamese)  → runs/track1_siamese/
  7.  Train CentroidPatchCNN       → models/centroid_patch/best.pt  (train only)
  8.  Train Siamese U-Net          → models/siamese_unet/best.pt    (train only)
  9.  Print final comparison table

Usage:
    py scripts/run_all_training.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
LOGS   = ROOT / "logs"
LOGS.mkdir(exist_ok=True)

PY = sys.executable   # same interpreter that launched this script


def run(label: str, cmd: list[str], log_file: Path) -> bool:
    """Run a subprocess, tee output to log_file and stdout. Returns True on success."""
    print(f"\n{'='*60}", flush=True)
    print(f"  STEP: {label}", flush=True)
    print(f"  CMD : {' '.join(cmd)}", flush=True)
    print(f"  LOG : {log_file}", flush=True)
    print(f"{'='*60}", flush=True)
    t0 = time.perf_counter()
    with open(log_file, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(ROOT),
            encoding="utf-8",
            errors="replace",
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
        proc.wait()
    elapsed = time.perf_counter() - t0
    ok = proc.returncode == 0
    status = "OK" if ok else f"FAILED (rc={proc.returncode})"
    print(f"  --> {status}  ({elapsed:.0f}s)\n", flush=True)
    return ok


STEPS: list[tuple[str, list[str], str]] = [
    # (label, cmd_args, log_stem)

    # --- Footprint U-Net ---
    (
        "Train footprint U-Net",
        [PY, "scripts/train_footprint_unet.py",
         "--index_csv", "data/processed/index.csv",
         "--out_dir",   "models/footprint_unet",
         "--epochs",    "30",
         "--batch",     "4",
         "--patch_size","512",
         "--stride",    "256"],
        "train_footprint",
    ),
    (
        "Benchmark: track1_deploy (footprint U-Net + six_channel CNN)",
        [PY, "-m", "disaster_bench.cli", "run",
         "--track",    "track1_deploy",
         "--config",   "configs/tracks/track1_deploy.yaml",
         "--run_dir",  "runs/track1_deploy",
         "--index_csv","data/processed/index.csv"],
        "run_track1_deploy",
    ),
    (
        "Eval: track1_deploy",
        [PY, "-m", "disaster_bench.cli", "eval-run",
         "--run_dir",  "runs/track1_deploy",
         "--index_csv","data/processed/index.csv"],
        "eval_track1_deploy",
    ),

    # --- PrePostDiffCNN ---
    (
        "Train PrePostDiffCNN (9-channel)",
        [PY, "scripts/train_damage.py",
         "--model_type","pre_post_diff",
         "--out_dir",   "models/pre_post_diff",
         "--epochs",    "40",
         "--batch",     "64",
         "--lr",        "3e-4"],
        "train_ppd",
    ),
    (
        "Benchmark: track1 (PrePostDiffCNN)",
        [PY, "-m", "disaster_bench.cli", "run",
         "--track",    "track1",
         "--config",   "configs/tracks/track1_pre_post_diff.yaml",
         "--run_dir",  "runs/track1_ppd",
         "--index_csv","data/processed/index.csv"],
        "run_track1_ppd",
    ),
    (
        "Eval: track1 (PrePostDiffCNN)",
        [PY, "-m", "disaster_bench.cli", "eval-run",
         "--run_dir",  "runs/track1_ppd",
         "--index_csv","data/processed/index.csv"],
        "eval_track1_ppd",
    ),

    # --- SiameseCNN ---
    (
        "Train SiameseCNN (dual-stream)",
        [PY, "scripts/train_damage.py",
         "--model_type","siamese",
         "--out_dir",   "models/siamese",
         "--epochs",    "40",
         "--batch",     "64",
         "--lr",        "3e-4"],
        "train_siamese",
    ),
    (
        "Benchmark: track1 (SiameseCNN)",
        [PY, "-m", "disaster_bench.cli", "run",
         "--track",    "track1",
         "--config",   "configs/tracks/track1_siamese.yaml",
         "--run_dir",  "runs/track1_siamese",
         "--index_csv","data/processed/index.csv"],
        "run_track1_siamese",
    ),
    (
        "Eval: track1 (SiameseCNN)",
        [PY, "-m", "disaster_bench.cli", "eval-run",
         "--run_dir",  "runs/track1_siamese",
         "--index_csv","data/processed/index.csv"],
        "eval_track1_siamese",
    ),

    # --- CentroidPatchCNN (train only — inference needs centroid pipeline) ---
    (
        "Train CentroidPatchCNN [train only, no benchmark run]",
        [PY, "scripts/train_damage.py",
         "--model_type","centroid_patch",
         "--out_dir",   "models/centroid_patch",
         "--epochs",    "40",
         "--batch",     "32",
         "--lr",        "3e-4"],
        "train_centroid",
    ),

    # --- Siamese U-Net (train only — no track2b learned pipeline yet) ---
    (
        "Train Siamese U-Net [train only, no benchmark run]",
        [PY, "scripts/train_siamese_unet.py",
         "--out_dir",  "models/siamese_unet",
         "--size",     "256",
         "--batch",    "4",
         "--epochs",   "30"],
        "train_siamese_unet",
    ),
]


def print_summary(results: list[tuple[str, bool]]) -> None:
    print(f"\n{'='*60}")
    print("  TRAINING RUN SUMMARY")
    print(f"{'='*60}")
    for label, ok in results:
        icon = "[OK]" if ok else "[FAIL]"
        print(f"  {icon}  {label}")
    n_ok   = sum(1 for _, ok in results if ok)
    n_fail = len(results) - n_ok
    print(f"\n  {n_ok}/{len(results)} steps succeeded", flush=True)
    if n_fail:
        print(f"  Check logs/ for failed steps.")

    # Print metrics comparison for completed runs
    import json, os
    run_dirs = [
        ("track1_final",    "runs/track1_final"),
        ("track1_deploy",   "runs/track1_deploy"),
        ("track1_ppd",      "runs/track1_ppd"),
        ("track1_siamese",  "runs/track1_siamese"),
    ]
    print(f"\n{'='*60}")
    print("  METRICS COMPARISON (available runs)")
    print(f"{'='*60}")
    print(f"  {'Run':<22} {'macro_F1':>9} {'FEMA_F1':>9} {'coverage':>9}")
    print(f"  {'-'*22} {'-'*9} {'-'*9} {'-'*9}")
    for name, d in run_dirs:
        mj = Path(d) / "metrics.json"
        if not mj.is_file():
            continue
        try:
            m = json.loads(mj.read_text(encoding="utf-8"))
            mf1 = m.get("macro_f1")
            ff1 = m.get("fema_macro_f1")
            cov = m.get("coverage")
            mf1_s = f"{mf1:.4f}" if mf1 is not None else "  --  "
            ff1_s = f"{ff1:.4f}" if ff1 is not None else "  --  "
            cov_s = f"{cov:.4f}" if cov is not None else "  --  "
            print(f"  {name:<22} {mf1_s:>9} {ff1_s:>9} {cov_s:>9}")
        except Exception:
            print(f"  {name:<22}   (error reading metrics)")


def main() -> None:
    results: list[tuple[str, bool]] = []
    for label, cmd, log_stem in STEPS:
        log_file = LOGS / f"{log_stem}.txt"
        ok = run(label, cmd, log_file)
        results.append((label, ok))
    print_summary(results)


if __name__ == "__main__":
    main()
