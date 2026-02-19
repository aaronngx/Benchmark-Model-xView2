#!/usr/bin/env python3
"""Evaluate a run (add GT, compute metrics). Example: python scripts/eval_run.py runs/default"""
import sys
from pathlib import Path
if len(sys.argv) < 2:
    print("Usage: python scripts/eval_run.py <run_dir> [index_csv]")
    sys.exit(1)
run_dir = sys.argv[1]
index_csv = sys.argv[2] if len(sys.argv) > 2 else "data/processed/index.csv"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from disaster_bench.cli import main
sys.argv = ["disaster-bench", "eval-run", "--run_dir", run_dir, "--index_csv", index_csv]
sys.exit(main())
