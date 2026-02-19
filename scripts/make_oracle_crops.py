#!/usr/bin/env python3
"""Make oracle crops. Run after build_index. Example: python scripts/make_oracle_crops.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from disaster_bench.cli import main
sys.argv = ["disaster-bench", "make-oracle-crops", "--index_csv", "data/processed/index.csv", "--out_dir", "data/processed/crops_oracle"]
sys.exit(main())
