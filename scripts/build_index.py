#!/usr/bin/env python3
"""Build dataset index. Example: python scripts/build_index.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from disaster_bench.cli import main
sys.argv = ["disaster-bench", "build-index", "--dataset_root", "test_images_labels_targets", "--out_csv", "data/processed/index.csv"]
sys.exit(main())
