#!/usr/bin/env python3
"""Run a track. Example: python scripts/run_track.py track1"""
import sys
from pathlib import Path
if len(sys.argv) < 2:
    print("Usage: python scripts/run_track.py <track1|track2a|track2b|track3> [run_id]")
    sys.exit(1)
track = sys.argv[1]
run_id = sys.argv[2] if len(sys.argv) > 2 else "default"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from disaster_bench.cli import main
config = f"configs/tracks/{track}.yaml"
sys.argv = ["disaster-bench", "run", "--track", track, "--config", config, "--run_dir", f"runs/{run_id}"]
sys.exit(main())
