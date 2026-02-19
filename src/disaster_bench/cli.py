"""
CLI: build-index, make-oracle-crops, run, eval-run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load_config(config_path: str | Path | None) -> dict:
    if not config_path or not Path(config_path).is_file():
        return {}
    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        return {}
    except Exception:
        return {}


def cmd_build_index(args: argparse.Namespace) -> int:
    from disaster_bench.data.io import scan_dataset, write_index_csv, SCOPE_DISASTERS
    if args.filter_disasters:
        filt: set[str] | None = set(d.strip() for d in args.filter_disasters.split(","))
    elif args.no_filter:
        filt = None
    else:
        filt = SCOPE_DISASTERS  # default: socal-fire only
    print(f"  Scope filter: {filt if filt else 'ALL (no filter)'}")
    rows = scan_dataset(args.dataset_root, filter_disasters=filt)
    write_index_csv(rows, args.out_csv)
    disasters = sorted(set(r.get("disaster", "") for r in rows))
    print(f"Disasters in index: {disasters}")
    print(f"Wrote {len(rows)} rows to {args.out_csv}")
    return 0


def cmd_make_oracle_crops(args: argparse.Namespace) -> int:
    from disaster_bench.data.io import read_index_csv
    from disaster_bench.data.crops import make_oracle_crops_for_tile
    rows = read_index_csv(args.index_csv)
    total = 0
    for row in rows:
        if not row.get("label_json_path"):
            continue
        n = make_oracle_crops_for_tile(
            row["tile_id"],
            row["pre_path"],
            row["post_path"],
            row["label_json_path"],
            args.out_dir,
            save_masked=args.save_masked,
        )
        total += n
    print(f"Wrote {total} oracle crops to {args.out_dir}")
    return 0


def cmd_overlays(args: argparse.Namespace) -> int:
    from disaster_bench.data.io import read_index_csv
    from disaster_bench.data.overlays import make_overlay_for_tile
    rows = read_index_csv(args.index_csv)
    out_dir = Path(args.out_dir)
    count = 0
    for row in rows[: args.limit]:
        if not row.get("label_json_path"):
            continue
        out_path = out_dir / f"{row['tile_id']}_{args.which}.png"
        if make_overlay_for_tile(
            row["tile_id"],
            row["pre_path"],
            row["post_path"],
            row["label_json_path"],
            out_path,
            which=args.which,
        ):
            count += 1
    print(f"Wrote {count} overlays to {args.out_dir}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    index_csv = args.index_csv or config.get("index_csv", "data/processed/index.csv")
    if not Path(index_csv).is_file():
        print(f"Index not found: {index_csv}", file=sys.stderr)
        return 1
    track = args.track.lower()
    if track == "track1" or track == "1":
        from disaster_bench.pipelines.track1 import run_track1_and_save
        run_track1_and_save(index_csv, run_dir, config)
    elif track == "track2a" or track == "2a":
        from disaster_bench.pipelines.track2a import run_track2a_and_save
        run_track2a_and_save(index_csv, run_dir, config)
    elif track == "track2b" or track == "2b":
        from disaster_bench.pipelines.track2b import run_track2b_and_save
        run_track2b_and_save(index_csv, run_dir, config)
    elif track == "track3" or track == "3":
        from disaster_bench.pipelines.track3 import run_track3_and_save
        run_track3_and_save(index_csv, run_dir, config)
    else:
        print(f"Unknown track: {args.track}", file=sys.stderr)
        return 1
    print(f"Run written to {run_dir}")
    return 0


def cmd_eval_run(args: argparse.Namespace) -> int:
    from disaster_bench.data.io import read_index_csv, load_label_json, get_buildings_from_label
    from disaster_bench.eval.metrics import compute_metrics, coverage
    from disaster_bench.eval.report import write_predictions_csv, write_metrics_json
    run_dir = Path(args.run_dir)
    pred_path = run_dir / "predictions.csv"
    if not pred_path.is_file():
        print(f"Predictions not found: {pred_path}", file=sys.stderr)
        return 1
    with open(pred_path, encoding="utf-8") as f:
        import csv
        reader = csv.DictReader(f)
        pred_rows = list(reader)
    index = read_index_csv(args.index_csv)
    # Build GT damage by (tile_id, uid) from label JSONs
    gt_damage: dict[tuple[str, str], str] = {}
    total_gt = 0
    for row in index:
        label_path = row.get("label_json_path")
        if not label_path:
            continue
        try:
            label_data = load_label_json(label_path)
            for b in get_buildings_from_label(label_data, use_xy=True):
                uid = b["uid"]
                subtype = b.get("subtype") or "no-damage"
                if subtype == "un-classified":
                    subtype = "no-damage"
                gt_damage[(row["tile_id"], uid)] = subtype
                total_gt += 1
        except Exception:
            pass
    # Match pred_instance_id to GT by tile_id (same uid = match for oracle-style runs)
    for r in pred_rows:
        key = (r["tile_id"], r["pred_instance_id"])
        r["gt_damage"] = gt_damage.get(key, "")
        if key in gt_damage:
            r["matched_gt_uid"] = r["pred_instance_id"]
            r["iou"] = 1.0
    matched_uids = set()
    for r in pred_rows:
        if r.get("matched_gt_uid"):
            matched_uids.add((r["tile_id"], r["matched_gt_uid"]))
    metrics = compute_metrics(pred_rows)
    metrics["coverage"] = round(coverage(len(matched_uids), total_gt), 4) if total_gt else None
    write_predictions_csv(pred_rows, pred_path)
    write_metrics_json(metrics, run_dir / "metrics.json")
    print(f"Metrics: macro_f1={metrics.get('macro_f1')}, coverage={metrics.get('coverage')}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="disaster-bench", description="SoCal wildfire damage benchmark")
    sub = p.add_subparsers(dest="command", required=True)
    # build-index
    b = sub.add_parser("build-index", help="Scan dataset and write index.csv")
    b.add_argument("--dataset_root", required=True, help="Path to test_images_labels_targets")
    b.add_argument("--out_csv", required=True, help="Output index CSV path")
    b.add_argument("--filter_disasters", default="", help="Comma-separated disaster names to include (default: socal-fire)")
    b.add_argument("--no_filter", action="store_true", help="Include all disaster types (override scope)")
    b.set_defaults(func=cmd_build_index)
    # make-oracle-crops
    c = sub.add_parser("make-oracle-crops", help="Generate oracle pre/post crops from GT polygons")
    c.add_argument("--index_csv", required=True, help="Index CSV from build-index")
    c.add_argument("--out_dir", required=True, help="Output directory (crops_oracle)")
    c.add_argument("--save_masked", action="store_true", help="Also save pre_masked.png, post_masked.png")
    c.set_defaults(func=cmd_make_oracle_crops)
    # overlays
    o = sub.add_parser("overlays", help="Debug overlay images (masks on pre/post)")
    o.add_argument("--index_csv", required=True, help="Index CSV")
    o.add_argument("--out_dir", required=True, help="Output directory")
    o.add_argument("--which", choices=("pre", "post"), default="post")
    o.add_argument("--limit", type=int, default=10, help="Max tiles to process")
    o.set_defaults(func=cmd_overlays)
    # run
    r = sub.add_parser("run", help="Run a track (1, 2a, 2b, 3)")
    r.add_argument("--track", required=True, help="track1, track2a, track2b, track3")
    r.add_argument("--config", help="YAML config (optional)")
    r.add_argument("--run_dir", required=True, help="Output run directory (e.g. runs/run1)")
    r.add_argument("--index_csv", help="Index CSV (default from config or data/processed/index.csv)")
    r.set_defaults(func=cmd_run)
    # eval-run
    e = sub.add_parser("eval-run", help="Add GT damage and compute metrics for a run")
    e.add_argument("--run_dir", required=True, help="Run directory with predictions.csv")
    e.add_argument("--index_csv", required=True, help="Index CSV (to load GT from labels)")
    e.set_defaults(func=cmd_eval_run)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
