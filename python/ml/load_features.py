"""
Build a minute-level feature table from NCAA play-by-play + Polymarket data.

Reads data/games/*.json (produced by fetch_games.py), replays each game's
play-by-play event stream, resamples to 1-minute game-clock intervals, merges
Polymarket price/trade features, and writes data/features_basketball.parquet.

Usage:
    python load_features.py
    python load_features.py --input data/games --output data/features_basketball.parquet
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import pandas as pd

from in_game_features import (
    add_derived_features,
    merge_polymarket,
    replay_game,
    resample_to_minutes,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Build basketball feature table")
    parser.add_argument("--input", default=str(REPO_ROOT / "data" / "games"),
                        help="Directory containing game JSON files")
    parser.add_argument("--output", default=str(REPO_ROOT / "data" / "features_basketball.parquet"),
                        help="Output Parquet path")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input, "*.json")))
    if not files:
        print(f"No game JSON files found in {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(files)} game files...")
    all_dfs = []
    for path in files:
        with open(path) as f:
            record = json.load(f)

        meta = record.get("meta", {})
        pbp = record.get("ncaa_pbp")
        poly = record.get("polymarket", {})

        if not pbp or not pbp.get("periods"):
            print(f"  SKIP {meta.get('title', path)}: no play-by-play data")
            continue

        events = replay_game(pbp)
        if len(events) < 5:
            print(f"  SKIP {meta.get('title', path)}: too few events ({len(events)})")
            continue

        df = resample_to_minutes(events, meta)

        game_start_epoch = 0.0
        if poly.get("trades"):
            timestamps = [float(t.get("timestamp", 0) or t.get("matchedAt", 0) or 0) for t in poly["trades"]]
            if timestamps:
                game_start_epoch = min(t for t in timestamps if t > 0) if any(t > 0 for t in timestamps) else 0

        df = merge_polymarket(df, poly, game_start_epoch)
        df = add_derived_features(df)
        all_dfs.append(df)
        poly_status = "with Poly" if poly.get("found") else "no Poly"
        print(f"  {meta.get('title', '')}: {len(df)} minute-rows ({poly_status})")

    if not all_dfs:
        print("ERROR: No valid games processed", file=sys.stderr)
        sys.exit(1)

    combined = pd.concat(all_dfs, ignore_index=True)

    if "game_date" in combined.columns:
        combined = combined.sort_values(["game_date", "game_id", "time_remaining_sec"],
                                        ascending=[True, True, False]).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    combined.to_parquet(args.output, index=False)
    print(f"\nWrote {len(combined)} rows x {len(combined.columns)} cols to {args.output}")
    print(f"Games: {combined['game_id'].nunique()}")
    print(f"Label distribution:\n{combined['label'].value_counts().to_string()}")
    print(f"Poly coverage: {(combined['has_poly'] == 1).sum()} / {len(combined)} rows")


if __name__ == "__main__":
    main()
