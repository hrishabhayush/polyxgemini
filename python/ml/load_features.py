"""
Load JSONL snapshots exported by cmd/export and assemble a Parquet feature table.

Usage:
    python load_features.py                           # defaults: data/snapshots -> data/features.parquet
    python load_features.py --input data/snapshots --output data/features.parquet
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


NUMERIC_FEATURES = [
    "current_price",
    "hurst_exp",
    "vol_ratio",
    "trade_count",
    "total_volume",
    "kyles_lambda",
    "vpin",
    "buy_fraction",
    "wallet_hhi",
    "article_count",
    "bullish_score",
    "bearish_score",
    "resolution_reliability",
]


def load_jsonl_dir(input_dir: str) -> pd.DataFrame:
    """Read all resolved_*.jsonl files and return a DataFrame."""
    pattern = os.path.join(input_dir, "resolved_*.jsonl")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No resolved_*.jsonl files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for path in files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    print(f"Loaded {len(rows)} rows from {len(files)} file(s)")
    return pd.DataFrame(rows)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features on top of the raw exported columns."""
    out = df.copy()

    # Parse dates
    for col in ("created_at", "end_date", "snapshot_at"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce", utc=True)

    # Binary label: 1 = Yes, 0 = No
    out["label"] = (out["outcome"].str.lower() == "yes").astype(int)

    # Sentiment net signal
    out["sentiment_net"] = out["bullish_score"] - out["bearish_score"]

    # Log volume (handles zero gracefully)
    out["log_volume"] = np.log1p(out["total_volume"])

    # Market age in days (end_date - created_at)
    if "created_at" in out.columns and "end_date" in out.columns:
        out["market_age_days"] = (
            (out["end_date"] - out["created_at"]).dt.total_seconds() / 86400
        )
    else:
        out["market_age_days"] = np.nan

    # Distance from maximum uncertainty (price = 0.5)
    out["price_distance_from_50"] = (out["current_price"] - 0.5).abs()

    # Encode categorical: jump_result -> ordinal
    jump_map = {"no_jumps": 0, "reversed": 1, "sustained": 2}
    out["jump_result_enc"] = out["jump_result"].map(jump_map).fillna(0).astype(int)

    return out


def main():
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Assemble ML feature table from JSONL snapshots")
    parser.add_argument("--input", default=str(repo_root / "data" / "snapshots"),
                        help="Directory containing resolved_*.jsonl files")
    parser.add_argument("--output", default=str(repo_root / "data" / "features.parquet"),
                        help="Output Parquet path")
    args = parser.parse_args()

    df = load_jsonl_dir(args.input)
    df = engineer_features(df)

    # Drop rows without a valid outcome
    before = len(df)
    df = df.dropna(subset=["label"])
    df = df[df["outcome"].isin(["Yes", "No", "yes", "no"])]
    print(f"Kept {len(df)} / {before} rows with valid outcomes")

    # Drop rows with no CLOB data — these are old markets where the API returned nothing.
    # A row with current_price=0 AND trade_count=0 has zero signal for the model.
    before = len(df)
    has_price_data = (df["current_price"] != 0) | (df["trade_count"] != 0)
    df = df[has_price_data]
    dropped = before - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} rows with no price/trade data (old markets with no CLOB history)")

    # Sort by end_date for time-based splitting downstream
    if "end_date" in df.columns:
        df = df.sort_values("end_date").reset_index(drop=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Wrote {len(df)} rows x {len(df.columns)} cols to {args.output}")

    # Summary
    print("\nLabel distribution:")
    print(df["label"].value_counts().to_string())
    print(f"\nFeature columns: {sorted(df.columns.tolist())}")


if __name__ == "__main__":
    main()
