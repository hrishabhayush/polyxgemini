"""
Show per-market model predictions vs actual outcomes on the held-out test set.

Usage:
    python python/ml/eval_test.py
    python python/ml/eval_test.py --train-frac 0.7 --val-frac 0.15
"""

import argparse
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from calibration_utils import apply_calibration, load_calibration
from train import add_interaction_features

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = Path(__file__).parent / "artifacts"


def load_pipeline():
    model = lgb.Booster(model_file=str(ARTIFACTS / "model.txt"))
    with open(ARTIFACTS / "model_meta.json") as f:
        meta = json.load(f)
    calibrator = load_calibration(ARTIFACTS / "calibration.json")
    return model, meta, calibrator


def predict(model, meta, calibrator, X):
    raw = model.predict(X)
    prob = apply_calibration(raw, calibrator)
    alpha = meta.get("blend_alpha", 1.0)
    return raw, prob, alpha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(REPO_ROOT / "data" / "features.parquet"))
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--min-edge", type=float, default=0.0,
                        help="Only show markets where |edge| >= this value")
    args = parser.parse_args()

    model, meta, calibrator = load_pipeline()
    feature_cols = meta["feature_names"]
    blend_alpha = meta.get("blend_alpha", 1.0)

    df = pd.read_parquet(args.input)
    df = add_interaction_features(df)
    df[feature_cols] = df[feature_cols].fillna(0)

    n = len(df)
    test_start = int(n * (args.train_frac + args.val_frac))
    test_df = df.iloc[test_start:].copy()
    print(f"Test set: {len(test_df)} markets (rows {test_start}–{n-1} of {n} total)\n")

    X = test_df[feature_cols].values
    raw, prob_yes, _ = predict(model, meta, calibrator, X)

    prices = test_df["current_price"].fillna(0.5).values
    if blend_alpha < 1.0:
        prob_yes = blend_alpha * prob_yes + (1.0 - blend_alpha) * prices

    prob_no = 1.0 - prob_yes
    edge_yes = prob_yes - prices
    edge_no = prob_no - (1.0 - prices)

    y_true = test_df["label"].values

    # Summary metrics
    ll = log_loss(y_true, np.clip(prob_yes, 1e-7, 1 - 1e-7))
    bs = brier_score_loss(y_true, prob_yes)
    auc = roc_auc_score(y_true, prob_yes) if len(np.unique(y_true)) > 1 else float("nan")
    edge_correct = np.where(y_true == 1, edge_yes, edge_no)
    correct = ((prob_yes > 0.5) == y_true.astype(bool)).mean()

    print("=== Summary ===")
    print(f"  AUC:               {auc:.4f}")
    print(f"  Log loss:          {ll:.4f}")
    print(f"  Brier score:       {bs:.4f}")
    print(f"  Accuracy (>0.5):   {correct:.1%}")
    print(f"  Mean edge:         {edge_correct.mean():+.4f}")
    print(f"  Median edge:       {np.median(edge_correct):+.4f}")
    print(f"  % positive edge:   {(edge_correct > 0).mean():.1%}")

    # Per-market table
    print(f"\n=== Per-market predictions (|edge| >= {args.min_edge}) ===")
    print(f"  {'correct':7}  {'outcome':7}  {'p_yes':>6}  {'price':>6}  {'edge':>7}  {'buy':>3}  slug")
    print("  " + "-" * 85)

    rows = []
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        ey = edge_yes[i]
        en = edge_no[i]
        py = prob_yes[i]
        actual = "YES" if y_true[i] == 1 else "NO"
        predicted = "YES" if py > 0.5 else "NO"
        is_correct = predicted == actual
        edge = ey if ey >= en else en
        buy_side = "YES" if ey >= en else "NO"
        rows.append((edge, i, is_correct, actual, py, prices[i], ey, en, buy_side, row.get("slug", "")))

    rows.sort(key=lambda x: abs(x[0]), reverse=True)

    shown = 0
    for edge, i, is_correct, actual, py, price, ey, en, buy_side, slug in rows:
        if abs(edge) < args.min_edge:
            continue
        tick = "✓" if is_correct else "✗"
        print(f"  {tick} {actual:7}  {actual:7}  {py:6.4f}  {price:6.4f}  {edge:+7.4f}  {buy_side:>3}  {slug}")
        shown += 1

    if shown == 0:
        print("  (no markets matched filter)")

    # Accuracy breakdown by confidence bucket
    print("\n=== Accuracy by confidence bucket ===")
    print(f"  {'bucket':20}  {'n':>4}  {'acc':>6}  {'mean_edge':>9}")
    buckets = [
        ("strong NO  (<0.25)", prob_yes < 0.25),
        ("lean NO  (0.25-0.4)", (prob_yes >= 0.25) & (prob_yes < 0.4)),
        ("toss-up  (0.4-0.6)", (prob_yes >= 0.4) & (prob_yes < 0.6)),
        ("lean YES (0.6-0.75)", (prob_yes >= 0.6) & (prob_yes < 0.75)),
        ("strong YES (>0.75)", prob_yes >= 0.75),
    ]
    for label, mask in buckets:
        if mask.sum() == 0:
            continue
        acc = ((prob_yes[mask] > 0.5) == y_true[mask].astype(bool)).mean()
        me = edge_correct[mask].mean()
        print(f"  {label:20}  {mask.sum():>4}  {acc:>6.1%}  {me:>+9.4f}")


if __name__ == "__main__":
    main()
