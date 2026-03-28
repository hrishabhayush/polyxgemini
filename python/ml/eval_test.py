"""
Evaluate the basketball in-game forecaster on the held-out test set.

Shows per-game predictions, accuracy by time-remaining bucket, and game-level
accuracy (did the model correctly predict the winner at each point in the game?).

Usage:
    python eval_test.py
    python eval_test.py --train-frac 0.7 --val-frac 0.15
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from calibration_utils import apply_calibration, load_calibration
from train import add_interaction_features, game_date_split

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = Path(__file__).parent / "artifacts"


def load_pipeline():
    model = lgb.Booster(model_file=str(ARTIFACTS / "model.txt"))
    with open(ARTIFACTS / "model_meta.json") as f:
        meta = json.load(f)
    calibrator = load_calibration(ARTIFACTS / "calibration.json")
    return model, meta, calibrator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(REPO_ROOT / "data" / "features_basketball.parquet"))
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    args = parser.parse_args()

    model, meta, calibrator = load_pipeline()
    feature_cols = meta["feature_names"]
    blend_alpha = meta.get("blend_alpha", 1.0)

    df = pd.read_parquet(args.input)
    df = add_interaction_features(df)
    df[feature_cols] = df[feature_cols].fillna(0)

    _, _, test_df = game_date_split(df, args.train_frac, args.val_frac)
    print(f"Test set: {len(test_df)} rows from {test_df['game_id'].nunique()} games\n")

    X = test_df[feature_cols].values
    raw = model.predict(X)
    prob_home = apply_calibration(raw, calibrator)

    prices = test_df["poly_price"].fillna(0.5).values
    if blend_alpha < 1.0:
        prob_home = blend_alpha * prob_home + (1.0 - blend_alpha) * prices

    y_true = test_df["label"].values

    # Summary metrics
    prob_clipped = np.clip(prob_home, 1e-7, 1 - 1e-7)
    ll = log_loss(y_true, prob_clipped)
    bs = brier_score_loss(y_true, prob_home)
    auc = roc_auc_score(y_true, prob_home) if len(np.unique(y_true)) > 1 else float("nan")
    accuracy = ((prob_home > 0.5) == y_true.astype(bool)).mean()

    print("=== Overall Metrics ===")
    print(f"  AUC:          {auc:.4f}")
    print(f"  Log loss:     {ll:.4f}")
    print(f"  Brier score:  {bs:.4f}")
    print(f"  Accuracy:     {accuracy:.1%}")

    # Accuracy by time-remaining bucket
    time_remain = test_df["time_remaining_sec"].values
    buckets = [
        ("Full game (>1800s)", time_remain > 1800),
        ("Early 2nd (900-1800s)", (time_remain > 900) & (time_remain <= 1800)),
        ("Mid 2nd (300-900s)", (time_remain > 300) & (time_remain <= 900)),
        ("Late game (60-300s)", (time_remain > 60) & (time_remain <= 300)),
        ("Final minute (<=60s)", time_remain <= 60),
    ]
    print("\n=== Accuracy by time remaining ===")
    print(f"  {'bucket':25}  {'n':>5}  {'acc':>7}  {'brier':>7}")
    print("  " + "-" * 55)
    for label, mask in buckets:
        if mask.sum() == 0:
            continue
        acc = ((prob_home[mask] > 0.5) == y_true[mask].astype(bool)).mean()
        br = brier_score_loss(y_true[mask], np.clip(prob_home[mask], 1e-7, 1 - 1e-7))
        print(f"  {label:25}  {mask.sum():>5}  {acc:>7.1%}  {br:>7.4f}")

    # Per-game accuracy: did the model predict the winner correctly at each minute?
    print("\n=== Per-game results (final-minute prediction) ===")
    print(f"  {'correct':7}  {'p_home':>6}  {'poly':>6}  {'result':>8}  game")
    print("  " + "-" * 65)

    game_groups = test_df.copy()
    game_groups["prob_home"] = prob_home
    game_correct = 0
    game_total = 0
    for gid, grp in game_groups.groupby("game_id"):
        final_row = grp.loc[grp["time_remaining_sec"].idxmin()]
        p = final_row["prob_home"]
        label = final_row["label"]
        poly_p = final_row.get("poly_price", 0.5)
        predicted_home = p > 0.5
        actual_home = label == 1
        correct = predicted_home == actual_home
        tick = "OK" if correct else "MISS"
        result = f"{'HOME' if actual_home else 'AWAY'} wins"
        home = final_row.get("home_team", "")
        away = final_row.get("away_team", "")
        game_total += 1
        if correct:
            game_correct += 1
        print(f"  {tick:7}  {p:6.3f}  {poly_p:6.3f}  {result:>8}  {away} @ {home}")

    print(f"\nGame-level accuracy: {game_correct}/{game_total} ({game_correct/game_total:.1%})")


if __name__ == "__main__":
    main()
