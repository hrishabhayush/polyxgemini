"""
Train a LightGBM binary classifier to predict market resolution (YES=1 / NO=0).

Usage:
    python train.py
    python train.py --input data/features.parquet --calibration-mode auto_oof

Post-hoc calibration uses time-series OOF predictions on the train split, then
applies the map at serve time. Default mode is isotonic (see --calibration-mode).
"""

import argparse
import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from calibration_utils import (
    apply_calibration,
    fit_isotonic_scaler,
    fit_temperature_scaler,
    pick_best_calibrator,
    pick_calibrator_via_validation,
    save_calibration,
)


def time_series_oof_predictions(
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: list[str],
    params: dict,
    num_trees: int,
    n_splits: int = 5,
) -> np.ndarray:
    """
    Out-of-fold predictions on the training set (time-ordered).
    Used to fit calibration without using the tiny validation slice only.
    """
    oof = np.zeros(len(X), dtype=float)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for tr_idx, va_idx in tscv.split(X):
        if len(tr_idx) < 25:
            oof[va_idx] = float(np.mean(y))
            continue
        dtr = lgb.Dataset(X[tr_idx], label=y[tr_idx], feature_name=feature_cols)
        fold_model = lgb.train(
            {**params, "verbose": -1},
            dtr,
            num_boost_round=max(1, num_trees),
        )
        oof[va_idx] = fold_model.predict(X[va_idx])
    return oof

FEATURE_COLS = [
    "current_price",
    "hurst_exp",
    "vol_ratio",
    "jump_result_enc",
    "trade_count",
    "total_volume",
    "log_volume",
    "kyles_lambda",
    "vpin",
    "buy_fraction",
    "wallet_hhi",
    "article_count",
    "bullish_score",
    "bearish_score",
    "sentiment_net",
    "resolution_reliability",
    "market_age_days",
    "price_distance_from_50",
]


def time_based_split(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15):
    """Split by end_date: train | val | test."""
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, prices: np.ndarray, split_name: str):
    """Print classification and trading-relevant metrics."""
    ll = log_loss(y_true, y_prob)
    bs = brier_score_loss(y_true, y_prob)
    metrics = {"log_loss": round(ll, 4), "brier_score": round(bs, 4)}

    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_prob)
        metrics["auc"] = round(auc, 4)

    # Simulated edge: for each market, edge on the correct side
    # If outcome=YES (label=1), edge = model_prob_yes - market_price
    # If outcome=NO  (label=0), edge = model_prob_no  - (1 - market_price)
    edge_yes = y_prob - prices
    edge_no = (1 - y_prob) - (1 - prices)
    edge = np.where(y_true == 1, edge_yes, edge_no)
    metrics["mean_edge"] = round(float(np.mean(edge)), 4)
    metrics["median_edge"] = round(float(np.median(edge)), 4)
    metrics["pct_positive_edge"] = round(float(np.mean(edge > 0)) * 100, 1)

    print(f"\n--- {split_name} ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Calibration: predicted prob vs observed freq in 10 bins
    try:
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
        print(f"  calibration (pred -> actual):")
        for mp, fp in zip(mean_pred, frac_pos):
            print(f"    {mp:.2f} -> {fp:.2f}")
    except ValueError:
        pass

    return metrics


def save_calibration_plot(y_true, y_prob, path):
    """Save a calibration plot to disk (optional, fails gracefully)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect")
        ax.plot(mean_pred, frac_pos, "s-", label="Model")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title("Calibration Plot (Test Set)")
        ax.legend()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved calibration plot to {path}")
    except Exception as e:
        print(f"Could not save calibration plot: {e}")


def main():
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Train LightGBM market resolution model")
    parser.add_argument("--input", default=str(repo_root / "data" / "features.parquet"))
    parser.add_argument("--artifacts", default=str(repo_root / "python" / "ml" / "artifacts"))
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Skip post-hoc calibration; use raw LightGBM probabilities only",
    )
    parser.add_argument(
        "--cal-oof-splits",
        type=int,
        default=5,
        help="TimeSeriesSplit folds for OOF predictions used to fit the calibrator (default: 5)",
    )
    parser.add_argument(
        "--cal-isotonic",
        action="store_true",
        help="Allow isotonic calibration (can tie-collapse probs; default: Platt+temperature only)",
    )
    parser.add_argument(
        "--calibration-mode",
        choices=["auto_oof", "auto", "temperature", "isotonic", "none"],
        default="isotonic",
        help=(
            "isotonic = OOF isotonic map (default; good log loss/Brier on small data); "
            "auto_oof = pick raw/temp/(iso if --cal-isotonic) by OOF log loss; "
            "auto = choose vs raw on validation; temperature = OOF T only; none = raw"
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.artifacts, exist_ok=True)

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Verify required columns exist
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"WARNING: missing feature columns (will be dropped): {missing}")
    feature_cols = available

    if "label" not in df.columns:
        print("ERROR: 'label' column not found. Run load_features.py first.")
        return

    # Drop rows with all-NaN features
    df = df.dropna(subset=["label"])
    df[feature_cols] = df[feature_cols].fillna(0)

    train_df, val_df, test_df = time_based_split(df, args.train_frac, args.val_frac)
    print(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    if len(train_df) < 10:
        print("ERROR: Not enough training data. Export more resolved markets first.")
        return

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["label"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=train_data)

    # Regularized booster: shallower trees + larger leaves reduce extreme raw probs
    # on small tabular data (helps both raw scores and any post-hoc map).
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.035,
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 25,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.4,
        "reg_lambda": 1.5,
        "verbose": -1,
        "seed": 42,
    }

    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50),
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    # Evaluate on all splits
    prices_train = train_df["current_price"].fillna(0.5).values
    prices_val = val_df["current_price"].fillna(0.5).values
    prices_test = test_df["current_price"].fillna(0.5).values

    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    evaluate(y_train, pred_train, prices_train, "Train (raw)")
    evaluate(y_val, pred_val, prices_val, "Validation (raw)")

    cal_path = os.path.join(args.artifacts, "calibration.json")
    calibrator = None
    if not args.no_calibration and args.calibration_mode != "none":
        n_trees = model.num_trees()
        print(
            f"\nFitting calibrator on time-series OOF train predictions "
            f"({args.cal_oof_splits} folds, {n_trees} trees per fold)..."
        )
        oof_train = time_series_oof_predictions(
            X_train,
            y_train,
            feature_cols,
            params,
            n_trees,
            n_splits=args.cal_oof_splits,
        )
        if args.calibration_mode == "temperature":
            calibrator = fit_temperature_scaler(oof_train, y_train)
            if calibrator:
                save_calibration(Path(cal_path), calibrator)
                print(
                    f"Saved OOF temperature calibrator (T={calibrator['temperature']:.4f}) to {cal_path}"
                )
            else:
                print("WARNING: Could not fit temperature on OOF; using raw probabilities.")
        elif args.calibration_mode == "isotonic":
            calibrator = fit_isotonic_scaler(oof_train, y_train)
            if calibrator:
                save_calibration(Path(cal_path), calibrator)
                print(f"Saved OOF isotonic calibrator to {cal_path}")
            else:
                print("WARNING: Could not fit isotonic on OOF; using raw probabilities.")
        elif args.calibration_mode == "auto_oof":
            calibrator = pick_best_calibrator(
                oof_train, y_train, include_isotonic=args.cal_isotonic
            )
            if calibrator:
                save_calibration(Path(cal_path), calibrator)
                print(
                    f"Saved calibrator ({calibrator['method']}) to {cal_path} "
                    "(OOF log loss / Brier vs raw)"
                )
            else:
                print("Calibration skipped (auto_oof): raw best on OOF.")
                stale = Path(cal_path)
                if stale.exists():
                    stale.unlink()
                    print(f"Removed {cal_path} so serve uses raw probabilities")
        else:
            calibrator = pick_calibrator_via_validation(
                oof_train,
                y_train,
                pred_val,
                y_val,
                include_isotonic=args.cal_isotonic,
            )
            if calibrator:
                save_calibration(Path(cal_path), calibrator)
                print(
                    f"Saved calibrator ({calibrator['method']}) to {cal_path} "
                    "(OOF fit; chosen vs raw by validation log loss / Brier)"
                )
            else:
                print("Calibration skipped (auto): raw best on validation.")
                stale = Path(cal_path)
                if stale.exists():
                    stale.unlink()
                    print(f"Removed {cal_path} so serve uses raw probabilities")

        if calibrator:
            pred_val_cal = apply_calibration(pred_val, calibrator)
            pred_test_cal = apply_calibration(pred_test, calibrator)
            evaluate(y_val, pred_val_cal, prices_val, "Validation (calibrated)")
            evaluate(y_test, pred_test, prices_test, "Test (raw, for comparison)")
            test_metrics = evaluate(y_test, pred_test_cal, prices_test, "Test (calibrated)")
        else:
            test_metrics = evaluate(y_test, pred_test, prices_test, "Test (raw)")
    else:
        test_metrics = evaluate(y_test, pred_test, prices_test, "Test (raw)")
        stale_cal = Path(cal_path)
        if stale_cal.exists():
            stale_cal.unlink()
            print(f"Removed {cal_path} (--no-calibration or --calibration-mode none)")

    # Feature importance
    importance = sorted(
        zip(feature_cols, model.feature_importance("gain")),
        key=lambda x: x[1],
        reverse=True,
    )
    print("\nFeature importance (gain):")
    for name, imp in importance:
        print(f"  {name}: {imp:.1f}")

    # Save artifacts
    model_path = os.path.join(args.artifacts, "model.txt")
    model.save_model(model_path)
    print(f"\nSaved model to {model_path}")

    meta = {
        "feature_names": feature_cols,
        "params": params,
        "num_boost_round": model.best_iteration,
        "test_metrics": test_metrics,
        "calibration": (
            calibrator["method"]
            if calibrator
            else (
                "none"
                if args.no_calibration or args.calibration_mode == "none"
                else "raw_best"
            )
        ),
        "calibration_mode": args.calibration_mode,
    }
    meta_path = os.path.join(args.artifacts, "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {meta_path}")

    # Calibration plot (prefer calibrated curve when available)
    plot_probs = pred_test
    if calibrator is not None:
        plot_probs = apply_calibration(pred_test, calibrator)
    save_calibration_plot(
        y_test,
        plot_probs,
        os.path.join(args.artifacts, "calibration.png"),
    )


if __name__ == "__main__":
    main()
