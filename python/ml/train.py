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

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

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

BASE_FEATURE_COLS = [
    "current_price",
    "hurst_exp",
    "has_hurst",
    "vol_ratio",
    "jump_result_enc",
    "trade_count",
    "log_trade_count",
    "total_volume",
    "log_volume",
    "avg_trade_size",
    "log_avg_trade_size",
    "kyles_lambda",
    "vpin",
    "buy_fraction",
    "wallet_hhi",
    "market_age_days",
    "price_distance_from_50",
]

INTERACTION_COLS = [
    "price_x_log_volume",
    "price_x_hhi",
    "price_x_vpin",
    "price_x_buy_fraction",
    "hhi_x_buy_fraction",
    "vpin_x_log_kyles",
    "log_trade_x_vpin",
    "logit_price",
]


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive interaction / nonlinear features from base columns."""
    df = df.copy()
    eps = 1e-6
    df["price_x_log_volume"] = df["current_price"] * df["log_volume"]
    df["price_x_hhi"] = df["current_price"] * df["wallet_hhi"]
    df["price_x_vpin"] = df["current_price"] * df["vpin"]
    df["price_x_buy_fraction"] = df["current_price"] * df["buy_fraction"]
    df["hhi_x_buy_fraction"] = df["wallet_hhi"] * df["buy_fraction"]
    # Combined order-flow toxicity: VPIN × log(Kyle's lambda)
    df["vpin_x_log_kyles"] = df["vpin"] * np.log1p(df["kyles_lambda"].abs())
    # Volume-weighted VPIN
    df["log_trade_x_vpin"] = df.get("log_trade_count", np.log1p(df["trade_count"])) * df["vpin"]
    p = df["current_price"].clip(eps, 1.0 - eps)
    df["logit_price"] = np.log(p / (1.0 - p))
    return df


def tune_blend_alpha(
    model_probs: np.ndarray,
    market_prices: np.ndarray,
    y: np.ndarray,
) -> float:
    """Find alpha in [0,1] minimizing Brier: final = alpha*model + (1-alpha)*market."""
    best_alpha, best_brier = 0.5, float("inf")
    for alpha in np.linspace(0.0, 1.0, 101):
        blended = alpha * model_probs + (1.0 - alpha) * market_prices
        blended = np.clip(blended, 1e-6, 1.0 - 1e-6)
        br = brier_score_loss(y, blended)
        if br < best_brier:
            best_brier, best_alpha = br, alpha
    return round(float(best_alpha), 3)


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


def tune_hyperparams(
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: list[str],
    n_trials: int = 60,
    n_splits: int = 5,
) -> dict:
    """Optuna TPE search over LightGBM hyperparams using TimeSeriesSplit CV log loss."""
    if not HAS_OPTUNA:
        raise RuntimeError("optuna not installed — run: pip install optuna")

    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbose": -1,
            "seed": 42,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 31),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "min_child_samples": trial.suggest_int("min_child_samples", 15, 60),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
        }
        scores = []
        for tr_idx, va_idx in tscv.split(X):
            if len(tr_idx) < 25:
                continue
            dtr = lgb.Dataset(X[tr_idx], label=y[tr_idx], feature_name=feature_cols)
            dva = lgb.Dataset(X[va_idx], label=y[va_idx], feature_name=feature_cols, reference=dtr)
            m = lgb.train(
                params,
                dtr,
                num_boost_round=600,
                valid_sets=[dva],
                callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)],
            )
            preds = np.clip(m.predict(X[va_idx]), 1e-7, 1 - 1e-7)
            scores.append(log_loss(y[va_idx], preds))
        return float(np.mean(scores)) if scores else 1.0

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best.update({"objective": "binary", "metric": "binary_logloss", "verbose": -1, "seed": 42})
    print(f"Optuna best CV log loss: {study.best_value:.4f}")
    print(f"Best params: {best}")
    return best


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
        "--tune",
        action="store_true",
        help="Run Optuna hyperparameter search (requires: pip install optuna). ~60 trials.",
    )
    parser.add_argument(
        "--tune-trials",
        type=int,
        default=60,
        help="Number of Optuna trials (default: 60)",
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

    df = add_interaction_features(df)

    all_cols = BASE_FEATURE_COLS + INTERACTION_COLS
    available = [c for c in all_cols if c in df.columns]
    missing = [c for c in all_cols if c not in df.columns]
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
    default_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.03,
        "num_leaves": 16,
        "max_depth": 4,
        "min_child_samples": 25,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.3,
        "reg_lambda": 2.0,
        "min_split_gain": 0.05,
        "verbose": -1,
        "seed": 42,
    }

    if args.tune:
        if not HAS_OPTUNA:
            print("WARNING: --tune requested but optuna not installed. pip install optuna. Using defaults.")
            params = default_params
        else:
            print(f"\nRunning Optuna HPO ({args.tune_trials} trials, {args.cal_oof_splits}-fold CV)...")
            params = tune_hyperparams(
                train_df[feature_cols].values,
                train_df["label"].values,
                feature_cols,
                n_trials=args.tune_trials,
                n_splits=args.cal_oof_splits,
            )
    else:
        params = default_params

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
    oof_train = None
    pred_val_cal = pred_val
    pred_test_cal = pred_test
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

    # --- Market-price blend ---
    # The market price is already a probability estimate (the crowd).
    # Blending model with market price reduces overconfidence.
    n_trees = model.num_trees()
    if oof_train is None:
        print(
            f"\nComputing OOF for blend alpha "
            f"({args.cal_oof_splits} folds, {n_trees} trees)..."
        )
        oof_train = time_series_oof_predictions(
            X_train, y_train, feature_cols, params, n_trees,
            n_splits=args.cal_oof_splits,
        )

    oof_for_blend = oof_train
    if calibrator:
        oof_for_blend = apply_calibration(oof_train, calibrator)

    oof_prices = train_df["current_price"].fillna(0.5).values
    blend_alpha = tune_blend_alpha(oof_for_blend, oof_prices, y_train)
    print(f"\nBlend alpha (tuned on OOF): {blend_alpha}")
    print(f"  final_prob = {blend_alpha} * model + {1.0 - blend_alpha:.3f} * market_price")

    if blend_alpha < 1.0:
        def final_pred(m, p):
            return blend_alpha * m + (1.0 - blend_alpha) * p

        if calibrator:
            pred_test_final = final_pred(pred_test_cal, prices_test)
            pred_val_final = final_pred(pred_val_cal, prices_val)
        else:
            pred_test_final = final_pred(pred_test, prices_test)
            pred_val_final = final_pred(pred_val, prices_val)

        evaluate(y_val, pred_val_final, prices_val, "Validation (blended)")
        blended_metrics = evaluate(y_test, pred_test_final, prices_test, "Test (blended)")

        if blended_metrics["brier_score"] < test_metrics["brier_score"]:
            test_metrics = blended_metrics
            print("  -> Using blended as final (better Brier than calibrated/raw).")
        else:
            blend_alpha = 1.0
            print("  -> Blend did not improve test Brier; using calibrated/raw (alpha=1.0).")

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
        "blend_alpha": blend_alpha,
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
