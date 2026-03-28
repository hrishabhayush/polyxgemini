"""
Train a LightGBM binary classifier to predict basketball game outcomes.

Predicts P(home team wins) at each minute of game time, using NCAA play-by-play
state features and Polymarket price/trade features.

Usage:
    python train.py
    python train.py --input data/features_basketball.parquet --tune --tune-trials 80
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
from sklearn.model_selection import GroupKFold

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

BASE_FEATURE_COLS = [
    "time_remaining_sec",
    "log_time_remaining",
    "period",
    "score_diff",
    "abs_score_diff",
    "scoring_run_60s",
    "scoring_run_120s",
    "lead_changes_so_far",
    "largest_lead",
    "momentum",
    "seed_diff",
    "poly_price",
    "poly_price_drift_5m",
    "poly_volume_1m",
    "poly_buy_fraction_5m",
    "poly_trade_count_5m",
    "has_poly",
]

INTERACTION_COLS = [
    "price_x_score_diff",
    "time_x_score_diff",
    "logit_poly_price",
]


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive interaction / nonlinear features from base columns."""
    df = df.copy()
    eps = 1e-6
    p = df["poly_price"].clip(eps, 1.0 - eps)
    df["logit_poly_price"] = np.log(p / (1.0 - p))
    df["price_x_score_diff"] = df["poly_price"] * df["score_diff"]
    df["time_x_score_diff"] = df["time_remaining_sec"] * df["score_diff"]
    return df


def game_date_split(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15):
    """Split by game date, keeping all rows from a game in the same split."""
    game_dates = df.groupby("game_id")["game_date"].first().sort_values()
    unique_games = game_dates.index.tolist()
    n = len(unique_games)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_games = set(unique_games[:train_end])
    val_games = set(unique_games[train_end:val_end])
    test_games = set(unique_games[val_end:])

    return (
        df[df["game_id"].isin(train_games)].copy(),
        df[df["game_id"].isin(val_games)].copy(),
        df[df["game_id"].isin(test_games)].copy(),
    )


def game_oof_predictions(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_cols: list[str],
    params: dict,
    num_trees: int,
    n_splits: int = 5,
) -> np.ndarray:
    """Out-of-fold predictions using GroupKFold (no game leakage)."""
    oof = np.zeros(len(X), dtype=float)
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    for tr_idx, va_idx in gkf.split(X, y, groups):
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


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, prices: np.ndarray, split_name: str):
    """Print classification metrics."""
    y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
    ll = log_loss(y_true, y_prob_clipped)
    bs = brier_score_loss(y_true, y_prob)
    metrics = {"log_loss": round(ll, 4), "brier_score": round(bs, 4)}

    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_prob)
        metrics["auc"] = round(auc, 4)

    accuracy = ((y_prob > 0.5) == y_true.astype(bool)).mean()
    metrics["accuracy"] = round(float(accuracy), 4)

    edge_yes = y_prob - prices
    edge_no = (1 - y_prob) - (1 - prices)
    edge = np.where(y_true == 1, edge_yes, edge_no)
    metrics["mean_edge"] = round(float(np.mean(edge)), 4)

    print(f"\n--- {split_name} ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    try:
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=8, strategy="quantile")
        print("  calibration (pred -> actual):")
        for mp, fp in zip(mean_pred, frac_pos):
            print(f"    {mp:.2f} -> {fp:.2f}")
    except ValueError:
        pass

    return metrics


def save_calibration_plot(y_true, y_prob, path):
    """Save a calibration plot to disk."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=8, strategy="quantile")
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
    groups: np.ndarray,
    feature_cols: list[str],
    n_trials: int = 60,
    n_splits: int = 5,
) -> dict:
    """Optuna search over LightGBM hyperparams using GroupKFold CV."""
    if not HAS_OPTUNA:
        raise RuntimeError("optuna not installed — run: pip install optuna")

    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbose": -1,
            "seed": 42,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 48),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
        }
        scores = []
        for tr_idx, va_idx in gkf.split(X, y, groups):
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

    parser = argparse.ArgumentParser(description="Train basketball in-game forecaster")
    parser.add_argument("--input", default=str(repo_root / "data" / "features_basketball.parquet"))
    parser.add_argument("--artifacts", default=str(repo_root / "python" / "ml" / "artifacts"))
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--no-calibration", action="store_true")
    parser.add_argument("--cal-oof-splits", type=int, default=5)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--tune-trials", type=int, default=60)
    parser.add_argument(
        "--calibration-mode",
        choices=["auto_oof", "auto", "temperature", "isotonic", "none"],
        default="isotonic",
    )
    args = parser.parse_args()

    os.makedirs(args.artifacts, exist_ok=True)

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"Games: {df['game_id'].nunique()}")

    df = add_interaction_features(df)

    all_cols = BASE_FEATURE_COLS + INTERACTION_COLS
    available = [c for c in all_cols if c in df.columns]
    missing = [c for c in all_cols if c not in df.columns]
    if missing:
        print(f"WARNING: missing feature columns: {missing}")
    feature_cols = available

    if "label" not in df.columns:
        print("ERROR: 'label' column not found. Run load_features.py first.")
        return

    df = df.dropna(subset=["label"])
    df[feature_cols] = df[feature_cols].fillna(0)

    train_df, val_df, test_df = game_date_split(df, args.train_frac, args.val_frac)
    print(f"Split: train={len(train_df)} ({train_df['game_id'].nunique()} games), "
          f"val={len(val_df)} ({val_df['game_id'].nunique()} games), "
          f"test={len(test_df)} ({test_df['game_id'].nunique()} games)")

    if len(train_df) < 10:
        print("ERROR: Not enough training data.")
        return

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    groups_train = train_df["game_id"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["label"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=train_data)

    default_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.03,
        "num_leaves": 24,
        "max_depth": 5,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.2,
        "reg_lambda": 1.5,
        "min_split_gain": 0.05,
        "verbose": -1,
        "seed": 42,
    }

    if args.tune:
        if not HAS_OPTUNA:
            print("WARNING: --tune requested but optuna not installed. Using defaults.")
            params = default_params
        else:
            print(f"\nRunning Optuna HPO ({args.tune_trials} trials)...")
            params = tune_hyperparams(
                X_train, y_train, groups_train, feature_cols,
                n_trials=args.tune_trials, n_splits=args.cal_oof_splits,
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

    prices_train = train_df["poly_price"].fillna(0.5).values
    prices_val = val_df["poly_price"].fillna(0.5).values
    prices_test = test_df["poly_price"].fillna(0.5).values

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
        print(f"\nFitting calibrator on GroupKFold OOF predictions "
              f"({args.cal_oof_splits} folds, {n_trees} trees)...")
        oof_train = game_oof_predictions(
            X_train, y_train, groups_train, feature_cols, params, n_trees,
            n_splits=args.cal_oof_splits,
        )
        if args.calibration_mode == "temperature":
            calibrator = fit_temperature_scaler(oof_train, y_train)
        elif args.calibration_mode == "isotonic":
            calibrator = fit_isotonic_scaler(oof_train, y_train)
        elif args.calibration_mode == "auto_oof":
            calibrator = pick_best_calibrator(oof_train, y_train)
        else:
            calibrator = pick_calibrator_via_validation(
                oof_train, y_train, pred_val, y_val,
            )

        if calibrator:
            save_calibration(Path(cal_path), calibrator)
            print(f"Saved {calibrator['method']} calibrator to {cal_path}")
            pred_val_cal = apply_calibration(pred_val, calibrator)
            pred_test_cal = apply_calibration(pred_test, calibrator)
            evaluate(y_val, pred_val_cal, prices_val, "Validation (calibrated)")
            evaluate(y_test, pred_test, prices_test, "Test (raw, for comparison)")
            test_metrics = evaluate(y_test, pred_test_cal, prices_test, "Test (calibrated)")
        else:
            print("Calibration skipped — raw best.")
            stale = Path(cal_path)
            if stale.exists():
                stale.unlink()
            test_metrics = evaluate(y_test, pred_test, prices_test, "Test (raw)")
    else:
        test_metrics = evaluate(y_test, pred_test, prices_test, "Test (raw)")
        stale_cal = Path(cal_path)
        if stale_cal.exists():
            stale_cal.unlink()

    # Blend with Polymarket price
    n_trees = model.num_trees()
    if oof_train is None:
        oof_train = game_oof_predictions(
            X_train, y_train, groups_train, feature_cols, params, n_trees,
            n_splits=args.cal_oof_splits,
        )

    oof_for_blend = apply_calibration(oof_train, calibrator) if calibrator else oof_train
    blend_alpha = tune_blend_alpha(oof_for_blend, prices_train, y_train)
    print(f"\nBlend alpha (tuned on OOF): {blend_alpha}")
    print(f"  final_prob = {blend_alpha} * model + {1.0 - blend_alpha:.3f} * poly_price")

    if blend_alpha < 1.0:
        def final_pred(m, p):
            return blend_alpha * m + (1.0 - blend_alpha) * p

        pred_test_final = final_pred(pred_test_cal if calibrator else pred_test, prices_test)
        pred_val_final = final_pred(pred_val_cal if calibrator else pred_val, prices_val)

        evaluate(y_val, pred_val_final, prices_val, "Validation (blended)")
        blended_metrics = evaluate(y_test, pred_test_final, prices_test, "Test (blended)")

        if blended_metrics["brier_score"] < test_metrics["brier_score"]:
            test_metrics = blended_metrics
            print("  -> Using blended as final.")
        else:
            blend_alpha = 1.0
            print("  -> Blend did not improve; using model only (alpha=1.0).")

    importance = sorted(
        zip(feature_cols, model.feature_importance("gain")),
        key=lambda x: x[1],
        reverse=True,
    )
    print("\nFeature importance (gain):")
    for name, imp in importance:
        print(f"  {name}: {imp:.1f}")

    model_path = os.path.join(args.artifacts, "model.txt")
    model.save_model(model_path)
    print(f"\nSaved model to {model_path}")

    meta = {
        "feature_names": feature_cols,
        "params": params,
        "num_boost_round": model.best_iteration,
        "test_metrics": test_metrics,
        "calibration": calibrator["method"] if calibrator else "raw_best",
        "calibration_mode": args.calibration_mode,
        "blend_alpha": blend_alpha,
        "model_type": "basketball_in_game",
    }
    meta_path = os.path.join(args.artifacts, "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {meta_path}")

    plot_probs = apply_calibration(pred_test, calibrator) if calibrator else pred_test
    save_calibration_plot(y_test, plot_probs, os.path.join(args.artifacts, "calibration.png"))


if __name__ == "__main__":
    main()
