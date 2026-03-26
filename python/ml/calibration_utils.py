"""Post-hoc calibration: Platt, temperature, isotonic; selection by Brier + log loss."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


def fit_platt_scaler(raw_probs: np.ndarray, y: np.ndarray) -> Optional[dict[str, Any]]:
    raw_probs = np.asarray(raw_probs, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=int)
    if len(y) < 12 or len(np.unique(y)) < 2:
        return None
    lr = LogisticRegression(max_iter=2000, solver="lbfgs")
    try:
        lr.fit(raw_probs, y)
    except ValueError:
        return None
    return {
        "method": "platt",
        "coef": float(lr.coef_.ravel()[0]),
        "intercept": float(lr.intercept_.ravel()[0]),
    }


def fit_temperature_scaler(raw_probs: np.ndarray, y: np.ndarray) -> Optional[dict[str, Any]]:
    """Grid-search T to minimize Brier score (better aligned with probability quality than log loss alone)."""
    raw_probs = np.asarray(raw_probs, dtype=float)
    y = np.asarray(y, dtype=int)
    if len(y) < 8 or len(np.unique(y)) < 2:
        return None
    eps = 1e-6
    p = np.clip(raw_probs, eps, 1.0 - eps)
    logit = np.log(p / (1.0 - p))
    best_T, best_brier = 1.0, float("inf")
    for T in np.linspace(0.15, 6.0, 118):
        if T <= 0:
            continue
        z = np.clip(logit / T, -50.0, 50.0)
        p_cal = 1.0 / (1.0 + np.exp(-z))
        br = brier_score_loss(y, p_cal)
        if br < best_brier:
            best_brier, best_T = br, T
    return {"method": "temperature", "temperature": float(best_T)}


def fit_isotonic_scaler(raw_probs: np.ndarray, y: np.ndarray) -> Optional[dict[str, Any]]:
    if len(y) < 15 or len(np.unique(y)) < 2:
        return None
    raw_probs = np.asarray(raw_probs, dtype=float)
    y = np.asarray(y, dtype=int)
    try:
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(raw_probs, y)
    except ValueError:
        return None
    return {
        "method": "isotonic",
        "x_thresholds": iso.X_thresholds_.astype(float).tolist(),
        "y_thresholds": iso.y_thresholds_.astype(float).tolist(),
    }


def _clip_probs(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)


def _calibration_objectives(y: np.ndarray, p_raw: np.ndarray) -> tuple[float, float]:
    p = _clip_probs(p_raw)
    return brier_score_loss(y, p), log_loss(y, p)


_METHOD_ORDER = {"temperature": 0, "platt": 1, "isotonic": 2}


def _rank_key_oof(cal: Optional[dict[str, Any]], br: float, ll: float) -> tuple[float, float, int]:
    """OOF ranking: log loss first (match prob training objective), then Brier, then method."""
    if cal is None:
        return (ll, br, -1)
    pref = _METHOD_ORDER.get(cal["method"], 9)
    return (ll, br, pref)


def _rank_key_val(cal: Optional[dict[str, Any]], br: float, ll: float) -> tuple[float, float, int]:
    """Validation ranking: log loss first (probability quality), then Brier, then method."""
    if cal is None:
        return (ll, br, -1)
    pref = _METHOD_ORDER.get(cal["method"], 9)
    return (ll, br, pref)


def pick_best_calibrator(
    raw_probs: np.ndarray,
    y: np.ndarray,
    *,
    include_isotonic: bool = False,
) -> Optional[dict[str, Any]]:
    """
    Legacy: pick calibrator by OOF metrics only. Prefer pick_calibrator_via_validation.
    """
    raw_probs = np.asarray(raw_probs, dtype=float)
    y = np.asarray(y, dtype=int)
    base_brier, base_ll = _calibration_objectives(y, raw_probs)

    # Temperature only (+ optional isotonic): Platt on OOF often wins Brier but
    # hurts held-out log loss for this booster.
    fitters = [fit_temperature_scaler]
    if include_isotonic:
        fitters.append(fit_isotonic_scaler)

    candidates: list[dict[str, Any]] = []
    for fit_fn in fitters:
        cal = fit_fn(raw_probs, y)
        if cal:
            candidates.append(cal)

    best_cal: Optional[dict[str, Any]] = None
    best_key = _rank_key_oof(None, base_brier, base_ll)

    for cal in candidates:
        p_cal = apply_calibration(raw_probs, cal)
        br, ll = _calibration_objectives(y, p_cal)
        k = _rank_key_oof(cal, br, ll)
        if k < best_key:
            best_key = k
            best_cal = cal

    return best_cal


def pick_calibrator_via_validation(
    oof_preds: np.ndarray,
    y_train: np.ndarray,
    pred_val: np.ndarray,
    y_val: np.ndarray,
    *,
    include_isotonic: bool = False,
) -> Optional[dict[str, Any]]:
    """
    Fit Platt / temperature / (optional isotonic) on OOF train predictions, then
    choose among raw + those methods by **validation** Brier (then log loss, then
    method tiebreak). Avoids picking a map that fits OOF noise but hurts the next slice.
    """
    oof_preds = np.asarray(oof_preds, dtype=float)
    y_train = np.asarray(y_train, dtype=int)
    pred_val = np.asarray(pred_val, dtype=float)
    y_val = np.asarray(y_val, dtype=int)

    fitters = [fit_temperature_scaler]
    if include_isotonic:
        fitters.append(fit_isotonic_scaler)

    cals: list[Optional[dict[str, Any]]] = [None]
    for fit_fn in fitters:
        cal = fit_fn(oof_preds, y_train)
        if cal:
            cals.append(cal)

    best_cal: Optional[dict[str, Any]] = None
    best_key: tuple[float, float, int] | None = None

    for cal in cals:
        p_val_cal = pred_val if cal is None else apply_calibration(pred_val, cal)
        br, ll = _calibration_objectives(y_val, p_val_cal)
        k = _rank_key_val(cal, br, ll)
        if best_key is None or k < best_key:
            best_key = k
            best_cal = cal

    return best_cal


def apply_calibration(raw: np.ndarray | float, cal: Optional[dict[str, Any]]) -> np.ndarray:
    x = np.asarray(raw, dtype=float)
    if cal is None:
        return x
    if cal["method"] == "platt":
        z = cal["coef"] * x + cal["intercept"]
        z = np.clip(z, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-z))
    if cal["method"] == "temperature":
        eps = 1e-6
        p = np.clip(x, eps, 1.0 - eps)
        logit = np.log(p / (1.0 - p))
        z = np.clip(logit / cal["temperature"], -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-z))
    if cal["method"] == "isotonic":
        xt = np.asarray(cal["x_thresholds"], dtype=float)
        yt = np.asarray(cal["y_thresholds"], dtype=float)
        flat = x.ravel()
        out = np.interp(flat, xt, yt, left=float(yt[0]), right=float(yt[-1]))
        return out.reshape(x.shape)
    return x


def load_calibration(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def save_calibration(path: Path, cal: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cal, f, indent=2)
