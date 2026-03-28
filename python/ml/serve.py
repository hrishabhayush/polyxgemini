"""
ML prediction server for basketball in-game win probability.

Start with:
    uvicorn serve:app --host 127.0.0.1 --port 8766

The LightGBM model is loaded once at startup from artifacts/model.txt.
"""

import json
import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import lightgbm as lgb
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from calibration_utils import apply_calibration, load_calibration


_model: Optional[lgb.Booster] = None
_feature_names: list[str] = []
_calibrator: Optional[dict[str, Any]] = None
_blend_alpha: float = 1.0

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _feature_names, _calibrator, _blend_alpha

    model_path = ARTIFACTS_DIR / "model.txt"
    meta_path = ARTIFACTS_DIR / "model_meta.json"
    cal_path = ARTIFACTS_DIR / "calibration.json"

    if not model_path.exists():
        print(f"WARNING: {model_path} not found — /predict will return 503 until a model is trained")
    else:
        print(f"Loading model from {model_path}...")
        _model = lgb.Booster(model_file=str(model_path))
        print("Model loaded.")

    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        _feature_names = meta.get("feature_names", [])
        _blend_alpha = meta.get("blend_alpha", 1.0)
        print(f"Feature names: {_feature_names}")
        if _blend_alpha < 1.0:
            print(f"Blend alpha: {_blend_alpha} (blends model with poly_price)")

    _calibrator = load_calibration(cal_path)
    if _calibrator:
        m = _calibrator.get("method", "?")
        print(f"Loaded {m} calibrator — /predict returns calibrated P(home_win)")
    else:
        print("No calibration.json — /predict returns raw booster probabilities")

    yield


app = FastAPI(title="Basketball In-Game Forecaster", lifespan=lifespan)


class PredictRequest(BaseModel):
    time_remaining_sec: float = 2400.0
    period: int = 1
    score_diff: int = 0
    scoring_run_60s: int = 0
    scoring_run_120s: int = 0
    lead_changes_so_far: int = 0
    largest_lead: int = 0
    momentum: float = 0.0
    seed_diff: float = 0.0
    poly_price: float = 0.5
    poly_price_drift_5m: float = 0.0
    poly_volume_1m: float = 0.0
    poly_buy_fraction_5m: float = 0.5
    poly_trade_count_5m: int = 0
    has_poly: int = 0


class PredictResponse(BaseModel):
    prob_home_win: float
    prob_away_win: float
    edge_vs_market: float
    model_side: str


def _compute_derived(req: PredictRequest) -> dict[str, float]:
    """Compute interaction features to match training pipeline."""
    eps = 1e-6
    p = max(eps, min(1.0 - eps, req.poly_price))
    return {
        "logit_poly_price": math.log(p / (1.0 - p)),
        "price_x_score_diff": req.poly_price * req.score_diff,
        "time_x_score_diff": req.time_remaining_sec * req.score_diff,
        "log_time_remaining": math.log1p(req.time_remaining_sec),
        "abs_score_diff": abs(req.score_diff),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    derived = _compute_derived(req)
    row: list[float] = []
    for name in _feature_names:
        if name in derived:
            row.append(derived[name])
        else:
            row.append(float(getattr(req, name, 0.0)))
    features = np.array([row])

    raw = float(_model.predict(features)[0])
    prob_home = float(apply_calibration(np.array([raw]), _calibrator)[0])

    if _blend_alpha < 1.0:
        prob_home = _blend_alpha * prob_home + (1.0 - _blend_alpha) * req.poly_price

    prob_away = 1.0 - prob_home
    edge = prob_home - req.poly_price
    side = "HOME" if prob_home >= 0.5 else "AWAY"

    return PredictResponse(
        prob_home_win=round(prob_home, 4),
        prob_away_win=round(prob_away, 4),
        edge_vs_market=round(edge, 4),
        model_side=side,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_type": "basketball_in_game",
    }
