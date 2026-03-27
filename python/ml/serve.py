"""
ML prediction server for market resolution probabilities.

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
            print(f"Blend alpha: {_blend_alpha} (blends model with market price)")

    _calibrator = load_calibration(cal_path)
    if _calibrator:
        m = _calibrator.get("method", "?")
        print(f"Loaded {m} calibrator from {cal_path} — /predict returns calibrated P(Yes)")
    else:
        print("No calibration.json — /predict returns raw booster probabilities")

    yield


app = FastAPI(title="Market Resolution Predictor", lifespan=lifespan)


class PredictRequest(BaseModel):
    current_price: float = 0.0
    hurst_exp: float = 0.0
    vol_ratio: float = 0.0
    jump_result_enc: int = 0
    trade_count: int = 0
    total_volume: float = 0.0
    log_volume: float = 0.0
    kyles_lambda: float = 0.0
    vpin: float = 0.0
    buy_fraction: float = 0.0
    wallet_hhi: float = 0.0
    article_count: int = 0
    bullish_score: float = 0.0
    bearish_score: float = 0.0
    sentiment_net: float = 0.0
    resolution_reliability: float = 0.0
    market_age_days: float = 0.0
    price_distance_from_50: float = 0.0


class PredictResponse(BaseModel):
    prob_yes: float
    prob_no: float
    edge_yes: float
    edge_no: float


def _compute_derived(req: PredictRequest) -> dict[str, float]:
    """Compute interaction features to match training pipeline."""
    eps = 1e-6
    p = max(eps, min(1.0 - eps, req.current_price))
    return {
        "price_x_log_volume": req.current_price * req.log_volume,
        "price_x_hhi": req.current_price * req.wallet_hhi,
        "price_x_vpin": req.current_price * req.vpin,
        "hhi_x_buy_fraction": req.wallet_hhi * req.buy_fraction,
        "logit_price": math.log(p / (1.0 - p)),
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

    raw_yes = float(_model.predict(features)[0])
    prob_yes = float(apply_calibration(np.array([raw_yes]), _calibrator)[0])

    if _blend_alpha < 1.0:
        prob_yes = _blend_alpha * prob_yes + (1.0 - _blend_alpha) * req.current_price

    prob_no = 1.0 - prob_yes

    edge_yes = prob_yes - req.current_price
    edge_no = prob_no - (1.0 - req.current_price)

    return PredictResponse(
        prob_yes=round(prob_yes, 4),
        prob_no=round(prob_no, 4),
        edge_yes=round(edge_yes, 4),
        edge_no=round(edge_no, 4),
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}
