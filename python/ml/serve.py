"""
ML prediction server for market resolution probabilities.

Start with:
    uvicorn serve:app --host 127.0.0.1 --port 8766

The LightGBM model is loaded once at startup from artifacts/model.txt.
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


_model: Optional[lgb.Booster] = None
_feature_names: list[str] = []

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _feature_names

    model_path = ARTIFACTS_DIR / "model.txt"
    meta_path = ARTIFACTS_DIR / "model_meta.json"

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
        print(f"Feature names: {_feature_names}")

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


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = np.array([[
        getattr(req, name, 0.0) for name in _feature_names
    ]])

    prob_yes = float(_model.predict(features)[0])
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
