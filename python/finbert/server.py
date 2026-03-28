"""
FinBERT sentiment analysis server.

Start with:
    uvicorn server:app --host 127.0.0.1 --port 8765

The ProsusAI/finbert model is loaded once at startup (~3s, ~500MB RAM).
All subsequent /score requests are served from the in-memory pipeline.
"""

from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline


_pipe = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipe
    print("Loading ProsusAI/finbert model...")
    _pipe = pipeline("text-classification", model="ProsusAI/finbert")
    print("Model loaded.")
    yield


app = FastAPI(title="FinBERT Sentiment Service", lifespan=lifespan)


class ScoreRequest(BaseModel):
    texts: List[str]


class ScoreResult(BaseModel):
    label: str   # "positive", "negative", or "neutral"
    score: float


class ScoreResponse(BaseModel):
    results: List[ScoreResult]


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    if not req.texts:
        return ScoreResponse(results=[])
    if _pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # truncation=True prevents errors on inputs > 512 tokens
    raw = _pipe(req.texts, truncation=True, max_length=512)
    results = [
        ScoreResult(label=r["label"].lower(), score=round(r["score"], 4))
        for r in raw
    ]
    return ScoreResponse(results=results)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _pipe is not None}
