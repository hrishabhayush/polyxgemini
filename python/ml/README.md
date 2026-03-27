# ML Pipeline — Market Resolution Predictor

Predicts whether a Polymarket binary (YES/NO) market will resolve YES, using
LightGBM trained on historical resolved markets.

Run all commands from the **repo root** unless noted otherwise.

---

## Setup (once)

```bash
source .venv-ocr/bin/activate
pip install -U pandas numpy scikit-learn lightgbm pyarrow fastapi uvicorn optuna
```

---

## 1. Build feature table from resolved snapshots

Requires `data/snapshots/resolved_*.jsonl` to exist.

```bash
python python/ml/load_features.py
```

Output: `data/features.parquet`

---

## 2. Train the model

```bash
python python/ml/train.py
```

With Optuna hyperparameter search (~80 trials, takes a few minutes):

```bash
python python/ml/train.py --tune --tune-trials 80
```

Artifacts written to `python/ml/artifacts/`:
- `model.txt` — LightGBM booster
- `model_meta.json` — feature names, params, test metrics
- `calibration.json` — post-hoc calibration map (if used)

---

## 3. Evaluate on held-out test data

Shows per-market predictions vs actual outcomes for the last 15% of resolved markets.

```bash
python python/ml/eval_test.py
```

Filter to only high-confidence calls:

```bash
python python/ml/eval_test.py --min-edge 0.3
```

---

## 4. Start the prediction server

Run in a dedicated terminal (keep it running):

```bash
cd python/ml
source ../../.venv-ocr/bin/activate
uvicorn serve:app --host 127.0.0.1 --port 8766
```

---

## 5. Predict a specific market (with reasoning)

Look up any market by slug, URL, or search query and get a full prediction report.
**Requires the server from Step 4 to be running.**

```bash
# By slug
go run ./cmd/predict -market "will-the-iranian-regime-fall-by-june-30"

# By search query
go run ./cmd/predict -market "will bitcoin hit 100k by june"

# By Polymarket URL
go run ./cmd/predict -market "https://polymarket.com/event/will-trump-visit-china-by-april-30"
```

Output includes: model probability, edge vs crowd price, signal strength, and
per-feature reasoning (VPIN, buy pressure, Hurst, wallet concentration, etc.).

---

## 6. Score all live markets

Fetches the latest active snapshot and scores every binary market against the model.

```bash
# First get a fresh snapshot (run once, takes ~30 min for 100 markets)
go run ./cmd/snapshot -max 100

# Then score (server must be running)
python python/ml/score_live.py
```

Output: top 15 markets ranked by edge, filtered to trade_count ≥ 50 and volume ≥ $5000.

---

## Retraining workflow

When you have new resolved market data:

```bash
# 1. Export new resolved markets
go run ./cmd/export -max 500 -lookback 365 -sentiment-window 1m

# 2. Rebuild features
python python/ml/load_features.py

# 3. Retrain
python python/ml/train.py --tune

# 4. Restart the server
# (Ctrl+C the running uvicorn, then re-run Step 4)
```

---

## File reference

| File | Purpose |
|---|---|
| `load_features.py` | Builds `features.parquet` from resolved JSONL snapshots |
| `train.py` | Trains LightGBM model, saves artifacts |
| `serve.py` | FastAPI prediction server |
| `score_live.py` | Scores latest active snapshot, prints top edges |
| `eval_test.py` | Evaluates model on held-out test set |
| `calibration_utils.py` | Calibration helpers (Platt, temperature, isotonic) |
| `artifacts/model.txt` | Trained LightGBM model |
| `artifacts/model_meta.json` | Feature names, hyperparams, test metrics |
| `cmd/predict/main.go` | Go CLI for single-market prediction with reasoning |
