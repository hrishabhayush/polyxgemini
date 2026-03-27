# ML Pipeline — Basketball In-Game Win Probability

Predicts P(home team wins) at every minute of an NCAA March Madness game,
using LightGBM trained on play-by-play state + Polymarket price/trade features.

Run all commands from the **repo root** unless noted otherwise.

---

## Setup (once)

```bash
source .venv-ocr/bin/activate
pip install -U -r python/ml/requirements.txt
pip install optuna  # optional, for hyperparameter tuning
```

---

## 1. Fetch game data (NCAA PBP + Polymarket)

Downloads play-by-play and market data for all March Madness tournament games.

```bash
python python/ml/fetch_games.py
```

Options:
```bash
python python/ml/fetch_games.py --start 2026-03-13 --end 2026-03-27 --out data/games
```

Output: `data/games/{gameId}.json` (one file per game)

---

## 2. Build feature table

Replays each game's PBP, resamples to 1-minute intervals, merges Polymarket
price/trade features at each timestamp.

```bash
python python/ml/load_features.py
```

Output: `data/features_basketball.parquet`

---

## 3. Train the model

```bash
python python/ml/train.py
```

With Optuna hyperparameter search:

```bash
python python/ml/train.py --tune --tune-trials 80
```

Artifacts written to `python/ml/artifacts/`:
- `model.txt` — LightGBM booster
- `model_meta.json` — feature names, params, test metrics
- `calibration.json` — post-hoc calibration map (if used)

---

## 4. Evaluate on held-out test games

Shows accuracy by time-remaining bucket and per-game predictions.

```bash
python python/ml/eval_test.py
```

---

## 5. Start the prediction server

Run in a dedicated terminal (keep it running):

```bash
cd python/ml
source ../../.venv-ocr/bin/activate
uvicorn serve:app --host 127.0.0.1 --port 8766
```

### Prediction API

**POST /predict**

```json
{
  "time_remaining_sec": 600,
  "period": 2,
  "score_diff": 5,
  "scoring_run_60s": 3,
  "scoring_run_120s": 7,
  "lead_changes_so_far": 4,
  "largest_lead": 12,
  "momentum": 3.5,
  "seed_diff": -3,
  "poly_price": 0.65,
  "poly_price_drift_5m": 0.02,
  "poly_volume_1m": 500,
  "poly_buy_fraction_5m": 0.6,
  "poly_trade_count_5m": 15,
  "has_poly": 1
}
```

Response:
```json
{
  "prob_home_win": 0.7234,
  "prob_away_win": 0.2766,
  "edge_vs_market": 0.0734,
  "model_side": "HOME"
}
```

---

## 6. Predict a single live game

Run this while the server is up to fetch NCAA state (+ optional Polymarket),
build one aligned feature snapshot, and call `/predict`.

```bash
# By NCAA game id
python python/ml/predict_live_game.py --game-id 6534602

# By date + team hints (resolves game id from scoreboard)
python python/ml/predict_live_game.py --date 2026-03-20 --away UCF --home UCLA --require-bracket

# From an existing saved game JSON (offline NCAA/Poly fetch path)
python python/ml/predict_live_game.py --from-json data/games/6534602.json

# Skip Polymarket and predict from NCAA game state only
python python/ml/predict_live_game.py --game-id 6534602 --no-poly
```

Optional:

```bash
python python/ml/predict_live_game.py --game-id 6534602 --server http://127.0.0.1:8766 --poly-slug cbb-ucf-ucla-2026-03-20
```

---

## Retraining workflow

When new games are available:

```bash
# 1. Fetch new games
python python/ml/fetch_games.py --start 2026-03-28 --end 2026-04-07

# 2. Rebuild features (reads all data/games/*.json)
python python/ml/load_features.py

# 3. Retrain
python python/ml/train.py --tune

# 4. Restart the server
```

---

## Feature reference

### Game state (from NCAA PBP)
| Feature | Description |
|---|---|
| `time_remaining_sec` | Seconds left in the game |
| `period` | 1 (1st half), 2 (2nd half), 3+ (OT) |
| `score_diff` | Home score minus away score |
| `scoring_run_60s` | Net scoring run in last 60s of game clock |
| `scoring_run_120s` | Net scoring run in last 120s |
| `lead_changes_so_far` | Cumulative lead changes |
| `largest_lead` | Max lead by either team so far |
| `momentum` | EMA of recent scoring differential |
| `seed_diff` | Home seed minus away seed |

### Polymarket (matched by timestamp)
| Feature | Description |
|---|---|
| `poly_price` | Moneyline price for home team |
| `poly_price_drift_5m` | Price change over last 5 minutes |
| `poly_volume_1m` | Trade volume in last 1 minute |
| `poly_buy_fraction_5m` | Fraction of buys in last 5 minutes |
| `poly_trade_count_5m` | Number of trades in last 5 minutes |
| `has_poly` | 1 if Polymarket data available |

### Derived
| Feature | Description |
|---|---|
| `logit_poly_price` | Logit transform of poly_price |
| `price_x_score_diff` | Interaction: poly_price * score_diff |
| `time_x_score_diff` | Interaction: time_remaining * score_diff |

## File reference

| File | Purpose |
|---|---|b
| `fetch_games.py` | Downloads NCAA PBP + Polymarket data to `data/games/` |
| `predict_live_game.py` | One-shot live inference: NCAA + Poly -> `/predict` |
| `in_game_features.py` | Shared feature math for training/live parity |
| `load_features.py` | Builds `features_basketball.parquet` from game JSONs |
| `train.py` | Trains LightGBM model, saves artifacts |
| `serve.py` | FastAPI prediction server |
| `eval_test.py` | Evaluates model on held-out test games |
| `calibration_utils.py` | Calibration helpers (Platt, temperature, isotonic) |
| `artifacts/model.txt` | Trained LightGBM model |
| `artifacts/model_meta.json` | Feature names, hyperparams, test metrics |
