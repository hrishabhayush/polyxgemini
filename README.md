# polyxgemini

NCAA basketball live prediction + trading engine. Combines real-time play-by-play data with Polymarket prices to find edges, then sizes positions via a risk engine with Beta-curve position sizing.

## Setup

```bash
# Python 3.12+
python3 -m venv .venv
source .venv/bin/activate
pip install -r python/ml/requirements.txt
```

## Run (Demo Mode)

Demo mode replays historical Polymarket data from a completed game against any NCAA game's play-by-play — no live markets needed.

```bash
# Terminal 1: start ML server
cd python/ml
uvicorn serve:app --host 127.0.0.1 --port 8766

# Terminal 2: run the bot (from repo root)
python python/ml/predict_live_game.py \
  --game-id 6534713 \
  --live \
  --demo data/games/6534602.json \
  --trade-engine \
  --risk-engine \
  --interval-sec 2
```

This steps through every play from tipoff to final buzzer, with Polymarket prices tracking the game clock.

## Run (Live)

For a game currently in progress with a live Polymarket market:

```bash
python python/ml/predict_live_game.py \
  --game-id <NCAA_GAME_ID> \
  --live \
  --trade-engine \
  --risk-engine
```

## Project Structure

- `python/ml/predict_live_game.py` — main live trading loop
- `python/ml/serve.py` — ML prediction server (LightGBM)
- `python/ml/trading_engine.py` — 5-state trading machine (when to trade)
- `python/ml/risk_engine.py` — Beta-curve position sizing (how much to trade)
- `python/ml/paper_portfolio.py` — paper P&L tracking
- `data/games/` — historical game JSONs with Polymarket price/trade data
- `internal/` — Go backend for exchange integrations (Polymarket CLOB, Gemini)
- `config/` — API keys (`config.yaml`), market pairs (`markets.yaml`)
