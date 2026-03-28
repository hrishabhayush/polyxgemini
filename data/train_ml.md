# ML Workflow: Historical Training + Live Prediction

Run from repo root:

`/Users/caedylee/Documents/Blockchain/penn`

## 0) Environment setup (once)

```bash
source .venv-ocr/bin/activate
pip install -U pandas numpy scikit-learn lightgbm pyarrow fastapi uvicorn
```

## 1) Export historical resolved markets (training raw data)

```bash
go run ./cmd/export -max 500 -lookback 365 -sentiment-window 1m
```

Output file(s):

- `data/snapshots/resolved_YYYYMMDD.jsonl`

If you want a larger training set:

```bash
go run ./cmd/export -max 2000 -lookback 730 -sentiment-window 3m
```

## 2) Build parquet feature table

```bash
python python/ml/load_features.py
```

Output:

- `data/features.parquet`

Explicit paths:

```bash
python python/ml/load_features.py --input data/snapshots --output data/features.parquet
```

## 3) Train ML model

```bash
python python/ml/train.py
```

Artifacts created in `python/ml/artifacts/`:

- `model.txt`
- `model_meta.json`
- `calibration.json` (if calibration selected)
- `calibration.png`

## 4) Start ML prediction server

Terminal A:

```bash
cd python/ml
uvicorn serve:app --host 127.0.0.1 --port 8766
```

Terminal B:

```bash
cd /Users/caedylee/Documents/Blockchain/penn
```

## 5) Pull fresh live market snapshots

```bash
go run ./cmd/snapshot -max 100
```

Output:

- `data/snapshots/active_<timestamp>.jsonl`

## 6) Score latest live snapshot with ML server (automated)

```bash
python - <<'PY'
  import json, glob, math, datetime as dt, urllib.request

  files = sorted(glob.glob("data/snapshots/active_*.jsonl"))
  if not files:
      raise SystemExit("No active_*.jsonl files found in data/snapshots")
  path = files[-1]
  print(f"Scoring snapshot file: {path}")

  def jump_enc(s):
      s = (s or "").strip().lower()
      return 2 if s == "sustained" else 1 if s == "reversed" else 0

  def market_age_days(created_at, end_date):
      try:
          c = dt.datetime.fromisoformat(created_at.replace("Z","+00:00"))
          if end_date:
              e = dt.datetime.fromisoformat(end_date.replace("Z","+00:00"))
              return max(0.0, (e - c).total_seconds()/86400.0)
      except Exception:
          pass
      return 0.0

  rows = []
  with open(path) as f:
      for line in f:
          if not line.strip():
              continue
          m = json.loads(line)
          cp = float(m.get("current_price", 0.0) or 0.0)
          tv = float(m.get("total_volume", 0.0) or 0.0)
          tc = int(m.get("trade_count", 0) or 0)

          payload = {
              "current_price": cp,
              "hurst_exp": float(m.get("hurst_exp", 0.0) or 0.0),
              "vol_ratio": float(m.get("vol_ratio", 0.0) or 0.0),
              "jump_result_enc": jump_enc(m.get("jump_result")),
              "trade_count": tc,
              "total_volume": tv,
              "log_volume": math.log1p(tv),
              "kyles_lambda": float(m.get("kyles_lambda", 0.0) or 0.0),
              "vpin": float(m.get("vpin", 0.0) or 0.0),
              "buy_fraction": float(m.get("buy_fraction", 0.0) or 0.0),
              "wallet_hhi": float(m.get("wallet_hhi", 0.0) or 0.0),
              "market_age_days": market_age_days(m.get("created_at",""), m.get("end_date","")),
              "avg_trade_size": tv / (tc + 1),
              "log_avg_trade_size": math.log1p(tv / (tc + 1)),
              "log_trade_count": math.log1p(tc),
              "has_hurst": 1 if float(m.get("hurst_exp", 0.0) or 0.0) != 0.0 else 0,
              "price_distance_from_50": abs(cp - 0.5),
          }

          req = urllib.request.Request(
              "http://127.0.0.1:8766/predict",
              data=json.dumps(payload).encode(),
              headers={"Content-Type":"application/json"},
              method="POST",
          )
          with urllib.request.urlopen(req, timeout=10) as r:
              pred = json.loads(r.read().decode())

          edge = max(pred["edge_yes"], pred["edge_no"])
          side = "YES" if pred["edge_yes"] >= pred["edge_no"] else "NO"
          rows.append((edge, side, m.get("slug",""), cp, pred, m))

  # Quality filters
  rows = [x for x in rows if x[5].get("trade_count",0) >= 50 and x[5].get("total_volume",0) >= 5000]
  rows.sort(key=lambda x: x[0], reverse=True)

  print("\nTop 15 by edge:")
  for edge, side, slug, cp, pred, _ in rows[:15]:
      print(f"{edge:+.4f}  buy={side:3}  price={cp:.4f}  p_yes={pred['prob_yes']:.4f}  slug={slug}")
  PY
```

## 7) Repeat live cycle

1. Refresh snapshots:

```bash
go run ./cmd/snapshot -max 100
```

2. Re-run scoring script (Step 6).

## Optional: continuous scanning via bot

Instead of manual scoring, you can run:

```bash
go run ./cmd/bot -config config/config.yaml
```

This computes features and calls ML `/predict` continuously for watched markets.

