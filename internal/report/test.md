  How to run:
  # Default market (Timberwolves vs. Celtics right now)
  go test ./internal/report/... -v -run TestMarketReport

  # Custom market
  MARKET_NAME="St. John's Red Storm vs. Kansas Jayhawks" \
    go test ./internal/report/... -v -run TestMarketReport

  # With NewsAPI key for full sentiment
  MARKET_NAME="UCLA Bruins vs. Connecticut Huskies" \
    NEWSAPI_KEY=your_key \
    go test ./internal/report/... -v -run TestMarketReport

cp config/config.example.yaml config/config.yaml
     1) Export historical resolved-market snapshots (API calls happen here)
go run ./cmd/export -max 100 -lookback 365
# 2) Build parquet features from exported JSONL
python python/ml/load_features.py
# 3) Train model
python python/ml/train.py