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