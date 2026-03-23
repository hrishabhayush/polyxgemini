  Run all 4 tests at once:                                                                                                                 
  go test ./internal/exchange/polymarket/polymarket_test/... -v -timeout 60s | cat                                                         

  Run individual tests:                                                                                                                    
  go test ./internal/exchange/polymarket/polymarket_test/... -v -run TestGammaFetchMarkets -timeout 30s | cat
  go test ./internal/exchange/polymarket/polymarket_test/... -v -run TestDataTrades     -timeout 30s | cat                                 
  go test ./internal/exchange/polymarket/polymarket_test/... -v -run TestDataPositions  -timeout 30s | cat                                 
  go test ./internal/exchange/polymarket/polymarket_test/... -v -run TestDataPrices     -timeout 30s | cat                                 
                                                                                                                                           
  ▎ The | cat at the end is needed on macOS — without it, go test swallows fmt.Printf output when tests pass.                              
                                                                                                                                           
  Three env vars to override for any test:
  POLYMARKET_CONDITION_ID=0x...   # trades, positions
  POLYMARKET_TOKEN_ID=123...      # prices (YES token)
  POLYMARKET_SLUG=some-slug       # gamma single-market lookup