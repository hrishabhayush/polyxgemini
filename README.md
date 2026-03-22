# polyxgemini

Prediction market arbitrage bot between Polymarket and Gemini.

## Quick Start

```bash
# Copy example configs
cp config/config.example.yaml config/config.yaml
cp config/markets.example.yaml config/markets.yaml

# Edit markets.yaml with Polymarket slugs you want to watch
# (slug = the URL path on polymarket.com)

# Build and run
make build
make run
```

## Config Files

- `config/config.yaml` — Exchange endpoints, API keys, engine settings
- `config/markets.yaml` — List of Polymarket market slugs to subscribe to
