package config

import (
	"os"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Polymarket PolymarketConfig `yaml:"polymarket"`
	Gemini     GeminiConfig     `yaml:"gemini"`
	Engine     EngineConfig     `yaml:"engine"`
	Hedge      HedgeConfig      `yaml:"hedge"`
}

type PolymarketConfig struct {
	CLOBBaseURL   string `yaml:"clob_base_url"`
	GammaBaseURL  string `yaml:"gamma_base_url"`
	WSURL         string `yaml:"ws_url"`
	WatchlistPath string `yaml:"watchlist_path"`
	APIKey        string `yaml:"api_key"`
	APISecret     string `yaml:"api_secret"`
	Passphrase    string `yaml:"passphrase"`
}

type GeminiConfig struct {
	BaseURL    string `yaml:"base_url"`
	WSURL      string `yaml:"ws_url"`
	SandboxURL string `yaml:"sandbox_url"`
	APIKey     string `yaml:"api_key"`
	APISecret  string `yaml:"api_secret"`
	UseSandbox bool   `yaml:"use_sandbox"`
}

type EngineConfig struct {
	PollIntervalMS int     `yaml:"poll_interval_ms"`
	MinSpreadBPS   int     `yaml:"min_spread_bps"`
	MaxPositionUSD float64 `yaml:"max_position_usd"`
	DryRun         bool    `yaml:"dry_run"`
}

type HedgeConfig struct {
	Mode                 string  `yaml:"mode"`
	CurveAlpha           float64 `yaml:"curve_alpha"`
	CurveBeta            float64 `yaml:"curve_beta"`
	MaxHedgeQty          float64 `yaml:"max_hedge_qty"`
	LossThresholdPct     float64 `yaml:"loss_threshold_pct"`
	MildLossThresholdPct float64 `yaml:"mild_loss_threshold_pct"`
	MaxTotalExposure     float64 `yaml:"max_total_exposure"`
	GeminiHalfSpread     float64 `yaml:"gemini_half_spread"`
	RebalanceThreshold   float64 `yaml:"rebalance_threshold"`
	PollIntervalMS       int     `yaml:"poll_interval_ms"`
	GameDurationMin      int     `yaml:"game_duration_min"` // fallback game duration (default 40)
}

// MarketPair is a resolved pair from markets.yaml for hedge monitor use.
type MarketPair struct {
	Name           string
	PolymarketSlug string
	GeminiTicker   string
	GeminiSymbols  []string // resolved instrument symbols
}

// LoadMarketPairs loads pairs from markets.yaml and returns them.
func LoadMarketPairs(path string) ([]PairEntry, error) {
	wl, err := LoadWatchlist(path)
	if err != nil {
		return nil, err
	}
	return wl.Pairs, nil
}

func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}
	return &cfg, nil
}

type MarketEntry struct {
	Slug     string `yaml:"slug"`
	Category string `yaml:"category"`
}

type GeminiMarketEntry struct {
	Ticker   string `yaml:"ticker"`
	Category string `yaml:"category"`
}

// OutcomeMapping maps a Polymarket outcome to a Gemini contract label.
type OutcomeMapping struct {
	PolyOutcome string `yaml:"poly_outcome"` // "yes" or "no"
	GeminiLabel string `yaml:"gemini_label"` // e.g. "UCLA", "UConn"
}

// PairEntry defines a matched market across both exchanges.
type PairEntry struct {
	Name           string           `yaml:"name"`
	Category       string           `yaml:"category"` // "sports" (default) or "crypto"
	PolymarketSlug string           `yaml:"polymarket_slug"`
	GeminiTicker   string           `yaml:"gemini_ticker"`
	Mapping        []OutcomeMapping `yaml:"mapping"`
}

type Watchlist struct {
	Markets       []MarketEntry       `yaml:"markets"`
	GeminiMarkets []GeminiMarketEntry `yaml:"gemini_markets"`
	Pairs         []PairEntry         `yaml:"pairs"`
}

func LoadWatchlist(path string) (*Watchlist, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var wl Watchlist
	if err := yaml.Unmarshal(data, &wl); err != nil {
		return nil, err
	}
	return &wl, nil
}
