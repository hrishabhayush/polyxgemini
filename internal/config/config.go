package config

import (
	"os"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Polymarket PolymarketConfig `yaml:"polymarket"`
	Gemini     GeminiConfig     `yaml:"gemini"`
	Engine     EngineConfig     `yaml:"engine"`
	Sentiment  SentimentConfig  `yaml:"sentiment"`
}

type PolymarketConfig struct {
	CLOBBaseURL   string `yaml:"clob_base_url"`
	GammaBaseURL  string `yaml:"gamma_base_url"`
	WSURL         string `yaml:"ws_url"`
	WatchlistPath string `yaml:"watchlist_path"`
	APIKey        string `yaml:"api_key"`
	APISecret     string `yaml:"api_secret"`
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

type SentimentConfig struct {
	NewsAPIKey          string   `yaml:"newsapi_key"`
	NewsAPIBaseURL      string   `yaml:"newsapi_base_url"`
	GDELTBaseURL        string   `yaml:"gdelt_base_url"`
	RedditBaseURL       string   `yaml:"reddit_base_url"`
	RedditUserAgent     string   `yaml:"reddit_user_agent"`
	RedditClientID      string   `yaml:"reddit_client_id"`
	RedditClientSecret  string   `yaml:"reddit_client_secret"`
	RedditSubreddits    []string `yaml:"reddit_subreddits"`
	FinBERTServerURL    string   `yaml:"finbert_server_url"`
	PollIntervalSec     int      `yaml:"poll_interval_sec"`
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
	Slug string `yaml:"slug"`
}

type Watchlist struct {
	Markets []MarketEntry `yaml:"markets"`
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
