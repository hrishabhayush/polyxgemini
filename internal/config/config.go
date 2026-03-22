package config

type Config struct {
	Polymarket PolymarketConfig `yaml:"polymarket"`
	Gemini     GeminiConfig     `yaml:"gemini"`
	Engine     EngineConfig     `yaml:"engine"`
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

func Load(path string) (*Config, error) {
	// TODO: Read YAML file, unmarshal into Config
	return nil, nil
}

type MarketEntry struct {
	Slug string `yaml:"slug"`
}

type Watchlist struct {
	Markets []MarketEntry `yaml:"markets"`
}

func LoadWatchlist(path string) (*Watchlist, error) {
	// TODO: Read markets YAML file, unmarshal into Watchlist
	return nil, nil
}
