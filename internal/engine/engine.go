package engine

import (
	"github.com/hrishabhayush/polyxgemini/internal/config"
	"github.com/hrishabhayush/polyxgemini/internal/exchange"
	"github.com/hrishabhayush/polyxgemini/internal/matcher"
)

type Opportunity struct {
	Pair       matcher.MarketPair
	BuyOn      string  // "polymarket" or "gemini"
	SellOn     string
	SpreadBPS  int
	BuyPrice   float64
	SellPrice  float64
}

type Engine struct {
	poly     exchange.Exchange
	gemini   exchange.Exchange
	matcher  *matcher.Matcher
	cfg      config.EngineConfig
}

func NewEngine(poly, gemini exchange.Exchange, m *matcher.Matcher, cfg config.EngineConfig) *Engine {
	return &Engine{poly: poly, gemini: gemini, matcher: m, cfg: cfg}
}

func (e *Engine) Run() error {
	// TODO: Poll loop — fetch books, find opportunities, send to executor
	return nil
}
