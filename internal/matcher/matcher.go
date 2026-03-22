package matcher

import "github.com/hrishabhayush/polyxgemini/internal/exchange"

// MarketPair links equivalent markets across the two exchanges.
type MarketPair struct {
	Polymarket exchange.Market
	Gemini     exchange.Market
}

type Matcher struct{}

func NewMatcher() *Matcher {
	return &Matcher{}
}

func (m *Matcher) FindPairs(polymarkets, geminiMarkets []exchange.Market) []MarketPair {
	// TODO: Match by title similarity, category, or manual mapping
	return nil
}
