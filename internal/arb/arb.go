package arb

import (
	"log"
	"sync"

	"github.com/hrishabhayush/polyxgemini/internal/fees"
)

// PriceUpdate is sent by WS clients when a new ask price is observed.
type PriceUpdate struct {
	PairID   string  // matches PairConfig.Name
	Exchange string  // "poly" or "gemini"
	Outcome  string  // "yes" or "no"
	AskPrice float64 // best ask price (0 to 1)
}

// pairState tracks latest ask prices for both outcomes on both exchanges.
type pairState struct {
	PolyYesAsk  float64
	PolyNoAsk   float64
	GeminiYesAsk float64
	GeminiNoAsk  float64
}

// Detector listens for price updates and checks for arb opportunities.
type Detector struct {
	updates chan PriceUpdate
	state   map[string]*pairState // keyed by PairID
	mu      sync.Mutex
}

func NewDetector(bufSize int) *Detector {
	return &Detector{
		updates: make(chan PriceUpdate, bufSize),
		state:   make(map[string]*pairState),
	}
}

// Updates returns the channel to send price updates to.
func (d *Detector) Updates() chan<- PriceUpdate {
	return d.updates
}

// Run starts the arb detection loop. Blocks until the channel is closed.
func (d *Detector) Run() {
	for u := range d.updates {
		d.mu.Lock()
		ps, ok := d.state[u.PairID]
		if !ok {
			ps = &pairState{}
			d.state[u.PairID] = ps
		}

		switch {
		case u.Exchange == "poly" && u.Outcome == "yes":
			ps.PolyYesAsk = u.AskPrice
		case u.Exchange == "poly" && u.Outcome == "no":
			ps.PolyNoAsk = u.AskPrice
		case u.Exchange == "gemini" && u.Outcome == "yes":
			ps.GeminiYesAsk = u.AskPrice
		case u.Exchange == "gemini" && u.Outcome == "no":
			ps.GeminiNoAsk = u.AskPrice
		}

		// Check both directions:
		// 1. Buy YES on Poly + Buy NO on Gemini
		// 2. Buy NO on Poly + Buy YES on Gemini
		d.checkArb(u.PairID, ps)
		d.mu.Unlock()
	}
}

func (d *Detector) checkArb(pairID string, ps *pairState) {
	// Direction 1: Poly YES + Gemini NO
	if ps.PolyYesAsk > 0 && ps.GeminiNoAsk > 0 {
		polyFee := fees.PolymarketSportsTakerFee(ps.PolyYesAsk)
		geminiFee := fees.GeminiTakerFee(ps.GeminiNoAsk)
		polyCost := fees.EffectiveBuyCost(ps.PolyYesAsk, polyFee)
		geminiCost := fees.EffectiveBuyCost(ps.GeminiNoAsk, geminiFee)
		totalCost := polyCost + geminiCost
		if totalCost < 1.0 {
			profit := 1.0 - totalCost
			log.Printf("[ARB] %s | BUY YES on Poly @ %.3f (fee %.4f) + BUY NO on Gemini @ %.3f (fee %.4f) = %.4f | profit: %.4f (%.2f%%)",
				pairID, ps.PolyYesAsk, polyFee, ps.GeminiNoAsk, geminiFee, totalCost, profit, profit*100)
		} else {
			log.Printf("[SCAN] %s | Poly YES %.3f + Gemini NO %.3f = %.4f (no arb)",
				pairID, polyCost, geminiCost, totalCost)
		}
	}

	// Direction 2: Poly NO + Gemini YES
	if ps.PolyNoAsk > 0 && ps.GeminiYesAsk > 0 {
		polyFee := fees.PolymarketSportsTakerFee(ps.PolyNoAsk)
		geminiFee := fees.GeminiTakerFee(ps.GeminiYesAsk)
		polyCost := fees.EffectiveBuyCost(ps.PolyNoAsk, polyFee)
		geminiCost := fees.EffectiveBuyCost(ps.GeminiYesAsk, geminiFee)
		totalCost := polyCost + geminiCost
		if totalCost < 1.0 {
			profit := 1.0 - totalCost
			log.Printf("[ARB] %s | BUY NO on Poly @ %.3f (fee %.4f) + BUY YES on Gemini @ %.3f (fee %.4f) = %.4f | profit: %.4f (%.2f%%)",
				pairID, ps.PolyNoAsk, polyFee, ps.GeminiYesAsk, geminiFee, totalCost, profit, profit*100)
		} else {
			log.Printf("[SCAN] %s | Poly NO %.3f + Gemini YES %.3f = %.4f (no arb)",
				pairID, polyCost, geminiCost, totalCost)
		}
	}
}
