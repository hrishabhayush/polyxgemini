package arb

import (
	"log"
	"math"
	"sync"

	"github.com/hrishabhayush/polyxgemini/internal/budget"
	"github.com/hrishabhayush/polyxgemini/internal/fees"
	"github.com/hrishabhayush/polyxgemini/internal/metrics"
)

// PriceUpdate is sent by WS clients when a new ask price is observed.
type PriceUpdate struct {
	PairID   string  // matches PairConfig.Name
	Exchange string  // "poly" or "gemini"
	Outcome  string  // "yes" or "no"
	AskPrice float64 // best ask price (0 to 1)
	AskQty   float64 // quantity available at best ask
	Category string  // "sports" (default) or "crypto"
}

// pairState tracks latest ask prices and quantities for both outcomes on both exchanges.
type pairState struct {
	Category        string
	PolyYesAsk      float64
	PolyYesAskQty   float64
	PolyNoAsk       float64
	PolyNoAskQty    float64
	GeminiYesAsk    float64
	GeminiYesAskQty float64
	GeminiNoAsk     float64
	GeminiNoAskQty  float64
}

// Detector listens for price updates and checks for arb opportunities.
type Detector struct {
	updates chan PriceUpdate
	state   map[string]*pairState // keyed by PairID
	mu      sync.Mutex
	budget  *budget.Budget
}

func NewDetector(bufSize int, b *budget.Budget) *Detector {
	return &Detector{
		updates: make(chan PriceUpdate, bufSize),
		state:   make(map[string]*pairState),
		budget:  b,
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
		if u.Category != "" {
			ps.Category = u.Category
		}

		switch {
		case u.Exchange == "poly" && u.Outcome == "yes":
			ps.PolyYesAsk = u.AskPrice
			ps.PolyYesAskQty = u.AskQty
		case u.Exchange == "poly" && u.Outcome == "no":
			ps.PolyNoAsk = u.AskPrice
			ps.PolyNoAskQty = u.AskQty
		case u.Exchange == "gemini" && u.Outcome == "yes":
			ps.GeminiYesAsk = u.AskPrice
			ps.GeminiYesAskQty = u.AskQty
		case u.Exchange == "gemini" && u.Outcome == "no":
			ps.GeminiNoAsk = u.AskPrice
			ps.GeminiNoAskQty = u.AskQty
		}

		d.checkArb(u.PairID, ps)
		d.mu.Unlock()
	}
}

// polyFeeFunc returns the correct Polymarket fee function for the given category.
func polyFeeFunc(category string) func(float64) float64 {
	if category == "crypto" {
		return fees.PolymarketCryptoTakerFee
	}
	return fees.PolymarketSportsTakerFee
}

func (d *Detector) checkArb(pairID string, ps *pairState) {
	polyFeeFn := polyFeeFunc(ps.Category)

	// Direction 1: Buy YES on Poly + Buy NO on Gemini
	if ps.PolyYesAsk > 0 && ps.GeminiNoAsk > 0 {
		polyFee := polyFeeFn(ps.PolyYesAsk)
		geminiFee := fees.GeminiTakerFee(ps.GeminiNoAsk)
		polyCost := fees.EffectiveBuyCost(ps.PolyYesAsk, polyFee)
		geminiCost := fees.EffectiveBuyCost(ps.GeminiNoAsk, geminiFee)
		costPerPair := polyCost + geminiCost
		spreadBPS := (1.0 - costPerPair) * 10000
		metrics.ArbSpreadBPS.WithLabelValues(pairID, "poly_yes_gemini_no").Set(spreadBPS)
		if costPerPair < 1.0 {
			metrics.ArbOpportunitiesDetected.WithLabelValues(pairID).Inc()
			d.executeFAK(pairID, "YES", "NO", ps.PolyYesAsk, polyFee, ps.PolyYesAskQty, ps.GeminiNoAsk, geminiFee, ps.GeminiNoAskQty, costPerPair)
		} else {
			log.Printf("[SCAN] %s | Poly YES %.3f + Gemini NO %.3f = %.4f (no arb)",
				pairID, polyCost, geminiCost, costPerPair)
		}
	}

	// Direction 2: Buy NO on Poly + Buy YES on Gemini
	if ps.PolyNoAsk > 0 && ps.GeminiYesAsk > 0 {
		polyFee := polyFeeFn(ps.PolyNoAsk)
		geminiFee := fees.GeminiTakerFee(ps.GeminiYesAsk)
		polyCost := fees.EffectiveBuyCost(ps.PolyNoAsk, polyFee)
		geminiCost := fees.EffectiveBuyCost(ps.GeminiYesAsk, geminiFee)
		costPerPair := polyCost + geminiCost
		spreadBPS := (1.0 - costPerPair) * 10000
		metrics.ArbSpreadBPS.WithLabelValues(pairID, "poly_no_gemini_yes").Set(spreadBPS)
		if costPerPair < 1.0 {
			metrics.ArbOpportunitiesDetected.WithLabelValues(pairID).Inc()
			d.executeFAK(pairID, "NO", "YES", ps.PolyNoAsk, polyFee, ps.PolyNoAskQty, ps.GeminiYesAsk, geminiFee, ps.GeminiYesAskQty, costPerPair)
		} else {
			log.Printf("[SCAN] %s | Poly NO %.3f + Gemini YES %.3f = %.4f (no arb)",
				pairID, polyCost, geminiCost, costPerPair)
		}
	}
}

// executeFAK simulates a Fill-and-Kill order: buy as many contracts as available
// at the best ask on both sides, limited by the remaining budget.
func (d *Detector) executeFAK(pairID, polySide, geminiSide string, polyAsk, polyFee, polyQty, geminiAsk, geminiFee, geminiQty, costPerPair float64) {
	if d.budget == nil || d.budget.Exhausted() {
		return
	}

	profitPerPair := 1.0 - costPerPair

	// FAK: max contracts = min(qty available on each side, budget allows)
	maxQty := math.Min(polyQty, geminiQty)
	if maxQty <= 0 {
		log.Printf("[ARB] %s | BUY %s on Poly @ %.3f + BUY %s on Gemini @ %.3f = %.4f | profit/contract: %.4f (%.2f%%) | NO QTY available",
			pairID, polySide, polyAsk, geminiSide, geminiAsk, costPerPair, profitPerPair, profitPerPair*100)
		return
	}

	budgetQty := math.Floor(d.budget.Remaining() / costPerPair)
	qty := math.Min(maxQty, budgetQty)
	if qty <= 0 {
		log.Printf("[ARB] %s | BUY %s on Poly @ %.3f + BUY %s on Gemini @ %.3f = %.4f | profit/contract: %.4f (%.2f%%) | budget too low (remaining: $%.2f)",
			pairID, polySide, polyAsk, geminiSide, geminiAsk, costPerPair, profitPerPair, profitPerPair*100, d.budget.Remaining())
		return
	}

	totalCost := qty * costPerPair
	totalProfit := qty * profitPerPair

	spent, ok := d.budget.TrySpend(totalCost)
	if !ok {
		return
	}

	// Emit execution metrics
	metrics.ArbOpportunitiesExecuted.WithLabelValues(pairID).Inc()
	metrics.CapitalDeployedUSD.WithLabelValues("polymarket").Add(qty * (polyAsk + polyFee))
	metrics.CapitalDeployedUSD.WithLabelValues("gemini").Add(qty * (geminiAsk + geminiFee))
	metrics.CapitalTotalUSD.Add(spent)
	metrics.FeesTotalUSD.WithLabelValues("polymarket").Add(qty * polyFee)
	metrics.FeesTotalUSD.WithLabelValues("gemini").Add(qty * geminiFee)
	metrics.PnLRealizedUSD.WithLabelValues(pairID).Add(totalProfit)
	metrics.ActivePositions.WithLabelValues("polymarket", pairID, polySide).Add(qty)
	metrics.ActivePositions.WithLabelValues("gemini", pairID, geminiSide).Add(qty)

	log.Printf("[TRADE-DRY] %s | BUY %.0f %s on Poly @ %.3f (fee %.4f) + BUY %.0f %s on Gemini @ %.3f (fee %.4f) | cost: $%.4f | profit: $%.4f (%.2f%%) | spent: $%.4f | remaining: $%.2f",
		pairID,
		qty, polySide, polyAsk, polyFee,
		qty, geminiSide, geminiAsk, geminiFee,
		spent, totalProfit, profitPerPair*100,
		d.budget.Spent(), d.budget.Remaining())
}
