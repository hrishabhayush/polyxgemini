package engine

import "fmt"

// GeminiPaperPortfolio tracks simulated Gemini hedge positions.
type GeminiPaperPortfolio struct {
	Position   string  // "HOME" | "AWAY" | ""
	Qty        float64 // absolute contract count
	Entry      float64 // VWAP entry price
	Realised   float64 // cumulative realised PnL
	Unrealised float64 // mark-to-market unrealised PnL
}

// OnFill handles a new paper fill. VWAP for same-side adds, close-then-open for flips.
func (p *GeminiPaperPortfolio) OnFill(side string, qty float64, price float64) {
	if qty <= 0 {
		return
	}

	// Same side — VWAP add
	if p.Position == side && p.Qty > 0 {
		newQty := p.Qty + qty
		p.Entry = (p.Qty*p.Entry + qty*price) / newQty
		p.Qty = newQty
		return
	}

	// Different side or flat — close existing first
	if p.Position != "" && p.Qty > 0 {
		// Close at the fill price (simplified — real would use bid)
		leg := p.Qty * (price - p.Entry)
		// If closing opposite side, the PnL logic flips
		// For simplicity: realise at given price
		p.Realised += leg
		p.Qty = 0
		p.Entry = 0
		p.Position = ""
	}

	// Open new side
	p.Position = side
	p.Qty = qty
	p.Entry = price
}

// MarkToMarket updates unrealised PnL from current book.
func (p *GeminiPaperPortfolio) MarkToMarket(bestBid, bestAsk float64) {
	if p.Position == "" || p.Qty <= 0 {
		p.Unrealised = 0
		return
	}
	mid := (bestBid + bestAsk) / 2.0
	p.Unrealised = p.Qty * (mid - p.Entry)
}

// ReducePosition partially closes at bid price.
func (p *GeminiPaperPortfolio) ReducePosition(qty float64, bidPrice float64) {
	if p.Position == "" || p.Qty <= 0 || qty <= 0 {
		return
	}
	closeQty := qty
	if closeQty > p.Qty {
		closeQty = p.Qty
	}
	leg := closeQty * (bidPrice - p.Entry)
	p.Realised += leg
	p.Qty -= closeQty
	if p.Qty < 1e-9 {
		p.Qty = 0
		p.Position = ""
		p.Entry = 0
	}
}

// CombinedPnL returns total PnL across both Poly and Gemini positions.
func (p *GeminiPaperPortfolio) CombinedPnL(polyRealised, polyUnrealised float64) float64 {
	return (polyRealised + polyUnrealised) + (p.Realised + p.Unrealised)
}

// Summary returns a log-friendly one-liner.
func (p *GeminiPaperPortfolio) Summary() string {
	pos := "FLAT"
	if p.Position != "" {
		pos = p.Position
	}
	return fmt.Sprintf("[GEMINI] %s qty=%.2f entry=%.4f real=%.4f unreal=%.4f",
		pos, p.Qty, p.Entry, p.Realised, p.Unrealised)
}
