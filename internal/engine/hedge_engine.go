package engine

import (
	"fmt"
	"math"
)

// HedgeConfig controls the Gemini hedge layer (green curve).
type HedgeConfig struct {
	Mode                 string  `yaml:"mode"`                   // "opposite" | "same_side" | "auto"
	CurveAlpha           float64 `yaml:"curve_alpha"`            // 2.0 — Beta dist param (peaks later)
	CurveBeta            float64 `yaml:"curve_beta"`             // 3.0
	MaxHedgeQty          float64 `yaml:"max_hedge_qty"`          // 15 — green curve max contracts
	LossThresholdPct     float64 `yaml:"loss_threshold_pct"`     // 0.15 — switch to opposite side
	MildLossThresholdPct float64 `yaml:"mild_loss_threshold_pct"` // 0.05 — switch to cost-avg
	MaxTotalExposure     float64 `yaml:"max_total_exposure"`     // 50.0 dollar cap across both
	GeminiHalfSpread     float64 `yaml:"gemini_half_spread"`     // 0.01
	RebalanceThreshold   float64 `yaml:"rebalance_threshold"`    // 1.0 contracts
	PollIntervalMS       int     `yaml:"poll_interval_ms"`       // 5000
}

// DefaultHedgeConfig returns production defaults matching the plan.
func DefaultHedgeConfig() HedgeConfig {
	return HedgeConfig{
		Mode:                 "auto",
		CurveAlpha:           2.0,
		CurveBeta:            3.0,
		MaxHedgeQty:          15.0,
		LossThresholdPct:     0.15,
		MildLossThresholdPct: 0.05,
		MaxTotalExposure:     50.0,
		GeminiHalfSpread:     0.01,
		RebalanceThreshold:   1.0,
		PollIntervalMS:       5000,
	}
}

// HedgeState is the JSON payload written by Python each tick.
type HedgeState struct {
	PolyPosition string  `json:"position"`    // "HOME" | "AWAY" | "FLAT"
	PolyQty      float64 `json:"qty"`         // current Poly contract count
	PolyEntry    float64 `json:"entry"`       // Poly entry price
	PolyMid      float64 `json:"mid"`         // current Poly mid price
	PnL          float64 `json:"pnl"`         // Poly unrealised PnL (dollars)
	GameID       string  `json:"game_id"`     // NCAA game ID
	TNorm        float64 `json:"t_norm"`      // normalised game time [0,1]
	NewsActive   bool    `json:"news_active"` // has any news event been triggered?
	Timestamp    string  `json:"timestamp"`   // ISO 8601
}

// HedgeDirective is the output of Evaluate — tells the monitor what to do.
type HedgeDirective struct {
	Action    string  // "buy_opposite" | "buy_same" | "reduce" | "hold" | "close"
	Side      string  // Gemini contract side ("HOME" | "AWAY")
	DeltaQty  float64 // contracts to add/remove
	TargetQty float64 // desired total Gemini position
	Reason    string
}

// HedgeEngine computes the green curve and picks hedge direction.
type HedgeEngine struct {
	Cfg      HedgeConfig
	betaNorm float64 // normaliser so peak of GeminiCurve = 1.0
}

// NewHedgeEngine creates an engine with pre-computed Beta normaliser.
func NewHedgeEngine(cfg HedgeConfig) *HedgeEngine {
	e := &HedgeEngine{Cfg: cfg}
	e.betaNorm = e.betaPDFRaw(e.betaMode())
	return e
}

func (e *HedgeEngine) betaMode() float64 {
	a, b := e.Cfg.CurveAlpha, e.Cfg.CurveBeta
	if a <= 1 && b <= 1 {
		return 0.5
	}
	return (a - 1.0) / (a + b - 2.0)
}

func logBeta(a, b float64) float64 {
	lg1, _ := math.Lgamma(a)
	lg2, _ := math.Lgamma(b)
	lg3, _ := math.Lgamma(a + b)
	return lg1 + lg2 - lg3
}

func (e *HedgeEngine) betaPDFRaw(t float64) float64 {
	a, b := e.Cfg.CurveAlpha, e.Cfg.CurveBeta
	if t <= 0.0 || t >= 1.0 {
		return 0.0
	}
	logVal := (a-1.0)*math.Log(t) + (b-1.0)*math.Log(1.0-t) - logBeta(a, b)
	return math.Exp(logVal)
}

// GeminiCurve returns [0,1] from Beta(alpha, beta) normalised to peak=1.0.
func (e *HedgeEngine) GeminiCurve(tNorm float64) float64 {
	t := math.Max(0.0, math.Min(1.0, tNorm))
	if e.betaNorm <= 0 {
		return 0.0
	}
	return e.betaPDFRaw(t) / e.betaNorm
}

// TargetPosition returns desired Gemini contract count.
// Returns 0 if news hasn't been triggered yet (delayed start).
func (e *HedgeEngine) TargetPosition(tNorm float64, newsActive bool) float64 {
	if !newsActive {
		return 0.0
	}
	return e.GeminiCurve(tNorm) * e.Cfg.MaxHedgeQty
}

// PickDirection decides the Gemini side based on Poly P&L percentage.
// Returns ("opposite"|"same"|"amplify", reason).
func (e *HedgeEngine) PickDirection(polyPnLPct float64) (string, string) {
	mode := e.Cfg.Mode

	if mode == "opposite" {
		return "opposite", "forced_opposite"
	}
	if mode == "same_side" {
		return "same", "forced_same"
	}

	// Auto mode — severity of Poly P&L determines action
	if polyPnLPct < -e.Cfg.LossThresholdPct {
		return "opposite", fmt.Sprintf("losing_%.1f%%_>_%.0f%%_threshold", polyPnLPct*100, e.Cfg.LossThresholdPct*100)
	}
	if polyPnLPct < -e.Cfg.MildLossThresholdPct {
		return "same", fmt.Sprintf("mild_loss_%.1f%%_cost_avg", polyPnLPct*100)
	}
	return "amplify", fmt.Sprintf("winning_or_flat_%.1f%%", polyPnLPct*100)
}

// oppositeSide returns the other side.
func oppositeSide(side string) string {
	if side == "HOME" {
		return "AWAY"
	}
	return "HOME"
}

// Evaluate produces a HedgeDirective given current state and Gemini book.
func (e *HedgeEngine) Evaluate(state HedgeState, geminiPosition string, geminiQty float64, geminiBestBid, geminiBestAsk float64) HedgeDirective {
	// Near market end — close everything
	if state.TNorm >= 0.95 {
		if geminiQty > 0 {
			return HedgeDirective{
				Action:    "close",
				Side:      geminiPosition,
				DeltaQty:  -geminiQty,
				TargetQty: 0,
				Reason:    "market_end_taper",
			}
		}
		return HedgeDirective{Action: "hold", Reason: "market_end_flat"}
	}

	target := e.TargetPosition(state.TNorm, state.NewsActive)

	// No news yet — hold
	if target <= 0 {
		if geminiQty > 0 {
			return HedgeDirective{
				Action:    "close",
				Side:      geminiPosition,
				DeltaQty:  -geminiQty,
				TargetQty: 0,
				Reason:    "no_news_yet",
			}
		}
		return HedgeDirective{Action: "hold", Reason: "awaiting_news"}
	}

	// Compute Poly P&L percentage
	polyPnLPct := 0.0
	if state.PolyQty > 0 && state.PolyEntry > 0 {
		invested := state.PolyQty * state.PolyEntry
		if invested > 0 {
			polyPnLPct = state.PnL / invested
		}
	}

	direction, reason := e.PickDirection(polyPnLPct)

	// Determine desired Gemini side
	var desiredSide string
	switch direction {
	case "opposite":
		if state.PolyPosition == "FLAT" || state.PolyPosition == "" {
			return HedgeDirective{Action: "hold", Reason: "poly_flat_no_hedge"}
		}
		desiredSide = oppositeSide(state.PolyPosition)
	case "same", "amplify":
		if state.PolyPosition == "FLAT" || state.PolyPosition == "" {
			return HedgeDirective{Action: "hold", Reason: "poly_flat_no_amplify"}
		}
		desiredSide = state.PolyPosition
	}

	// Exposure cap check
	polyExposure := state.PolyQty * state.PolyMid
	maxGeminiExposure := e.Cfg.MaxTotalExposure - polyExposure
	if maxGeminiExposure < 0 {
		maxGeminiExposure = 0
	}
	geminiMid := (geminiBestBid + geminiBestAsk) / 2.0
	if geminiMid > 0 {
		maxGeminiQty := maxGeminiExposure / geminiMid
		if target > maxGeminiQty {
			target = maxGeminiQty
		}
	}

	// Side flip — close first
	if geminiPosition != "" && geminiPosition != desiredSide && geminiQty > 0 {
		return HedgeDirective{
			Action:    "close",
			Side:      geminiPosition,
			DeltaQty:  -geminiQty,
			TargetQty: 0,
			Reason:    fmt.Sprintf("side_flip_%s_to_%s_%s", geminiPosition, desiredSide, reason),
		}
	}

	delta := target - geminiQty
	if math.Abs(delta) < e.Cfg.RebalanceThreshold {
		return HedgeDirective{Action: "hold", TargetQty: geminiQty, Reason: "within_threshold"}
	}

	if delta > 0 {
		action := "buy_same"
		if direction == "opposite" {
			action = "buy_opposite"
		}
		return HedgeDirective{
			Action:    action,
			Side:      desiredSide,
			DeltaQty:  delta,
			TargetQty: target,
			Reason:    reason,
		}
	}

	return HedgeDirective{
		Action:    "reduce",
		Side:      desiredSide,
		DeltaQty:  delta,
		TargetQty: target,
		Reason:    "curve_trim_" + reason,
	}
}

// CheckRebalance produces trim-only directives when the curve declines.
func (e *HedgeEngine) CheckRebalance(state HedgeState, geminiPosition string, geminiQty float64) HedgeDirective {
	if geminiQty <= 0 {
		return HedgeDirective{Action: "hold", Reason: "flat"}
	}

	target := e.TargetPosition(state.TNorm, state.NewsActive)
	delta := target - geminiQty

	// Only trim down
	if delta >= -e.Cfg.RebalanceThreshold {
		return HedgeDirective{Action: "hold", TargetQty: geminiQty, Reason: "no_trim_needed"}
	}

	return HedgeDirective{
		Action:    "reduce",
		Side:      geminiPosition,
		DeltaQty:  delta,
		TargetQty: target,
		Reason:    "curve_trim_rebalance",
	}
}
