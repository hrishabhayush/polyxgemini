package engine

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/hrishabhayush/polyxgemini/internal/metrics"
)

// GeminiBook holds the latest bid/ask for a contract.
type GeminiBook struct {
	BestBid float64
	BestAsk float64
}

// PolySignal is the minimal JSON written by Python each tick.
type PolySignal struct {
	Side      string  `json:"side"`       // "HOME" | "AWAY" | "FLAT"
	Qty       float64 `json:"qty"`        // contract count
	Entry     float64 `json:"entry"`      // entry price
	TNorm     float64 `json:"t_norm"`     // normalised game time [0,1]
	NewsCount int     `json:"news_count"` // number of news events detected
	GameID    string  `json:"game_id"`
}

// HedgeLogEntry is one line in data/hedge_log.jsonl.
type HedgeLogEntry struct {
	Timestamp      string  `json:"timestamp"`
	GameID         string  `json:"game_id"`
	TNorm          float64 `json:"t_norm"`
	PolyPosition   string  `json:"poly_position"`
	PolyQty        float64 `json:"poly_qty"`
	PolyEntry      float64 `json:"poly_entry"`
	PolyMid        float64 `json:"poly_mid"`
	PolyPnL        float64 `json:"poly_pnl"`
	GeminiCurve    float64 `json:"gemini_curve"`
	GeminiTarget   float64 `json:"gemini_target"`
	GeminiPosition string  `json:"gemini_position"`
	GeminiQty      float64 `json:"gemini_qty"`
	GeminiEntry    float64 `json:"gemini_entry"`
	GeminiRealised float64 `json:"gemini_realised"`
	GeminiUnreal   float64 `json:"gemini_unrealised"`
	Action         string  `json:"action"`
	DeltaQty       float64 `json:"delta_qty"`
	Reason         string  `json:"reason"`
	CombinedPnL    float64 `json:"combined_pnl"`
	GeminiBid      float64 `json:"gemini_bid"`
	GeminiAsk      float64 `json:"gemini_ask"`
	NewsActive     bool    `json:"news_active"`
}

// PairBookMapping tells the monitor which WS symbols belong to a pair.
type PairBookMapping struct {
	PairName       string
	PolyTokenIDs   []string // Poly YES/NO token IDs
	GeminiSymbols  []string // Gemini instrument symbols (lowercase)
}

// HedgeMonitor watches Poly + Gemini book prices and manages the hedge layer.
type HedgeMonitor struct {
	engine    *HedgeEngine
	portfolio *GeminiPaperPortfolio
	logPath      string
	signalPath   string
	pollInterval time.Duration

	// Pair-aware book selection
	activePair string            // pair name to filter books
	pairPoly   map[string]bool   // poly token IDs for active pair
	pairGemini map[string]bool   // gemini symbols (lowercase) for active pair

	// Poly position (from signal file)
	polyPosition string
	polyQty      float64
	polyEntry    float64

	// Game timing — from signal file t_norm, with wall-clock fallback
	tNormFromSignal float64
	signalFresh     bool // true if signal was read this tick
	gameStart       time.Time
	gameDuration    time.Duration

	// News detection from signal + Poly mid-price volatility
	lastPolyMid   float64
	newsActive    bool
	newsThreshold float64 // fractional mid shift to trigger (0.10 = 10%)

	mu        sync.Mutex
	polyBooks map[string]GeminiBook // tokenID -> latest book
	gemBooks  map[string]GeminiBook // symbol (lowercase) -> latest book
	done      chan struct{}
	logDirOK  bool
	tickCount int
}

// NewHedgeMonitor creates the monitor (not yet started).
func NewHedgeMonitor(cfg HedgeConfig, gameDuration time.Duration) *HedgeMonitor {
	pollMS := cfg.PollIntervalMS
	if pollMS <= 0 {
		pollMS = 5000
	}
	return &HedgeMonitor{
		engine:        NewHedgeEngine(cfg),
		portfolio:     &GeminiPaperPortfolio{},
		logPath:       "data/hedge_log.jsonl",
		signalPath:    "data/poly_position.json",
		pollInterval:  time.Duration(pollMS) * time.Millisecond,
		pairPoly:      make(map[string]bool),
		pairGemini:    make(map[string]bool),
		gameStart:     time.Now(),
		gameDuration:  gameDuration,
		newsThreshold: 0.10,
		polyBooks:     make(map[string]GeminiBook),
		gemBooks:      make(map[string]GeminiBook),
		done:          make(chan struct{}),
	}
}

// SetActivePair configures which pair's books to use for hedge decisions.
func (m *HedgeMonitor) SetActivePair(mapping PairBookMapping) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.activePair = mapping.PairName
	m.pairPoly = make(map[string]bool)
	for _, id := range mapping.PolyTokenIDs {
		m.pairPoly[id] = true
	}
	m.pairGemini = make(map[string]bool)
	for _, sym := range mapping.GeminiSymbols {
		m.pairGemini[strings.ToLower(sym)] = true
	}
	log.Printf("[hedge] active pair=%q poly_tokens=%d gemini_symbols=%d",
		mapping.PairName, len(mapping.PolyTokenIDs), len(mapping.GeminiSymbols))
}

// UpdatePolyBook is called by Poly WS with each orderbook update.
func (m *HedgeMonitor) UpdatePolyBook(assetID string, bestBid, bestAsk float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.polyBooks[assetID] = GeminiBook{BestBid: bestBid, BestAsk: bestAsk}
}

// UpdateGeminiBook is called by Gemini WS with each bookTicker update.
func (m *HedgeMonitor) UpdateGeminiBook(symbol string, bestBid, bestAsk float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.gemBooks[strings.ToLower(symbol)] = GeminiBook{BestBid: bestBid, BestAsk: bestAsk}
}

// Run starts the polling loop. Call in a goroutine.
func (m *HedgeMonitor) Run() {
	log.Printf("[hedge] monitor started (poll=%v, game_duration=%v)", m.pollInterval, m.gameDuration)
	ticker := time.NewTicker(m.pollInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			m.tick()
		case <-m.done:
			log.Println("[hedge] monitor stopped")
			return
		}
	}
}

// Stop signals the monitor to shut down.
func (m *HedgeMonitor) Stop() {
	close(m.done)
}

// tNorm returns the best available normalised game time.
// Prefers the Python signal's t_norm; falls back to wall-clock estimate.
func (m *HedgeMonitor) tNorm() float64 {
	if m.signalFresh && m.tNormFromSignal > 0 {
		return m.tNormFromSignal
	}
	if m.gameDuration <= 0 {
		return 0
	}
	elapsed := time.Since(m.gameStart).Seconds()
	total := m.gameDuration.Seconds()
	t := elapsed / total
	return math.Max(0, math.Min(1, t))
}

func (m *HedgeMonitor) tick() {
	m.tickCount++
	statusTick := m.tickCount%12 == 0 // log status every ~60s at 5s poll

	// Try reading Python signal file for Poly position + t_norm
	m.signalFresh = false
	var lastGameID string
	if sig, err := m.readSignal(); err == nil {
		m.mu.Lock()
		if sig.Side == "HOME" || sig.Side == "AWAY" {
			m.polyPosition = sig.Side
		} else {
			m.polyPosition = ""
		}
		m.polyQty = sig.Qty
		m.polyEntry = sig.Entry
		m.tNormFromSignal = sig.TNorm
		m.signalFresh = true
		lastGameID = sig.GameID
		if sig.NewsCount > 0 {
			m.newsActive = true
		}
		m.mu.Unlock()
	} else if statusTick {
		log.Printf("[hedge] waiting for signal file (%s)", m.signalPath)
	}

	m.mu.Lock()

	// Get Poly book for active pair
	polyBid, polyAsk := m.pairBook(m.polyBooks, m.pairPoly)

	// Get Gemini book for active pair
	gemBid, gemAsk := m.pairBook(m.gemBooks, m.pairGemini)

	polySide := m.polyPosition
	polyQty := m.polyQty
	polyEntry := m.polyEntry
	nPolyBooks := len(m.polyBooks)
	nGemBooks := len(m.gemBooks)

	m.mu.Unlock()

	// Need Gemini book at minimum
	if gemBid <= 0 && gemAsk <= 0 {
		if statusTick {
			log.Printf("[hedge] waiting for Gemini book data (poly_books=%d gemini_books=%d)", nPolyBooks, nGemBooks)
		}
		return
	}

	tNorm := m.tNorm()

	// Use Poly book mid if available, else estimate from entry
	polyMid := 0.0
	if polyBid > 0 || polyAsk > 0 {
		polyMid = (polyBid + polyAsk) / 2.0
	} else if polyEntry > 0 {
		polyMid = polyEntry
	}

	// News detection: large Poly mid-price shift
	if polyMid > 0 && m.lastPolyMid > 0 {
		shift := math.Abs(polyMid-m.lastPolyMid) / m.lastPolyMid
		if shift >= m.newsThreshold {
			m.newsActive = true
			log.Printf("[hedge] news detected: poly mid shifted %.1f%% (%.4f → %.4f)",
				shift*100, m.lastPolyMid, polyMid)
		}
	}
	if polyMid > 0 {
		m.lastPolyMid = polyMid
	}

	// Compute Poly unrealised PnL
	polyPnL := 0.0
	if polySide != "" && polyQty > 0 && polyEntry > 0 && polyMid > 0 {
		polyPnL = polyQty * (polyMid - polyEntry)
	}

	// Build hedge state
	state := HedgeState{
		PolyPosition: polySide,
		PolyQty:      polyQty,
		PolyEntry:    polyEntry,
		PolyMid:      polyMid,
		PnL:          polyPnL,
		TNorm:        tNorm,
		NewsActive:   m.newsActive,
	}

	// Evaluate hedge directive
	directive := m.engine.Evaluate(
		state,
		m.portfolio.Position, m.portfolio.Qty,
		gemBid, gemAsk,
	)

	// Also check rebalance (trim) every tick
	if directive.Action == "hold" {
		rebal := m.engine.CheckRebalance(state, m.portfolio.Position, m.portfolio.Qty)
		if rebal.Action == "reduce" {
			directive = rebal
		}
	}

	// Execute on paper portfolio
	halfSpread := m.engine.Cfg.GeminiHalfSpread
	switch directive.Action {
	case "buy_opposite", "buy_same":
		price := gemAsk + halfSpread
		m.portfolio.OnFill(directive.Side, directive.DeltaQty, price)
	case "reduce":
		reduceQty := -directive.DeltaQty
		if reduceQty > 0 {
			m.portfolio.ReducePosition(reduceQty, gemBid-halfSpread)
		}
	case "close":
		if m.portfolio.Qty > 0 {
			m.portfolio.ReducePosition(m.portfolio.Qty, gemBid-halfSpread)
		}
	}

	// Mark to market
	m.portfolio.MarkToMarket(gemBid, gemAsk)

	// Combined PnL
	combined := m.portfolio.CombinedPnL(0, polyPnL)

	curve := m.engine.GeminiCurve(tNorm)
	target := m.engine.TargetPosition(tNorm, m.newsActive)

	// Prometheus metrics
	metrics.HedgeTNorm.Set(tNorm)
	metrics.HedgeCurveValue.Set(curve)
	metrics.HedgeTargetQty.Set(target)
	metrics.HedgeGeminiQty.Set(m.portfolio.Qty)
	metrics.HedgeGeminiRealised.Set(m.portfolio.Realised)
	metrics.HedgeGeminiUnrealised.Set(m.portfolio.Unrealised)
	metrics.HedgeCombinedPnL.Set(combined)
	metrics.HedgePolyPnL.Set(polyPnL)
	if m.newsActive {
		metrics.HedgeNewsActive.Set(1)
	} else {
		metrics.HedgeNewsActive.Set(0)
	}
	if directive.Action != "hold" {
		metrics.HedgeActionsTotal.WithLabelValues(directive.Action).Inc()
	}

	entry := HedgeLogEntry{
		Timestamp:      time.Now().UTC().Format(time.RFC3339),
		GameID:         lastGameID,
		TNorm:          tNorm,
		PolyPosition:   polySide,
		PolyQty:        polyQty,
		PolyEntry:      polyEntry,
		PolyMid:        polyMid,
		PolyPnL:        polyPnL,
		GeminiCurve:    curve,
		GeminiTarget:   target,
		GeminiPosition: m.portfolio.Position,
		GeminiQty:      m.portfolio.Qty,
		GeminiEntry:    m.portfolio.Entry,
		GeminiRealised: m.portfolio.Realised,
		GeminiUnreal:   m.portfolio.Unrealised,
		Action:         directive.Action,
		DeltaQty:       directive.DeltaQty,
		Reason:         directive.Reason,
		CombinedPnL:    combined,
		GeminiBid:      gemBid,
		GeminiAsk:      gemAsk,
		NewsActive:     m.newsActive,
	}

	if directive.Action != "hold" {
		log.Printf("[hedge] %s side=%s delta=%.1f target=%.1f reason=%s | %s | combined=$%.4f",
			directive.Action, directive.Side, directive.DeltaQty, directive.TargetQty,
			directive.Reason, m.portfolio.Summary(), combined)
	} else if statusTick {
		newsStr := "no"
		if m.newsActive {
			newsStr = "yes"
		}
		sigStr := "stale"
		if m.signalFresh {
			sigStr = "fresh"
		}
		log.Printf("[hedge] status: t=%.3f curve=%.3f target=%.1f | poly=%s qty=%.1f pnl=$%.4f | %s | news=%s signal=%s reason=%s",
			tNorm, curve, target, polySide, polyQty, polyPnL,
			m.portfolio.Summary(), newsStr, sigStr, directive.Reason)
	}

	m.appendLog(entry)
}

// pairBook returns the best bid/ask from the books map, filtered to keys in the allowed set.
// If the allowed set is empty, returns the first available book (backwards compat).
func (m *HedgeMonitor) pairBook(books map[string]GeminiBook, allowed map[string]bool) (bid, ask float64) {
	if len(allowed) > 0 {
		for key, b := range books {
			if allowed[key] {
				return b.BestBid, b.BestAsk
			}
		}
		return 0, 0
	}
	// Fallback: first available
	for _, b := range books {
		return b.BestBid, b.BestAsk
	}
	return 0, 0
}

func (m *HedgeMonitor) readSignal() (*PolySignal, error) {
	data, err := os.ReadFile(m.signalPath)
	if err != nil {
		return nil, err
	}
	var sig PolySignal
	if err := json.Unmarshal(data, &sig); err != nil {
		return nil, fmt.Errorf("parse poly_position.json: %w", err)
	}
	return &sig, nil
}

func (m *HedgeMonitor) appendLog(entry HedgeLogEntry) {
	if !m.logDirOK {
		os.MkdirAll(filepath.Dir(m.logPath), 0755)
		m.logDirOK = true
	}
	f, err := os.OpenFile(m.logPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return
	}
	defer f.Close()
	data, _ := json.Marshal(entry)
	f.Write(data)
	f.Write([]byte("\n"))
}
