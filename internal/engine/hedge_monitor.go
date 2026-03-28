package engine

import (
	"encoding/json"
	"log"
	"math"
	"os"
	"strings"
	"sync"
	"time"
)

// GeminiBook holds the latest bid/ask for a Gemini contract.
type GeminiBook struct {
	BestBid float64
	BestAsk float64
}

// HedgeLogEntry is one line in data/hedge_log.jsonl.
type HedgeLogEntry struct {
	Timestamp      string  `json:"timestamp"`
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

// HedgeMonitor watches Poly + Gemini book prices and manages the hedge layer.
type HedgeMonitor struct {
	engine    *HedgeEngine
	portfolio *GeminiPaperPortfolio
	logPath   string
	pollInterval time.Duration

	// Game timing: t_norm = (now - gameStart) / gameDuration
	gameStart    time.Time
	gameDuration time.Duration

	// Poly paper position (tracked from WS book updates)
	polyPosition string  // "HOME" | "AWAY" | ""
	polyQty      float64
	polyEntry    float64

	// News detection: tracks Poly mid-price for large shifts
	lastPolyMid   float64
	newsActive    bool
	newsThreshold float64 // fractional shift to trigger news (e.g. 0.10 = 10%)

	mu        sync.Mutex
	polyBooks map[string]GeminiBook // tokenID -> latest book
	gemBooks  map[string]GeminiBook // symbol (lowercase) -> latest book
	done      chan struct{}
}

// NewHedgeMonitor creates the monitor (not yet started).
func NewHedgeMonitor(cfg HedgeConfig, gameStart time.Time, gameDuration time.Duration) *HedgeMonitor {
	pollMS := cfg.PollIntervalMS
	if pollMS <= 0 {
		pollMS = 5000
	}
	return &HedgeMonitor{
		engine:        NewHedgeEngine(cfg),
		portfolio:     &GeminiPaperPortfolio{},
		logPath:       "data/hedge_log.jsonl",
		pollInterval:  time.Duration(pollMS) * time.Millisecond,
		gameStart:     gameStart,
		gameDuration:  gameDuration,
		newsThreshold: 0.10,
		polyBooks:     make(map[string]GeminiBook),
		gemBooks:      make(map[string]GeminiBook),
		done:          make(chan struct{}),
	}
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

// SetPolyPosition sets the simulated Poly position (called externally or from a signal).
func (m *HedgeMonitor) SetPolyPosition(side string, qty, entry float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.polyPosition = side
	m.polyQty = qty
	m.polyEntry = entry
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

func (m *HedgeMonitor) tNorm() float64 {
	if m.gameDuration <= 0 {
		return 0
	}
	elapsed := time.Since(m.gameStart).Seconds()
	total := m.gameDuration.Seconds()
	t := elapsed / total
	return math.Max(0, math.Min(1, t))
}

func (m *HedgeMonitor) tick() {
	m.mu.Lock()

	// Get best Poly book (first available)
	var polyBid, polyAsk float64
	for _, b := range m.polyBooks {
		polyBid = b.BestBid
		polyAsk = b.BestAsk
		break
	}

	// Get best Gemini book (first available)
	var gemBid, gemAsk float64
	for _, b := range m.gemBooks {
		gemBid = b.BestBid
		gemAsk = b.BestAsk
		break
	}

	polySide := m.polyPosition
	polyQty := m.polyQty
	polyEntry := m.polyEntry

	m.mu.Unlock()

	// Need both books
	if (polyBid <= 0 && polyAsk <= 0) || (gemBid <= 0 && gemAsk <= 0) {
		return
	}

	tNorm := m.tNorm()
	polyMid := (polyBid + polyAsk) / 2.0

	// News detection: large mid-price shift
	if m.lastPolyMid > 0 {
		shift := math.Abs(polyMid-m.lastPolyMid) / m.lastPolyMid
		if shift >= m.newsThreshold {
			m.newsActive = true
			log.Printf("[hedge] news detected: poly mid shifted %.1f%% (%.4f → %.4f)",
				shift*100, m.lastPolyMid, polyMid)
		}
	}
	m.lastPolyMid = polyMid

	// Compute Poly unrealised PnL
	polyPnL := 0.0
	if polySide != "" && polyQty > 0 && polyEntry > 0 {
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

	entry := HedgeLogEntry{
		Timestamp:      time.Now().UTC().Format(time.RFC3339),
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
	}

	m.appendLog(entry)
}

func (m *HedgeMonitor) appendLog(entry HedgeLogEntry) {
	f, err := os.OpenFile(m.logPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return
	}
	defer f.Close()
	data, _ := json.Marshal(entry)
	f.Write(data)
	f.Write([]byte("\n"))
}
