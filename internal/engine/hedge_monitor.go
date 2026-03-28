package engine

import (
	"encoding/json"
	"fmt"
	"log"
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
	Timestamp       string  `json:"timestamp"`
	GameID          string  `json:"game_id"`
	TNorm           float64 `json:"t_norm"`
	PolyPosition    string  `json:"poly_position"`
	PolyQty         float64 `json:"poly_qty"`
	PolyPnL         float64 `json:"poly_pnl"`
	GeminiCurve     float64 `json:"gemini_curve"`
	GeminiTarget    float64 `json:"gemini_target"`
	GeminiPosition  string  `json:"gemini_position"`
	GeminiQty       float64 `json:"gemini_qty"`
	GeminiEntry     float64 `json:"gemini_entry"`
	GeminiRealised  float64 `json:"gemini_realised"`
	GeminiUnreal    float64 `json:"gemini_unrealised"`
	Action          string  `json:"action"`
	DeltaQty        float64 `json:"delta_qty"`
	Reason          string  `json:"reason"`
	CombinedPnL     float64 `json:"combined_pnl"`
	GeminiBid       float64 `json:"gemini_bid"`
	GeminiAsk       float64 `json:"gemini_ask"`
}

// HedgeMonitor polls hedge_state.json and manages the Gemini hedge layer.
type HedgeMonitor struct {
	engine        *HedgeEngine
	portfolio     *GeminiPaperPortfolio
	statePath     string
	logPath       string
	pollInterval  time.Duration
	pairToSymbols map[string]string // pair name -> gemini instrument symbol

	mu    sync.Mutex
	books map[string]GeminiBook // symbol (lowercase) -> latest book
	done  chan struct{}
}

// NewHedgeMonitor creates the polling goroutine (not yet started).
func NewHedgeMonitor(cfg HedgeConfig) *HedgeMonitor {
	return &HedgeMonitor{
		engine:        NewHedgeEngine(cfg),
		portfolio:     &GeminiPaperPortfolio{},
		statePath:     "data/hedge_state.json",
		logPath:       "data/hedge_log.jsonl",
		pollInterval:  time.Duration(cfg.PollIntervalMS) * time.Millisecond,
		pairToSymbols: make(map[string]string),
		books:         make(map[string]GeminiBook),
		done:          make(chan struct{}),
	}
}

// UpdateBook is called by the Gemini WS readLoop to feed book prices.
func (m *HedgeMonitor) UpdateBook(symbol string, bestBid, bestAsk float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.books[strings.ToLower(symbol)] = GeminiBook{BestBid: bestBid, BestAsk: bestAsk}
}

// Run starts the polling loop. Call in a goroutine.
func (m *HedgeMonitor) Run() {
	log.Printf("[hedge] monitor started (poll=%v)", m.pollInterval)
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

func (m *HedgeMonitor) tick() {
	state, err := m.readState()
	if err != nil {
		return // file doesn't exist yet or parse error — silent
	}

	// Find Gemini book for this game's market pair
	bid, ask := m.bestBook()
	if bid <= 0 && ask <= 0 {
		return // no Gemini book data yet
	}

	// Evaluate hedge directive
	directive := m.engine.Evaluate(
		*state,
		m.portfolio.Position, m.portfolio.Qty,
		bid, ask,
	)

	// Also check rebalance (trim) every tick
	if directive.Action == "hold" {
		rebal := m.engine.CheckRebalance(*state, m.portfolio.Position, m.portfolio.Qty)
		if rebal.Action == "reduce" {
			directive = rebal
		}
	}

	// Execute on paper portfolio
	switch directive.Action {
	case "buy_opposite", "buy_same":
		price := ask + m.engine.Cfg.GeminiHalfSpread // fill at ask + half spread
		m.portfolio.OnFill(directive.Side, directive.DeltaQty, price)
	case "reduce":
		reduceQty := -directive.DeltaQty
		if reduceQty > 0 {
			bidPrice := bid - m.engine.Cfg.GeminiHalfSpread
			m.portfolio.ReducePosition(reduceQty, bidPrice)
		}
	case "close":
		if m.portfolio.Qty > 0 {
			bidPrice := bid - m.engine.Cfg.GeminiHalfSpread
			m.portfolio.ReducePosition(m.portfolio.Qty, bidPrice)
		}
	}

	// Mark to market
	m.portfolio.MarkToMarket(bid, ask)

	// Compute combined PnL
	combined := m.portfolio.CombinedPnL(0, state.PnL) // Poly unrealised from state

	// Log
	curve := m.engine.GeminiCurve(state.TNorm)
	target := m.engine.TargetPosition(state.TNorm, state.NewsActive)

	entry := HedgeLogEntry{
		Timestamp:      time.Now().UTC().Format(time.RFC3339),
		GameID:         state.GameID,
		TNorm:          state.TNorm,
		PolyPosition:   state.PolyPosition,
		PolyQty:        state.PolyQty,
		PolyPnL:        state.PnL,
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
		GeminiBid:      bid,
		GeminiAsk:      ask,
	}

	if directive.Action != "hold" {
		log.Printf("[hedge] %s side=%s delta=%.1f target=%.1f reason=%s | %s | combined=$%.4f",
			directive.Action, directive.Side, directive.DeltaQty, directive.TargetQty,
			directive.Reason, m.portfolio.Summary(), combined)
	}

	m.appendLog(entry)
}

func (m *HedgeMonitor) readState() (*HedgeState, error) {
	data, err := os.ReadFile(m.statePath)
	if err != nil {
		return nil, err
	}
	var s HedgeState
	if err := json.Unmarshal(data, &s); err != nil {
		return nil, fmt.Errorf("parse hedge_state.json: %w", err)
	}
	return &s, nil
}

func (m *HedgeMonitor) bestBook() (bid, ask float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	// Use the first available book (single-game mode for now)
	for _, b := range m.books {
		return b.BestBid, b.BestAsk
	}
	return 0, 0
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
