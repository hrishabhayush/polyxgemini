package gemini

import (
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"sync"

	"github.com/gorilla/websocket"
	"github.com/hrishabhayush/polyxgemini/internal/arb"
)

const defaultWSURL = "wss://ws.gemini.com"

// BookTicker is the real-time best bid/ask update from Gemini WS.
type BookTicker struct {
	UpdateID  int64  `json:"u"`
	EventTime int64  `json:"E"`
	Symbol    string `json:"s"`
	BestBid   string `json:"b"`
	BidQty    string `json:"B"`
	BestAsk   string `json:"a"`
	AskQty    string `json:"A"`
}

// WSClient connects to Gemini WebSocket for prediction market data.
type WSClient struct {
	wsURL        string
	conn         *websocket.Conn
	markets      []ResolvedEvent
	symbolLabels map[string]string
	// Maps lowercase instrumentSymbol -> {pairID, outcome}
	symbolToPair map[string]symbolPairInfo
	updates      chan<- arb.PriceUpdate
	mu           sync.Mutex
	done         chan struct{}
}

type symbolPairInfo struct {
	PairID  string
	Outcome string // "yes" or "no"
}

// PairMapping tells the WS client which instrument symbols belong to which arb pair.
type PairMapping struct {
	PairID           string
	InstrumentSymbol string // as returned by API (uppercase)
	Outcome          string // "yes" or "no"
}

// NewWSClient creates a Gemini WS client for the given resolved events.
func NewWSClient(wsURL string, markets []ResolvedEvent, updates chan<- arb.PriceUpdate, pairMappings []PairMapping) *WSClient {
	if wsURL == "" {
		wsURL = defaultWSURL
	}
	labels := make(map[string]string)
	for _, m := range markets {
		for _, c := range m.Contracts {
			short := m.Title
			if len(short) > 25 {
				short = short[:25] + "..."
			}
			labels[strings.ToLower(c.InstrumentSymbol)] = fmt.Sprintf("%s %s", short, c.Label)
		}
	}

	symbolToPair := make(map[string]symbolPairInfo)
	for _, pm := range pairMappings {
		symbolToPair[strings.ToLower(pm.InstrumentSymbol)] = symbolPairInfo{
			PairID:  pm.PairID,
			Outcome: pm.Outcome,
		}
	}

	return &WSClient{
		wsURL:        wsURL,
		markets:      markets,
		symbolLabels: labels,
		symbolToPair: symbolToPair,
		updates:      updates,
		done:         make(chan struct{}),
	}
}

func (w *WSClient) label(symbol string) string {
	if l, ok := w.symbolLabels[symbol]; ok {
		return l
	}
	return symbol
}

// Connect establishes the WS connection and subscribes to bookTicker for all contracts.
func (w *WSClient) Connect() error {
	conn, _, err := websocket.DefaultDialer.Dial(w.wsURL, nil)
	if err != nil {
		return fmt.Errorf("gemini ws dial failed: %w", err)
	}
	w.conn = conn
	log.Println("gemini ws: connected")

	// Subscribe to bookTicker for each contract
	var streams []string
	for _, m := range w.markets {
		for _, c := range m.Contracts {
			streams = append(streams, strings.ToLower(c.InstrumentSymbol)+"@bookTicker")
		}
	}

	subMsg := map[string]any{
		"id":     "1",
		"method": "subscribe",
		"params": streams,
	}
	if err := conn.WriteJSON(subMsg); err != nil {
		return fmt.Errorf("gemini ws subscribe failed: %w", err)
	}
	log.Printf("gemini ws: subscribed to %d streams", len(streams))

	go w.readLoop()

	return nil
}

func (w *WSClient) readLoop() {
	defer close(w.done)
	for {
		_, msg, err := w.conn.ReadMessage()
		if err != nil {
			log.Printf("gemini ws: read error: %v", err)
			return
		}

		var bt BookTicker
		if err := json.Unmarshal(msg, &bt); err != nil {
			continue
		}
		if bt.Symbol == "" || bt.BestAsk == "" {
			continue
		}

		log.Printf("[GEMI book]  %-40s | buy @ %s¢ | ask_qty: %s",
			w.label(bt.Symbol), centsFromDecimal(bt.BestAsk), bt.AskQty)

		// Push to arb detector if this symbol is in a pair
		if info, ok := w.symbolToPair[bt.Symbol]; ok && w.updates != nil {
			var askF float64
			fmt.Sscanf(bt.BestAsk, "%f", &askF)
			w.updates <- arb.PriceUpdate{
				PairID:   info.PairID,
				Exchange: "gemini",
				Outcome:  info.Outcome,
				AskPrice: askF,
			}
		}
	}
}

// centsFromDecimal converts "0.87" -> "87.0"
func centsFromDecimal(s string) string {
	var f float64
	fmt.Sscanf(s, "%f", &f)
	return fmt.Sprintf("%.1f", f*100)
}

// Close disconnects the WebSocket.
func (w *WSClient) Close() error {
	if w.conn != nil {
		return w.conn.Close()
	}
	return nil
}
