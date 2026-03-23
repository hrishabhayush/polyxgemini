package gemini

import (
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"sync"

	"github.com/gorilla/websocket"
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
	wsURL   string
	conn    *websocket.Conn
	markets []ResolvedEvent
	// Maps instrumentSymbol -> human label
	symbolLabels map[string]string
	mu           sync.Mutex
	done         chan struct{}
}

// NewWSClient creates a Gemini WS client for the given resolved events.
func NewWSClient(wsURL string, markets []ResolvedEvent) *WSClient {
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
	return &WSClient{
		wsURL:        wsURL,
		markets:      markets,
		symbolLabels: labels,
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
			// Gemini WS requires lowercase stream names
			streams = append(streams, strings.ToLower(c.InstrumentSymbol)+"@bookTicker")
		}
	}

	subMsg := map[string]interface{}{
		"id":     "1",
		"method": "subscribe",
		"params": streams,
	}
	if err := conn.WriteJSON(subMsg); err != nil {
		return fmt.Errorf("gemini ws subscribe failed: %w", err)
	}
	log.Printf("gemini ws: subscribed to %d streams", len(streams))

	// Read loop
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

		log.Printf("[gemini] %-40s | buy @ %s¢ | ask_qty: %s",
			w.label(bt.Symbol), centsFromDecimal(bt.BestAsk), bt.AskQty)
	}
}

// centsFromDecimal converts "0.87" -> "87.0"
func centsFromDecimal(s string) string {
	// Gemini prices are already 0-1 decimals
	// Parse and multiply by 100 for cents display
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
