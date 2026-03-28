package gemini

import (
	"crypto/hmac"
	"crypto/sha512"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/hrishabhayush/polyxgemini/internal/arb"
	"github.com/hrishabhayush/polyxgemini/internal/config"
	"github.com/hrishabhayush/polyxgemini/internal/metrics"
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
	cfg          config.GeminiConfig
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
	PairID   string
	Outcome  string // "yes" or "no"
	Category string // "sports" or "crypto"
}

// PairMapping tells the WS client which instrument symbols belong to which arb pair.
type PairMapping struct {
	PairID           string
	InstrumentSymbol string // as returned by API (uppercase)
	Outcome          string // "yes" or "no"
	Category         string // "sports" (default) or "crypto"
}

// NewWSClient creates a Gemini WS client for the given resolved events.
func NewWSClient(cfg config.GeminiConfig, markets []ResolvedEvent, updates chan<- arb.PriceUpdate, pairMappings []PairMapping) *WSClient {
	wsURL := cfg.WSURL
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
			PairID:   pm.PairID,
			Outcome:  pm.Outcome,
			Category: pm.Category,
		}
	}

	return &WSClient{
		wsURL:        wsURL,
		cfg:          cfg,
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

// Connect establishes the authenticated WS connection and subscribes to bookTicker for all contracts.
func (w *WSClient) Connect() error {
	// Build auth headers for the handshake
	nonce := strconv.FormatInt(time.Now().Unix(), 10)
	b64Payload := base64.StdEncoding.EncodeToString([]byte(nonce))
	mac := hmac.New(sha512.New384, []byte(w.cfg.APISecret))
	mac.Write([]byte(b64Payload))
	sig := hex.EncodeToString(mac.Sum(nil))

	headers := http.Header{}
	headers.Set("X-GEMINI-APIKEY", w.cfg.APIKey)
	headers.Set("X-GEMINI-NONCE", nonce)
	headers.Set("X-GEMINI-PAYLOAD", b64Payload)
	headers.Set("X-GEMINI-SIGNATURE", sig)

	dialer := websocket.Dialer{
		HandshakeTimeout: 10 * time.Second,
	}
	conn, _, err := dialer.Dial(w.wsURL, headers)
	if err != nil {
		return fmt.Errorf("gemini ws dial failed: %w", err)
	}
	w.conn = conn
	log.Println("gemini ws: connected (authenticated)")
	metrics.WSConnected.WithLabelValues("gemini").Set(1)

	// Subscribe to bookTicker for each contract
	var streams []string
	for _, m := range w.markets {
		for _, c := range m.Contracts {
			streams = append(streams, c.InstrumentSymbol+"@bookTicker")
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
	defer func() {
		metrics.WSConnected.WithLabelValues("gemini").Set(0)
		metrics.WSDisconnectsTotal.WithLabelValues("gemini").Inc()
	}()
	for {
		_, msg, err := w.conn.ReadMessage()
		if err != nil {
			log.Printf("gemini ws: read error: %v", err)
			return
		}

		metrics.WSMessagesTotal.WithLabelValues("gemini", "bookTicker").Inc()
		metrics.WSLastMessageTimestamp.WithLabelValues("gemini").Set(float64(time.Now().Unix()))

		var bt BookTicker
		if err := json.Unmarshal(msg, &bt); err != nil {
			continue
		}
		if bt.Symbol == "" || bt.BestAsk == "" {
			continue
		}

		label := w.label(strings.ToLower(bt.Symbol))
		log.Printf("[GEMI book]  %-40s | buy @ %s¢ | ask_qty: %s",
			label, centsFromDecimal(bt.BestAsk), bt.AskQty)

		// Emit book metrics
		var askF, bidF, qtyF float64
		fmt.Sscanf(bt.BestAsk, "%f", &askF)
		fmt.Sscanf(bt.BestBid, "%f", &bidF)
		fmt.Sscanf(bt.AskQty, "%f", &qtyF)
		metrics.BookBestAsk.WithLabelValues("gemini", bt.Symbol).Set(askF * 100)
		metrics.BookBestBid.WithLabelValues("gemini", bt.Symbol).Set(bidF * 100)

		// Push to arb detector if this symbol is in a pair
		if info, ok := w.symbolToPair[strings.ToLower(bt.Symbol)]; ok && w.updates != nil {
			w.updates <- arb.PriceUpdate{
				PairID:   info.PairID,
				Exchange: "gemini",
				Outcome:  info.Outcome,
				AskPrice: askF,
				AskQty:   qtyF,
				Category: info.Category,
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
