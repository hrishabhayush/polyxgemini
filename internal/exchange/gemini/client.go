package gemini

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha512"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"time"

	"github.com/hrishabhayush/polyxgemini/internal/config"
	"github.com/hrishabhayush/polyxgemini/internal/exchange"
)

type Client struct {
	cfg        config.GeminiConfig
	httpClient *http.Client
}

func NewClient(cfg config.GeminiConfig) *Client {
	return &Client{cfg: cfg, httpClient: &http.Client{}}
}

func (c *Client) FetchMarkets(ctx context.Context) ([]exchange.Market, error) {
	// TODO: GET base_url/events -> normalize to []Market
	return nil, nil
}

func (c *Client) FetchOrderBook(ctx context.Context, symbol string) (*exchange.OrderBook, error) {
	// TODO: WS subscribe to symbol@bookTicker or use REST orderbook
	return nil, nil
}

func (c *Client) PlaceOrder(ctx context.Context, order *exchange.OrderRequest) (*exchange.Order, error) {
	nonce := strconv.FormatInt(time.Now().UnixMilli(), 10)
	endpoint := "/v1/order/new"

	payload := map[string]any{
		"request":       endpoint,
		"nonce":         nonce,
		"symbol":        order.MarketID,
		"amount":        strconv.FormatFloat(order.Quantity, 'f', 0, 64),
		"price":         strconv.FormatFloat(order.Price, 'f', 4, 64),
		"side":          "buy",
		"type":          "exchange limit",
		"options":       []string{"fill-or-kill"},
	}
	payloadJSON, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("gemini: marshal order: %w", err)
	}

	b64Payload := base64.StdEncoding.EncodeToString(payloadJSON)
	sig := c.sign(b64Payload)

	url := fmt.Sprintf("%s%s", c.baseURL(), endpoint)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader([]byte{}))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "text/plain")
	req.Header.Set("Content-Length", "0")
	req.Header.Set("X-GEMINI-APIKEY", c.cfg.APIKey)
	req.Header.Set("X-GEMINI-PAYLOAD", b64Payload)
	req.Header.Set("X-GEMINI-SIGNATURE", sig)
	req.Header.Set("Cache-Control", "no-cache")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("gemini: order request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("gemini: order returned status %d: %s", resp.StatusCode, string(respBody))
	}

	var result struct {
		OrderID       string `json:"order_id"`
		ExecutedAmt   string `json:"executed_amount"`
		IsLive        bool   `json:"is_live"`
		IsCancelled   bool   `json:"is_cancelled"`
	}
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("gemini: decode order response: %w", err)
	}

	var filledQty float64
	fmt.Sscanf(result.ExecutedAmt, "%f", &filledQty)

	status := "filled"
	if result.IsCancelled {
		status = "cancelled"
	} else if filledQty == 0 {
		status = "unfilled"
	}

	return &exchange.Order{
		ID:        result.OrderID,
		Status:    status,
		FilledQty: filledQty,
	}, nil
}

func (c *Client) sign(b64Payload string) string {
	mac := hmac.New(sha512.New384, []byte(c.cfg.APISecret))
	mac.Write([]byte(b64Payload))
	return hex.EncodeToString(mac.Sum(nil))
}

func (c *Client) CancelOrder(ctx context.Context, orderID string) error {
	// TODO: POST base_url/order/cancel (authenticated)
	return nil
}

// ResolvedContract holds one tradeable contract within an event.
type ResolvedContract struct {
	Label            string
	InstrumentSymbol string
	BestAsk          string
	BestBid          string
}

// ResolvedEvent holds a Gemini prediction market event with its contracts.
type ResolvedEvent struct {
	Title     string
	Ticker    string
	Category  string
	Contracts []ResolvedContract
}

// geminiEventsResponse is the raw API response wrapper.
type geminiEventsResponse struct {
	Data []geminiEvent `json:"data"`
}

type geminiEvent struct {
	Title    string           `json:"title"`
	Ticker   string           `json:"ticker"`
	Category string           `json:"category"`
	Status   string           `json:"status"`
	Contracts []geminiContract `json:"contracts"`
}

type geminiContract struct {
	Label            string       `json:"label"`
	InstrumentSymbol string       `json:"instrumentSymbol"`
	Prices           geminiPrices `json:"prices"`
	MarketState      string       `json:"marketState"`
}

type geminiPrices struct {
	BestBid string `json:"bestBid"`
	BestAsk string `json:"bestAsk"`
}

func (c *Client) baseURL() string {
	if c.cfg.UseSandbox {
		return c.cfg.SandboxURL
	}
	return c.cfg.BaseURL
}

// ResolveEvent looks up a prediction market event by ticker.
func (c *Client) ResolveEvent(ctx context.Context, ticker string) (*ResolvedEvent, error) {
	url := fmt.Sprintf("%s/events/%s", c.baseURL(), ticker)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("gemini api request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("gemini api returned status %d for ticker %s", resp.StatusCode, ticker)
	}

	var event geminiEvent
	if err := json.NewDecoder(resp.Body).Decode(&event); err != nil {
		return nil, fmt.Errorf("failed to decode gemini event: %w", err)
	}

	resolved := &ResolvedEvent{
		Title:    event.Title,
		Ticker:   event.Ticker,
		Category: event.Category,
	}

	for _, c := range event.Contracts {
		resolved.Contracts = append(resolved.Contracts, ResolvedContract{
			Label:            c.Label,
			InstrumentSymbol: c.InstrumentSymbol,
			BestAsk:          c.Prices.BestAsk,
			BestBid:          c.Prices.BestBid,
		})
	}

	return resolved, nil
}
