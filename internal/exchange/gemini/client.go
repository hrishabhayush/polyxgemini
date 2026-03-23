package gemini

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

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
	// TODO: POST base_url/order (authenticated, HMAC-SHA384)
	return nil, nil
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
