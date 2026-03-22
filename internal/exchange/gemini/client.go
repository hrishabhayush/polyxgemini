package gemini

import (
	"context"

	"github.com/hrishabhayush/polyxgemini/internal/config"
	"github.com/hrishabhayush/polyxgemini/internal/exchange"
)

type Client struct {
	cfg config.GeminiConfig
}

func NewClient(cfg config.GeminiConfig) *Client {
	return &Client{cfg: cfg}
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
