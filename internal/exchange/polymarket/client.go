package polymarket

import (
	"context"

	"github.com/hrishabhayush/polyxgemini/internal/config"
	"github.com/hrishabhayush/polyxgemini/internal/exchange"
)

type Client struct {
	cfg config.PolymarketConfig
}

func NewClient(cfg config.PolymarketConfig) *Client {
	return &Client{cfg: cfg}
}

func (c *Client) FetchMarkets(ctx context.Context) ([]exchange.Market, error) {
	// TODO: GET gamma_base_url/markets -> normalize to []Market
	return nil, nil
}

func (c *Client) FetchOrderBook(ctx context.Context, tokenID string) (*exchange.OrderBook, error) {
	// TODO: GET clob_base_url/order-book?token_id=tokenID
	return nil, nil
}

func (c *Client) PlaceOrder(ctx context.Context, order *exchange.OrderRequest) (*exchange.Order, error) {
	// TODO: POST clob_base_url/order (authenticated)
	return nil, nil
}

func (c *Client) CancelOrder(ctx context.Context, orderID string) error {
	// TODO: DELETE clob_base_url/order/orderID (authenticated)
	return nil
}

// ResolvedMarket holds the Gamma API response fields needed for trading.
type ResolvedMarket struct {
	Slug         string
	Question     string
	ConditionID  string
	ClobTokenIDs [2]string // [YES, NO]
}

func (c *Client) ResolveMarket(ctx context.Context, slug string) (*ResolvedMarket, error) {
	// TODO: GET gamma_base_url/markets?slug=slug
	// Parse response to extract conditionId and clobTokenIds
	return nil, nil
}
