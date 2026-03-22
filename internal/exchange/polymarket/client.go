package polymarket

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/hrishabhayush/polyxgemini/internal/config"
	"github.com/hrishabhayush/polyxgemini/internal/exchange"
)

type Client struct {
	cfg        config.PolymarketConfig
	httpClient *http.Client
}

func NewClient(cfg config.PolymarketConfig) *Client {
	return &Client{cfg: cfg, httpClient: &http.Client{}}
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

// gammaMarketResponse is the raw Gamma API response for a single market.
// Note: clobTokenIds comes back as a JSON-encoded string, not a native array.
type gammaMarketResponse struct {
	Slug         string `json:"slug"`
	Question     string `json:"question"`
	ConditionID  string `json:"conditionId"`
	ClobTokenIDs string `json:"clobTokenIds"`
}

// ResolveMarket looks up a market by slug via the Gamma API and returns its token IDs.
func (c *Client) ResolveMarket(ctx context.Context, slug string) (*ResolvedMarket, error) {
	url := fmt.Sprintf("%s/markets?slug=%s", c.cfg.GammaBaseURL, slug)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("gamma api request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("gamma api returned status %d", resp.StatusCode)
	}

	var markets []gammaMarketResponse
	if err := json.NewDecoder(resp.Body).Decode(&markets); err != nil {
		return nil, fmt.Errorf("failed to decode gamma response: %w", err)
	}

	if len(markets) == 0 {
		return nil, fmt.Errorf("no market found for slug: %s", slug)
	}

	m := markets[0]

	// clobTokenIds is a JSON string like '["token1", "token2"]' — parse it
	var tokenIDs []string
	if err := json.Unmarshal([]byte(m.ClobTokenIDs), &tokenIDs); err != nil {
		return nil, fmt.Errorf("failed to parse clobTokenIds for %s: %w", slug, err)
	}
	if len(tokenIDs) < 2 {
		return nil, fmt.Errorf("market %s has fewer than 2 token IDs", slug)
	}

	return &ResolvedMarket{
		Slug:         m.Slug,
		Question:     m.Question,
		ConditionID:  m.ConditionID,
		ClobTokenIDs: [2]string{tokenIDs[0], tokenIDs[1]},
	}, nil
}
