package polymarket

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/hrishabhayush/polyxgemini/internal/config"
	"github.com/hrishabhayush/polyxgemini/internal/exchange"
)

var nonAlnum = regexp.MustCompile(`[^a-z0-9]+`)

type Client struct {
	cfg        config.PolymarketConfig
	httpClient *http.Client
}

// gammaID handles Gamma market IDs that may arrive as either
// JSON numbers or JSON strings.
type gammaID int

func (g *gammaID) UnmarshalJSON(data []byte) error {
	s := strings.TrimSpace(string(data))
	if s == "null" || s == "" {
		*g = 0
		return nil
	}
	// String form: "12345"
	if strings.HasPrefix(s, "\"") && strings.HasSuffix(s, "\"") {
		s = strings.Trim(s, "\"")
	}
	v, err := strconv.Atoi(s)
	if err != nil {
		return fmt.Errorf("gamma id parse: %w", err)
	}
	*g = gammaID(v)
	return nil
}

func NewClient(cfg config.PolymarketConfig) *Client {
	return &Client{cfg: cfg, httpClient: &http.Client{Timeout: 10 * time.Second}}
}

// DiscoverMarkets lists Polymarket markets via the Gamma API.
// active=true filters to open markets; minVolume filters by minimum total volume (USD).
// Run on startup and re-scan every 15–30 min to pick up newly created markets.
// Returns a slice of ResolvedMarket with ConditionID and ClobTokenIDs populated.
func (c *Client) DiscoverMarkets(ctx context.Context, active bool, minVolume float64) ([]ResolvedMarket, error) {
	return c.discoverWithLimit(ctx, active, minVolume, 100)
}

// discoverWithLimit is the shared implementation for Gamma market list fetching.
func (c *Client) discoverWithLimit(ctx context.Context, active bool, minVolume float64, limit int) ([]ResolvedMarket, error) {
	params := url.Values{}
	if active {
		params.Set("active", "true")
	}
	if minVolume > 0 {
		params.Set("volume_num_min", strconv.FormatFloat(minVolume, 'f', 2, 64))
	}
	params.Set("limit", strconv.Itoa(limit))

	endpoint := c.cfg.GammaBaseURL + "/markets?" + params.Encode()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("gamma: build request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("gamma: request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("gamma: unexpected status %d", resp.StatusCode)
	}

	var raw []gammaMarketResponse
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return nil, fmt.Errorf("gamma: decode: %w", err)
	}

	markets := make([]ResolvedMarket, 0, len(raw))
	for _, m := range raw {
		var tokenIDs []string
		if err := json.Unmarshal([]byte(m.ClobTokenIDs), &tokenIDs); err != nil || len(tokenIDs) < 2 {
			continue // skip markets with malformed token IDs
		}
		markets = append(markets, ResolvedMarket{
			Slug:             m.Slug,
			Question:         m.Question,
			ConditionID:      m.ConditionID,
			ClobTokenIDs:     [2]string{tokenIDs[0], tokenIDs[1]},
			GammaID:          int(m.ID),
			EventID:          firstEventID(m.Events),
			ResolutionSource: m.ResolutionSource,
			Description:      m.Description,
		})
	}
	return markets, nil
}

// FindMarket searches active markets for the first case-insensitive substring match
// on Question or Slug. Scans up to 500 markets sorted by 24-hour volume (highest
// first) so that current, liquid markets are found before stale/historical ones.
// Intended for test/report use — not for high-frequency polling.
func (c *Client) FindMarket(ctx context.Context, query string) (*ResolvedMarket, error) {
	// First try direct slug resolution for deterministic lookup.
	// Handles raw slug input and full Polymarket event URLs.
	if slug := queryToSlug(query); slug != "" {
		if m, err := c.ResolveMarket(ctx, slug); err == nil {
			return m, nil
		}
	}

	params := url.Values{}
	params.Set("active", "true")
	params.Set("limit", "500")
	params.Set("order", "volume24hr")
	params.Set("ascending", "false")

	endpoint := c.cfg.GammaBaseURL + "/markets?" + params.Encode()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("gamma: FindMarket: build request: %w", err)
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("gamma: FindMarket: request: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("gamma: FindMarket: status %d", resp.StatusCode)
	}
	var raw []gammaMarketResponse
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return nil, fmt.Errorf("gamma: FindMarket: decode: %w", err)
	}

	qRaw := strings.ToLower(strings.TrimSpace(query))
	qNorm := normalizeSearchText(query)
	for _, m := range raw {
		questionLower := strings.ToLower(m.Question)
		slugLower := strings.ToLower(m.Slug)
		questionNorm := normalizeSearchText(m.Question)
		slugNorm := normalizeSearchText(m.Slug)

		if strings.Contains(questionLower, qRaw) ||
			strings.Contains(slugLower, qRaw) ||
			(qNorm != "" && (strings.Contains(questionNorm, qNorm) ||
				strings.Contains(qNorm, questionNorm) ||
				strings.Contains(slugNorm, qNorm))) {
			var tokenIDs []string
			if err := json.Unmarshal([]byte(m.ClobTokenIDs), &tokenIDs); err != nil || len(tokenIDs) < 2 {
				continue
			}
			return &ResolvedMarket{
				Slug:             m.Slug,
				Question:         m.Question,
				ConditionID:      m.ConditionID,
				ClobTokenIDs:     [2]string{tokenIDs[0], tokenIDs[1]},
				GammaID:          int(m.ID),
				EventID:          firstEventID(m.Events),
				ResolutionSource: m.ResolutionSource,
				Description:      m.Description,
			}, nil
		}
	}
	return nil, fmt.Errorf("polymarket: no active market matching %q", query)
}

func queryToSlug(query string) string {
	q := strings.TrimSpace(query)
	if q == "" {
		return ""
	}
	if strings.Contains(q, "/event/") {
		parts := strings.Split(q, "/event/")
		if len(parts) < 2 {
			return ""
		}
		tail := strings.Trim(parts[1], "/")
		if tail == "" {
			return ""
		}
		seg := strings.Split(tail, "/")
		if len(seg) == 0 {
			return ""
		}
		return strings.ToLower(strings.TrimSpace(seg[0]))
	}
	// Treat single-token kebab-case input as slug.
	if strings.Contains(q, "-") && !strings.Contains(q, " ") {
		return strings.ToLower(q)
	}
	return ""
}

func normalizeSearchText(s string) string {
	out := strings.ToLower(strings.TrimSpace(s))
	out = nonAlnum.ReplaceAllString(out, " ")
	return strings.TrimSpace(strings.Join(strings.Fields(out), " "))
}

// FetchMarketByConditionID returns full metadata for a single market by its condition ID.
//
// The Gamma API has no working filter param for condition ID, so this fetches
// the full market list and scans client-side. Adequate for our use case (<500 markets).
func (c *Client) FetchMarketByConditionID(ctx context.Context, conditionID string) (*ResolvedMarket, error) {
	// No volume filter — we need to find any market regardless of volume.
	markets, err := c.DiscoverMarkets(ctx, false, 0)
	if err != nil {
		return nil, fmt.Errorf("gamma: FetchMarketByConditionID: %w", err)
	}
	for _, m := range markets {
		if m.ConditionID == conditionID {
			return &m, nil
		}
	}
	return nil, fmt.Errorf("gamma: no market found for conditionID %s", conditionID)
}

// FetchMarkets implements exchange.Exchange. Delegates to DiscoverMarkets with default filters.
func (c *Client) FetchMarkets(ctx context.Context) ([]exchange.Market, error) {
	resolved, err := c.DiscoverMarkets(ctx, true, 0)
	if err != nil {
		return nil, err
	}
	out := make([]exchange.Market, 0, len(resolved))
	for _, r := range resolved {
		out = append(out, exchange.Market{
			ID:   r.ConditionID,
			Name: r.Question,
			Outcomes: []exchange.Outcome{
				{ID: r.ClobTokenIDs[0], Label: "YES"},
				{ID: r.ClobTokenIDs[1], Label: "NO"},
			},
		})
	}
	return out, nil
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
	Slug             string
	Question         string
	ConditionID      string
	ClobTokenIDs     [2]string // [YES, NO]
	GammaID          int       // numeric Gamma market ID (used for comments endpoint)
	EventID          int       // Gamma event ID (comments parent_entity_id for Event threads)
	ResolutionSource string    // URL or text describing how the market resolves
	Description      string    // full market description / rules
}

// gammaMarketResponse is the raw Gamma API response for a single market.
// Note: clobTokenIds comes back as a JSON-encoded string, not a native array.
type gammaMarketResponse struct {
	ID               gammaID `json:"id"`
	Slug             string `json:"slug"`
	Question         string `json:"question"`
	ConditionID      string `json:"conditionId"`
	ClobTokenIDs     string `json:"clobTokenIds"`
	Events           []gammaEvent `json:"events"`
	ResolutionSource string `json:"resolutionSource"`
	Description      string `json:"description"`
}

type gammaEvent struct {
	ID gammaID `json:"id"`
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
		Slug:             m.Slug,
		Question:         m.Question,
		ConditionID:      m.ConditionID,
		ClobTokenIDs:     [2]string{tokenIDs[0], tokenIDs[1]},
		GammaID:          int(m.ID),
		EventID:          firstEventID(m.Events),
		ResolutionSource: m.ResolutionSource,
		Description:      m.Description,
	}, nil
}

func firstEventID(events []gammaEvent) int {
	if len(events) == 0 {
		return 0
	}
	return int(events[0].ID)
}
