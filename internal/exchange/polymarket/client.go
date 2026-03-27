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

func NewClient(cfg config.PolymarketConfig) *Client {
	return &Client{cfg: cfg, httpClient: &http.Client{Timeout: 10 * time.Second}}
}

// DiscoverMarkets lists Polymarket markets via the Gamma API.
// active=true filters to open markets; minVolume filters by minimum total volume (USD).
// Run on startup and re-scan every 15–30 min to pick up newly created markets.
// Returns a slice of ResolvedMarket with ConditionID and ClobTokenIDs populated.
func (c *Client) DiscoverMarkets(ctx context.Context, active bool, minVolume float64) ([]ResolvedMarket, error) {
	return c.discoverWithLimit(ctx, active, minVolume, 100, 0, nil)
}

// discoverWithLimit is the shared implementation for Gamma market list fetching.
func (c *Client) discoverWithLimit(
	ctx context.Context,
	active bool,
	minVolume float64,
	limit int,
	offset int,
	extra map[string]string,
) ([]ResolvedMarket, error) {
	params := url.Values{}
	if active {
		params.Set("active", "true")
	}
	if minVolume > 0 {
		params.Set("volume_num_min", strconv.FormatFloat(minVolume, 'f', 2, 64))
	}
	params.Set("limit", strconv.Itoa(limit))
	if offset > 0 {
		params.Set("offset", strconv.Itoa(offset))
	}
	for k, v := range extra {
		params.Set(k, v)
	}

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
		rm := toResolvedMarket(m)
		if rm == nil {
			continue
		}
		markets = append(markets, *rm)
	}
	return markets, nil
}

// DiscoverOpenMarkets paginates Gamma with closed=false until it finds target open markets.
// This is useful as a fallback when the active=true feed is inconsistent.
func (c *Client) DiscoverOpenMarkets(ctx context.Context, minVolume float64, target int) ([]ResolvedMarket, error) {
	if target <= 0 {
		target = 100
	}
	pageSize := 200
	maxPages := 25
	seen := make(map[string]bool, target)
	out := make([]ResolvedMarket, 0, target)

	for page := 0; page < maxPages && len(out) < target; page++ {
		offset := page * pageSize
		batch, err := c.discoverWithLimit(
			ctx,
			false, // don't rely on active=true semantics
			minVolume,
			pageSize,
			offset,
			map[string]string{
				"closed":    "false",
				"order":     "volume24hr",
				"ascending": "false",
			},
		)
		if err != nil {
			return nil, err
		}
		if len(batch) == 0 {
			break
		}
		for _, m := range batch {
			if m.Closed {
				continue
			}
			if seen[m.ConditionID] {
				continue
			}
			seen[m.ConditionID] = true
			out = append(out, m)
			if len(out) >= target {
				break
			}
		}
	}
	return out, nil
}

// DiscoverResolvedMarkets fetches closed markets with known outcomes from the Gamma API.
// Useful for building ML training datasets from historical resolutions.
// offset enables pagination; the Gamma API returns up to limit results per call.
// closedAfter, if non-zero, filters to markets whose end date is after that time
// (use this to restrict to recently-resolved markets where CLOB data still exists).
func (c *Client) DiscoverResolvedMarkets(ctx context.Context, limit, offset int, closedAfter time.Time) ([]ResolvedMarket, error) {
	params := url.Values{}
	params.Set("closed", "true")
	params.Set("limit", strconv.Itoa(limit))
	params.Set("offset", strconv.Itoa(offset))
	params.Set("order", "endDateIso")
	params.Set("ascending", "false")
	if !closedAfter.IsZero() {
		params.Set("endDateMin", closedAfter.Format(time.RFC3339))
	}

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
		if resolveOutcomeFromGamma(m) == "" {
			continue // skip markets without a resolved outcome
		}
		rm := toResolvedMarket(m)
		if rm == nil {
			continue
		}
		markets = append(markets, *rm)
	}
	return markets, nil
}

// FindMarket searches active markets for the first case-insensitive substring match
// on Question or Slug. Scans up to 500 markets sorted by 24-hour volume (highest
// first) so that current, liquid markets are found before stale/historical ones.
// Intended for test/report use — not for high-frequency polling.
// FindMarket searches for a single binary market by slug, event slug, URL, or
// keyword query. Returns an error if zero or multiple markets are found — use
// FindMarketsInEvent for queries that may match an event with multiple markets.
func (c *Client) FindMarket(ctx context.Context, query string) (*ResolvedMarket, error) {
	// First try direct slug resolution for deterministic lookup.
	// Handles raw slug input and full Polymarket event URLs.
	if slug := queryToSlug(query); slug != "" {
		if m, err := c.ResolveMarket(ctx, slug); err == nil {
			return m, nil
		}
		// Slug didn't match a market — try it as an event slug.
		if markets, err := c.ResolveEvent(ctx, slug); err == nil && len(markets) == 1 {
			return &markets[0], nil
		} else if err == nil && len(markets) > 1 {
			return nil, fmt.Errorf("event %q contains %d markets — use FindMarketsInEvent or be more specific", slug, len(markets))
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
			rm := toResolvedMarket(m)
			if rm == nil {
				continue
			}
			return rm, nil
		}
	}
	return nil, fmt.Errorf("polymarket: no active market matching %q", query)
}

func queryToSlug(query string) string {
	q := strings.TrimSpace(query)
	if q == "" {
		return ""
	}
	// For any URL containing a host, extract the last non-empty path segment.
	if strings.Contains(q, "/") {
		parts := strings.Split(strings.TrimRight(q, "/"), "/")
		for i := len(parts) - 1; i >= 0; i-- {
			seg := strings.TrimSpace(parts[i])
			// Skip protocol/host segments (contain "." or ":")
			if seg != "" && !strings.Contains(seg, ".") && !strings.Contains(seg, ":") {
				return strings.ToLower(seg)
			}
		}
		return ""
	}
	// Treat kebab-case input (no spaces) as a slug.
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
	ResolutionSource string    // URL or text describing how the market resolves
	Description      string    // full market description / rules
	Outcome          string    // "Yes" / "No" / "" (populated for closed markets)
	Closed           bool
	EndDate          time.Time
	CreatedAt        time.Time
}

// gammaMarketResponse is the raw Gamma API response for a single market.
// Note: clobTokenIds, outcomes, and outcomePrices come back as JSON-encoded strings, not native arrays.
type gammaMarketResponse struct {
	Slug             string `json:"slug"`
	Question         string `json:"question"`
	ConditionID      string `json:"conditionId"`
	ClobTokenIDs     string `json:"clobTokenIds"`
	ResolutionSource string `json:"resolutionSource"`
	Description      string `json:"description"`
	Outcome          string `json:"outcome"` // set on some responses; often empty
	Outcomes         string `json:"outcomes"`
	OutcomePrices    string `json:"outcomePrices"`
	Closed           bool   `json:"closed"`
	EndDateISO       string `json:"endDateIso"`
	EndDate          string `json:"endDate"` // alternate to endDateIso on some payloads
	CreatedAt        string `json:"createdAt"`
}

// resolveOutcomeFromGamma returns the winning outcome label for a resolved market.
// Gamma encodes outcomes and prices as JSON strings, e.g. outcomes="[\"Yes\",\"No\"]"
// and outcomePrices="[\"1\",\"0\"]" for a YES resolution. If outcomePrices are all
// zero or parsing fails, returns "".
//
// For non-closed markets we only return m.Outcome when the API sets it; we never
// infer resolution from outcomePrices (those are live implied probabilities).
func resolveOutcomeFromGamma(m gammaMarketResponse) string {
	if s := strings.TrimSpace(m.Outcome); s != "" {
		return s
	}
	if !m.Closed {
		return ""
	}
	var prices []string
	if err := json.Unmarshal([]byte(m.OutcomePrices), &prices); err != nil || len(prices) == 0 {
		return ""
	}
	var labels []string
	if err := json.Unmarshal([]byte(m.Outcomes), &labels); err != nil || len(labels) == 0 {
		return ""
	}
	if len(prices) != len(labels) {
		return ""
	}
	// 1) Settled binary / n-ary: exactly one price is 1 (or ~1)
	for i, p := range prices {
		p = strings.TrimSpace(p)
		if p == "1" {
			return labels[i]
		}
		if f, err := strconv.ParseFloat(p, 64); err == nil && f >= 0.999 {
			return labels[i]
		}
	}
	// 2) All zeros / empty → unknown resolution in API
	allTiny := true
	for _, p := range prices {
		f, err := strconv.ParseFloat(strings.TrimSpace(p), 64)
		if err == nil && f > 1e-6 {
			allTiny = false
			break
		}
	}
	if allTiny {
		return ""
	}
	// 3) Closed market with only implied probs: pick argmax (last known prices)
	bestIdx := -1
	bestVal := -1.0
	for i, p := range prices {
		f, err := strconv.ParseFloat(strings.TrimSpace(p), 64)
		if err != nil {
			continue
		}
		if f > bestVal {
			bestVal = f
			bestIdx = i
		}
	}
	if bestIdx < 0 || bestVal <= 0 {
		return ""
	}
	return labels[bestIdx]
}

// isBinaryYesNo returns true only for markets with exactly ["Yes","No"] outcomes.
func isBinaryYesNo(outcomesJSON string) bool {
	var outcomes []string
	if err := json.Unmarshal([]byte(outcomesJSON), &outcomes); err != nil || len(outcomes) != 2 {
		return false
	}
	a, b := strings.ToLower(strings.TrimSpace(outcomes[0])), strings.ToLower(strings.TrimSpace(outcomes[1]))
	return (a == "yes" && b == "no") || (a == "no" && b == "yes")
}

// toResolvedMarket converts a gammaMarketResponse to a ResolvedMarket.
// Returns nil if the response has malformed token IDs or is not a binary Yes/No market.
func toResolvedMarket(m gammaMarketResponse) *ResolvedMarket {
	var tokenIDs []string
	if err := json.Unmarshal([]byte(m.ClobTokenIDs), &tokenIDs); err != nil || len(tokenIDs) != 2 {
		return nil
	}
	if !isBinaryYesNo(m.Outcomes) {
		return nil
	}
	rm := &ResolvedMarket{
		Slug:             m.Slug,
		Question:         m.Question,
		ConditionID:      m.ConditionID,
		ClobTokenIDs:     [2]string{tokenIDs[0], tokenIDs[1]},
		ResolutionSource: m.ResolutionSource,
		Description:      m.Description,
		Outcome:          resolveOutcomeFromGamma(m),
		Closed:           m.Closed,
	}
	dateStr := m.EndDateISO
	if dateStr == "" {
		dateStr = m.EndDate
	}
	if t, err := time.Parse(time.RFC3339, dateStr); err == nil {
		rm.EndDate = t
	}
	if t, err := time.Parse(time.RFC3339, m.CreatedAt); err == nil {
		rm.CreatedAt = t
	}
	return rm
}

// ResolveEvent looks up a Polymarket event by slug and returns all binary YES/NO
// markets contained within it. Events are top-level groupings (e.g. "fed-decision-in-october")
// that contain one or more individual markets.
func (c *Client) ResolveEvent(ctx context.Context, eventSlug string) ([]ResolvedMarket, error) {
	type eventResponse struct {
		Markets []gammaMarketResponse `json:"markets"`
	}

	// Normalize: extract slug from URLs or kebab-case queries.
	if normalized := queryToSlug(eventSlug); normalized != "" {
		eventSlug = normalized
	}

	url := fmt.Sprintf("%s/events?slug=%s", c.cfg.GammaBaseURL, eventSlug)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("gamma events request failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("gamma events returned status %d", resp.StatusCode)
	}

	var events []eventResponse
	if err := json.NewDecoder(resp.Body).Decode(&events); err != nil {
		return nil, fmt.Errorf("failed to decode events response: %w", err)
	}
	if len(events) == 0 {
		return nil, fmt.Errorf("no event found for slug: %s", eventSlug)
	}

	var out []ResolvedMarket
	for _, m := range events[0].Markets {
		// Accept any market with exactly 2 CLOB token IDs — not restricted to Yes/No labels.
		var tokenIDs []string
		if err := json.Unmarshal([]byte(m.ClobTokenIDs), &tokenIDs); err != nil || len(tokenIDs) != 2 {
			continue
		}
		rm := toResolvedMarket(m)
		if rm == nil {
			// toResolvedMarket rejects non-Yes/No outcomes; build manually for other outcome labels.
			rm = buildResolvedMarket(m, tokenIDs)
		}
		out = append(out, *rm)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no markets found in event %q", eventSlug)
	}
	return out, nil
}

// buildResolvedMarket builds a ResolvedMarket from a raw response and pre-parsed token IDs,
// skipping the binary Yes/No outcomes check. Used for event sub-markets with other outcome labels.
func buildResolvedMarket(m gammaMarketResponse, tokenIDs []string) *ResolvedMarket {
	rm := &ResolvedMarket{
		Slug:             m.Slug,
		Question:         m.Question,
		ConditionID:      m.ConditionID,
		ClobTokenIDs:     [2]string{tokenIDs[0], tokenIDs[1]},
		ResolutionSource: m.ResolutionSource,
		Description:      m.Description,
		Outcome:          resolveOutcomeFromGamma(m),
		Closed:           m.Closed,
	}
	dateStr := m.EndDateISO
	if dateStr == "" {
		dateStr = m.EndDate
	}
	if t, err := time.Parse(time.RFC3339, dateStr); err == nil {
		rm.EndDate = t
	}
	if t, err := time.Parse(time.RFC3339, m.CreatedAt); err == nil {
		rm.CreatedAt = t
	}
	return rm
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

	rm := toResolvedMarket(markets[0])
	if rm == nil {
		return nil, fmt.Errorf("market %s has malformed token IDs", slug)
	}
	return rm, nil
}
