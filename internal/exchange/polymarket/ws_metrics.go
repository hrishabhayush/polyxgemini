package polymarket

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/url"
	"strconv"
	"time"
)

// defaultCLOBBase is the public Polymarket CLOB API (used when FetchWSMetrics has no Client).
const defaultCLOBBase = "https://clob.polymarket.com"

// WSMetrics is a single snapshot of top-of-book and depth-derived stats (from HTTP /book, not a long-lived WS).
type WSMetrics struct {
	MidPrice     float64
	BidAskSpread float64
	OBI          float64 // (totalBidSize - totalAskSize) / (totalBidSize + totalAskSize), or 0 if empty
	BestBid      float64
	BestAsk      float64
	TotalBidSize float64
	TotalAskSize float64
}

type bookSide []struct {
	Price json.RawMessage `json:"price"`
	Size  json.RawMessage `json:"size"`
}

type clobBookResponse struct {
	Bids bookSide `json:"bids"`
	Asks bookSide `json:"asks"`
}

func parseJSONFloat(raw json.RawMessage) (float64, bool) {
	if len(raw) == 0 {
		return 0, false
	}
	var f float64
	if err := json.Unmarshal(raw, &f); err == nil {
		return f, true
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		v, err := strconv.ParseFloat(s, 64)
		if err == nil {
			return v, true
		}
	}
	return 0, false
}

// FetchWSMetrics fetches one CLOB order book for the market's YES token and computes mid, spread, depth, and OBI.
// It uses the public GET /book endpoint (snapshot), not a streaming WebSocket session.
func FetchWSMetrics(m ResolvedMarket, timeout time.Duration) (*WSMetrics, error) {
	token := m.ClobTokenIDs[0]
	if token == "" {
		return nil, fmt.Errorf("polymarket: empty YES clob token id")
	}

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	u, err := url.Parse(defaultCLOBBase + "/book")
	if err != nil {
		return nil, err
	}
	q := u.Query()
	q.Set("token_id", token)
	u.RawQuery = q.Encode()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return nil, fmt.Errorf("polymarket: book request: %w", err)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("polymarket: book fetch: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("polymarket: book status %d", resp.StatusCode)
	}

	var body clobBookResponse
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return nil, fmt.Errorf("polymarket: book decode: %w", err)
	}

	var bestBid, bestAsk float64
	var haveBid, haveAsk bool
	var totalBid, totalAsk float64

	for _, b := range body.Bids {
		p, ok := parseJSONFloat(b.Price)
		if !ok || p <= 0 {
			continue
		}
		sz, _ := parseJSONFloat(b.Size)
		if sz < 0 {
			sz = 0
		}
		totalBid += sz
		if !haveBid || p > bestBid {
			bestBid, haveBid = p, true
		}
	}
	for _, a := range body.Asks {
		p, ok := parseJSONFloat(a.Price)
		if !ok || p <= 0 {
			continue
		}
		sz, _ := parseJSONFloat(a.Size)
		if sz < 0 {
			sz = 0
		}
		totalAsk += sz
		if !haveAsk || p < bestAsk {
			bestAsk, haveAsk = p, true
		}
	}

	out := &WSMetrics{
		BestBid:      bestBid,
		BestAsk:      bestAsk,
		TotalBidSize: totalBid,
		TotalAskSize: totalAsk,
	}
	if haveBid && haveAsk {
		out.MidPrice = (bestBid + bestAsk) / 2
		out.BidAskSpread = bestAsk - bestBid
		if out.BidAskSpread < 0 {
			out.BidAskSpread = 0
		}
	} else if haveAsk {
		out.BestBid = 0
		out.MidPrice = bestAsk
	} else if haveBid {
		out.BestAsk = 0
		out.MidPrice = bestBid
	}

	denom := totalBid + totalAsk
	if denom > 1e-12 {
		out.OBI = (totalBid - totalAsk) / denom
	}

	// Avoid NaN in edge cases
	if math.IsNaN(out.MidPrice) {
		out.MidPrice = 0
	}
	if math.IsNaN(out.OBI) {
		out.OBI = 0
	}

	return out, nil
}
