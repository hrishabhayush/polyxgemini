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

// WalletHHI computes the Herfindahl-Hirschman Index for trading volume concentration
// across wallets from a set of trades. This is a proxy for position concentration
// since Polymarket does not expose a public market-wide positions endpoint.
//
// HHI = Σ (wallet_volume_i / total_volume)²
// Range: 0.0 (fully distributed) → 1.0 (one wallet dominates all trading).
func WalletHHI(trades []Trade) float64 {
	walletVol := make(map[string]float64)
	var total float64
	for _, t := range trades {
		if t.ProxyWallet != "" {
			walletVol[t.ProxyWallet] += t.Size
		}
		total += t.Size
	}
	if total == 0 {
		return 0
	}
	var hhi float64
	for _, v := range walletVol {
		share := v / total
		hhi += share * share
	}
	return hhi
}

// Trade is a single executed trade on Polymarket.
// Side is inferred via Lee-Ready classification (not always provided by the API).
// ProxyWallet is the trader's wallet address, used for HHI concentration analysis.
type Trade struct {
	ID          string
	Price       float64
	Size        float64
	Timestamp   time.Time
	Side        string // "buy" | "sell" (Lee-Ready inferred)
	ProxyWallet string
}

// FetchTrades fetches executed trades for a market from the Data API.
// conditionID is the market's condition ID (from Gamma API).
// limit caps the number of trades returned; use 200 for metric calculations.
//
// Polling note: fetch every 100ms for live Kyle's lambda / VPIN updates.
func (c *Client) FetchTrades(ctx context.Context, conditionID string, limit int) ([]Trade, error) {
	params := url.Values{}
	params.Set("market", conditionID)
	params.Set("limit", strconv.Itoa(limit))

	endpoint := c.cfg.DataBaseURL + "/trades?" + params.Encode()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("data/trades: build request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("data/trades: request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("data/trades: unexpected status %d", resp.StatusCode)
	}

	var raw []struct {
		ID          string  `json:"id"`
		Price       float64 `json:"price"`
		Size        float64 `json:"size"`
		Timestamp   int64   `json:"timestamp"`   // unix seconds
		Side        string  `json:"side"`        // "BUY" | "SELL" | ""
		ProxyWallet string  `json:"proxyWallet"` // trader wallet address
	}
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return nil, fmt.Errorf("data/trades: decode: %w", err)
	}

	trades := make([]Trade, 0, len(raw))
	for i, r := range raw {
		// Lee-Ready classification: if API doesn't tag direction, infer from
		// price movement relative to previous trade.
		side := normalizeSide(r.Side)
		if side == "" && i > 0 {
			prev := raw[i-1].Price
			if r.Price > prev {
				side = "buy"
			} else if r.Price < prev {
				side = "sell"
			} else {
				side = "buy" // tie-break: classify as buy (conservative)
			}
		}
		if side == "" {
			side = "buy"
		}

		trades = append(trades, Trade{
			ID:          r.ID,
			Price:       r.Price,
			Size:        r.Size,
			Timestamp:   time.Unix(r.Timestamp, 0).UTC(),
			Side:        side,
			ProxyWallet: r.ProxyWallet,
		})
	}
	return trades, nil
}

func normalizeSide(s string) string {
	switch s {
	case "BUY", "buy":
		return "buy"
	case "SELL", "sell":
		return "sell"
	}
	return ""
}

// KylesLambda estimates the price impact coefficient λ from a slice of trades.
//
//	λ = cov(ΔP, net_flow) / var(net_flow)
//
// net_flow per trade = +size (buy) or −size (sell).
// ΔP = price change between consecutive trades.
// Higher λ means each unit of order flow moves prices more → more informed trading.
// Returns 0 if there are fewer than 2 trades.
func KylesLambda(trades []Trade) float64 {
	n := len(trades)
	if n < 2 {
		return 0
	}

	deltaP := make([]float64, n-1)
	netFlow := make([]float64, n-1)

	for i := 1; i < n; i++ {
		deltaP[i-1] = trades[i].Price - trades[i-1].Price
		if trades[i].Side == "buy" {
			netFlow[i-1] = trades[i].Size
		} else {
			netFlow[i-1] = -trades[i].Size
		}
	}

	return covariance(deltaP, netFlow) / variance(netFlow)
}

// VPIN (Volume-synchronized Probability of Informed Trading) estimates the
// fraction of order flow driven by informed traders.
//
// Trades are bucketed into equal-volume increments of bucketSize.
// Per bucket: VPIN_bucket = |buy_vol − sell_vol| / bucket_vol
// Returns the average across all complete buckets.
// Values > 0.5 signal elevated informed trading activity.
//
// bucketSize: set to ~total_volume / 50 for 50-bucket resolution.
func VPIN(trades []Trade, bucketSize float64) float64 {
	if len(trades) == 0 || bucketSize <= 0 {
		return 0
	}

	var buckets []float64
	var buyVol, sellVol, accumulated float64

	for _, t := range trades {
		remaining := t.Size
		for remaining > 0 {
			space := bucketSize - accumulated
			fill := math.Min(remaining, space)

			if t.Side == "buy" {
				buyVol += fill
			} else {
				sellVol += fill
			}
			accumulated += fill
			remaining -= fill

			if accumulated >= bucketSize {
				imbalance := math.Abs(buyVol-sellVol) / bucketSize
				buckets = append(buckets, imbalance)
				buyVol, sellVol, accumulated = 0, 0, 0
			}
		}
	}

	if len(buckets) == 0 {
		return 0
	}
	var sum float64
	for _, b := range buckets {
		sum += b
	}
	return sum / float64(len(buckets))
}

// LeeReadyStats returns the fraction of trades classified as buyer-initiated.
func LeeReadyStats(trades []Trade) float64 {
	if len(trades) == 0 {
		return 0
	}
	var buys int
	for _, t := range trades {
		if t.Side == "buy" {
			buys++
		}
	}
	return float64(buys) / float64(len(trades))
}

// ── math helpers ──────────────────────────────────────────────────────────────

func mean(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	var s float64
	for _, x := range xs {
		s += x
	}
	return s / float64(len(xs))
}

func variance(xs []float64) float64 {
	m := mean(xs)
	var s float64
	for _, x := range xs {
		d := x - m
		s += d * d
	}
	if len(xs) <= 1 {
		return 0
	}
	return s / float64(len(xs)-1)
}

func covariance(xs, ys []float64) float64 {
	if len(xs) != len(ys) || len(xs) <= 1 {
		return 0
	}
	mx, my := mean(xs), mean(ys)
	var s float64
	for i := range xs {
		s += (xs[i] - mx) * (ys[i] - my)
	}
	return s / float64(len(xs)-1)
}
