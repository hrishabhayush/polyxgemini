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

// PricePoint is a timestamped price observation from the CLOB prices-history endpoint.
// The CLOB API returns a simple time-series (t, p) — not OHLC candles.
type PricePoint struct {
	Time  time.Time
	Price float64
}

// FetchPriceHistory fetches time-series price history for a market from the CLOB API.
//
// tokenID is the YES (or NO) token ID for the market — obtained from DiscoverMarkets
// or ResolveMarket (ClobTokenIDs[0] = YES, ClobTokenIDs[1] = NO).
// interval: "max" returns the full history; use "1m", "1h", "1d" for windowed history.
// fidelity: approximate number of data points to return.
//
// Polling note: re-fetch every 5 minutes — Hurst/GARCH are slow-moving statistical properties.
func (c *Client) FetchPriceHistory(ctx context.Context, tokenID, interval string, fidelity int) ([]PricePoint, error) {
	params := url.Values{}
	params.Set("market", tokenID) // CLOB uses "market" for the token/asset ID
	params.Set("interval", interval)
	params.Set("fidelity", strconv.Itoa(fidelity))

	endpoint := c.cfg.CLOBBaseURL + "/prices-history?" + params.Encode()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("clob/prices-history: build request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("clob/prices-history: request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("clob/prices-history: unexpected status %d", resp.StatusCode)
	}

	// CLOB prices-history response: {"history": [{"t": <unix_sec>, "p": <price>}, ...]}
	var body struct {
		History []struct {
			T int64   `json:"t"` // unix timestamp (seconds)
			P float64 `json:"p"` // price (0–1)
		} `json:"history"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return nil, fmt.Errorf("clob/prices-history: decode: %w", err)
	}

	points := make([]PricePoint, 0, len(body.History))
	for _, h := range body.History {
		points = append(points, PricePoint{
			Time:  time.Unix(h.T, 0).UTC(),
			Price: h.P,
		})
	}
	return points, nil
}

// Prices extracts the price series from a price history slice.
func Prices(points []PricePoint) []float64 {
	ps := make([]float64, len(points))
	for i, p := range points {
		ps[i] = p.Price
	}
	return ps
}

// HurstExponent estimates the Hurst exponent via Rescaled Range (R/S) analysis
// on the log-returns of the provided closing price series.
//
// Interpretation:
//   - H < 0.5  → mean-reverting (price is noisy, trend fades)
//   - H ≈ 0.5  → random walk (Brownian motion)
//   - H > 0.5  → trending (price moves persist)
//
// Use H as a confidence multiplier: high H means a price move is more likely to continue.
// Returns 0.5 (random walk baseline) if fewer than 20 data points are provided.
func HurstExponent(closes []float64) float64 {
	n := len(closes)
	if n < 20 {
		return 0.5 // not enough data — return random-walk baseline
	}

	// Compute log-returns
	returns := make([]float64, n-1)
	for i := 1; i < n; i++ {
		if closes[i-1] <= 0 {
			returns[i-1] = 0
		} else {
			returns[i-1] = math.Log(closes[i] / closes[i-1])
		}
	}

	// R/S analysis across multiple sub-series lengths
	var logN, logRS []float64
	for subLen := 10; subLen <= len(returns)/2; subLen += max(1, len(returns)/10) {
		rs := rescaledRange(returns, subLen)
		if rs > 0 {
			logN = append(logN, math.Log(float64(subLen)))
			logRS = append(logRS, math.Log(rs))
		}
	}
	if len(logN) < 2 {
		return 0.5
	}

	// OLS slope of log(R/S) ~ log(n) → Hurst exponent
	return olsSlope(logN, logRS)
}

// rescaledRange computes the average R/S statistic over non-overlapping sub-series of length subLen.
func rescaledRange(xs []float64, subLen int) float64 {
	var total float64
	count := 0
	for start := 0; start+subLen <= len(xs); start += subLen {
		sub := xs[start : start+subLen]
		m := mean(sub)

		// Cumulative deviation series
		cum := make([]float64, subLen)
		cum[0] = sub[0] - m
		for i := 1; i < subLen; i++ {
			cum[i] = cum[i-1] + (sub[i] - m)
		}

		r := maxSlice(cum) - minSlice(cum)
		s := stddev(sub)
		if s > 0 {
			total += r / s
			count++
		}
	}
	if count == 0 {
		return 0
	}
	return total / float64(count)
}

// GARCHVolatility returns a simplified estimate of current and historical volatility
// using rolling variance on closing prices.
//
// current: rolling variance over the last 7 candles (short-term vol regime)
// historical: full-series variance
//
// Use current/historical ratio to detect vol expansion:
//   - ratio > 1.5 → elevated volatility regime
//   - ratio < 0.5 → suppressed volatility
func GARCHVolatility(closes []float64) (current, historical float64) {
	n := len(closes)
	if n < 8 {
		return 0, 0
	}

	// Log-returns
	rets := make([]float64, n-1)
	for i := 1; i < n; i++ {
		if closes[i-1] > 0 {
			rets[i-1] = math.Log(closes[i] / closes[i-1])
		}
	}

	historical = variance(rets)

	windowSize := 7
	if len(rets) >= windowSize {
		current = variance(rets[len(rets)-windowSize:])
	}
	return current, historical
}

// JumpPersistence detects large price moves and determines whether they sustained or reversed.
//
// threshold: absolute price change to qualify as a jump (e.g., 0.05 = 5 cents on a $1 market).
// lookforward: number of candles after the jump to assess persistence (default 3).
//
// Returns:
//   - "sustained"  → jump direction continued on average over the look-forward window
//   - "reversed"   → price moved back toward pre-jump level
//   - "no_jumps"   → no moves exceeded threshold
func JumpPersistence(closes []float64, threshold float64) string {
	const lookforward = 3
	n := len(closes)
	if n < 5 {
		return "no_jumps"
	}

	sustained, reversed := 0, 0
	for i := 1; i < n-lookforward; i++ {
		move := closes[i] - closes[i-1]
		if math.Abs(move) < threshold {
			continue
		}
		// Check if subsequent closes continued in the jump direction
		afterAvg := mean(closes[i+1 : i+1+lookforward])
		if math.Signbit(afterAvg-closes[i]) == math.Signbit(-move) {
			// afterAvg moved opposite to jump direction → reversal
			reversed++
		} else {
			sustained++
		}
	}

	if sustained+reversed == 0 {
		return "no_jumps"
	}
	if sustained >= reversed {
		return "sustained"
	}
	return "reversed"
}

// ── math helpers ──────────────────────────────────────────────────────────────

func stddev(xs []float64) float64 {
	return math.Sqrt(variance(xs))
}

func maxSlice(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	m := xs[0]
	for _, x := range xs[1:] {
		if x > m {
			m = x
		}
	}
	return m
}

func minSlice(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	m := xs[0]
	for _, x := range xs[1:] {
		if x < m {
			m = x
		}
	}
	return m
}

// olsSlope computes the OLS slope of y ~ x.
func olsSlope(xs, ys []float64) float64 {
	return covariance(xs, ys) / variance(xs)
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
