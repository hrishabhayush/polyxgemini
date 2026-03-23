package polymarket_test

import (
	"context"
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/hrishabhayush/polyxgemini/internal/config"
	polymarket "github.com/hrishabhayush/polyxgemini/internal/exchange/polymarket"
)

// testTokenID returns the YES token ID to use for price history tests.
// Falls back to: Utah State Aggies vs. Arizona Wildcats (NCAA March Madness 2026) — Utah State (YES) token.
func testTokenID() string {
	if v := os.Getenv("POLYMARKET_TOKEN_ID"); v != "" {
		return v
	}
	return "97375804516138919402368296257548460708782641123645973538856878198805504642693"
}

func TestDataPrices(t *testing.T) {
	cfg := config.PolymarketConfig{
		GammaBaseURL: "https://gamma-api.polymarket.com",
		DataBaseURL:  "https://data-api.polymarket.com",
		CLOBBaseURL:  "https://clob.polymarket.com",
	}
	client := polymarket.NewClient(cfg)
	ctx := context.Background()
	tokenID := testTokenID()

	// Fetch full price history at max resolution (fidelity=100 data points)
	points, err := client.FetchPriceHistory(ctx, tokenID, "max", 100)
	if err != nil {
		t.Fatalf("FetchPriceHistory failed: %v", err)
	}

	if len(points) == 0 {
		t.Log("No price history returned — market may be new or token ID may be wrong.")
		t.Log("Set POLYMARKET_TOKEN_ID env var to override.")
		return
	}

	prices := polymarket.Prices(points)

	// Compute metrics
	hurst := polymarket.HurstExponent(prices)
	currentVol, historicalVol := polymarket.GARCHVolatility(prices)
	jumpResult := polymarket.JumpPersistence(prices, 0.03) // 3-cent threshold

	// Interpret Hurst
	hurstNote := "random walk (H ≈ 0.5)"
	switch {
	case hurst < 0.4:
		hurstNote = "MEAN-REVERTING — price tends to snap back"
	case hurst < 0.45:
		hurstNote = "slightly mean-reverting"
	case hurst > 0.6:
		hurstNote = "TRENDING — price moves tend to persist"
	case hurst > 0.55:
		hurstNote = "slightly trending"
	}

	// Interpret vol ratio
	volRatio := 0.0
	volNote := "no history"
	if historicalVol > 0 {
		volRatio = currentVol / historicalVol
		switch {
		case volRatio > 2.0:
			volNote = "HIGH — elevated vol regime"
		case volRatio > 1.5:
			volNote = "elevated"
		case volRatio < 0.5:
			volNote = "suppressed vol regime"
		default:
			volNote = "normal vol regime"
		}
	}

	jumpNote := map[string]string{
		"sustained": "price moves tend to follow through",
		"reversed":  "price moves tend to snap back (mean-revert)",
		"no_jumps":  "no large moves (>3¢) detected in window",
	}[jumpResult]

	printBox(t, "POLYMARKET — PRICE HISTORY METRICS", tokenID[:20]+"...", []row{
		{"Data points", fmt.Sprintf("%d", len(points)), "full history, fidelity=100"},
		{"Price range", fmt.Sprintf("%.4f – %.4f", minF(prices), maxF(prices)), ""},
		{"Latest price", fmt.Sprintf("%.4f", prices[len(prices)-1]), ""},
		{"─────────────────", "", ""},
		{"Hurst exponent (H)", fmt.Sprintf("%.4f", hurst), hurstNote},
		{"Current vol (7-bar)", fmt.Sprintf("%.6f", math.Sqrt(currentVol)), "log-return std dev"},
		{"Historical vol", fmt.Sprintf("%.6f", math.Sqrt(historicalVol)), "log-return std dev"},
		{"Vol ratio (cur/hist)", fmt.Sprintf("%.3f", volRatio), volNote},
		{"Jump persistence", jumpResult, jumpNote},
	})

	// Recent price points preview
	fmt.Printf("\n  %-22s  %s\n", "TIME (UTC)", "PRICE")
	fmt.Printf("  %s\n", "───────────────────────────────────")
	shown := points
	if len(shown) > 5 {
		shown = shown[len(shown)-5:] // most recent 5
	}
	for _, p := range shown {
		fmt.Printf("  %-22s  %.4f\n", p.Time.Format("2006-01-02 15:04:05"), p.Price)
	}
	fmt.Println()
}

func minF(xs []float64) float64 {
	m := xs[0]
	for _, x := range xs[1:] {
		if x < m {
			m = x
		}
	}
	return m
}

func maxF(xs []float64) float64 {
	m := xs[0]
	for _, x := range xs[1:] {
		if x > m {
			m = x
		}
	}
	return m
}
