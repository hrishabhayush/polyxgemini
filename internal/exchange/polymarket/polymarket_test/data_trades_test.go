package polymarket_test

import (
	"context"
	"fmt"
	"os"
	"testing"

	"github.com/hrishabhayush/polyxgemini/internal/config"
	polymarket "github.com/hrishabhayush/polyxgemini/internal/exchange/polymarket"
)

// Default test market: Utah State Aggies vs. Arizona Wildcats (NCAA March Madness 2026)
const (
	defaultConditionID = "0xd7313599d51c113b0101ffb4405dd6a788744a96bb5e405c6317d4a11b4f3904"
	defaultSlug        = "cbb-utahst-arz-2026-03-22"
)

func testConditionID() string {
	if v := os.Getenv("POLYMARKET_CONDITION_ID"); v != "" {
		return v
	}
	return defaultConditionID
}

func testSlug() string {
	if v := os.Getenv("POLYMARKET_SLUG"); v != "" {
		return v
	}
	return defaultSlug
}

func TestDataTrades(t *testing.T) {
	cfg := config.PolymarketConfig{
		GammaBaseURL: "https://gamma-api.polymarket.com",
		DataBaseURL:  "https://data-api.polymarket.com",
		CLOBBaseURL:  "https://clob.polymarket.com",
	}
	client := polymarket.NewClient(cfg)
	ctx := context.Background()
	conditionID := testConditionID()

	trades, err := client.FetchTrades(ctx, conditionID, 200)
	if err != nil {
		t.Fatalf("FetchTrades failed: %v", err)
	}

	if len(trades) == 0 {
		t.Log("No trades returned — market may be inactive or condition ID may be wrong.")
		t.Log("Set POLYMARKET_CONDITION_ID env var to override.")
		return
	}

	// Compute metrics
	lambda := polymarket.KylesLambda(trades)
	totalVol := totalVolume(trades)
	bucketSize := totalVol / 50 // 50-bucket VPIN resolution
	vpin := polymarket.VPIN(trades, bucketSize)
	buyFrac := polymarket.LeeReadyStats(trades)
	hhi, _ := polymarket.WalletConcentration(trades, 0)

	// Interpret VPIN
	vpinNote := "normal — flow looks uninformed"
	if vpin > 0.7 {
		vpinNote = "HIGH — strong informed trading signal"
	} else if vpin > 0.5 {
		vpinNote = "elevated — possible informed activity"
	}

	// Interpret Kyle's lambda
	lambdaNote := "low price impact per unit flow"
	if lambda > 0.01 {
		lambdaNote = "HIGH — informed flow moving prices"
	} else if lambda > 0.005 {
		lambdaNote = "moderate price impact"
	}

	// Interpret HHI
	hhiNote := "distributed market"
	if hhi > 0.25 {
		hhiNote = "HIGH — whale dominates trading"
	} else if hhi > 0.10 {
		hhiNote = "moderate concentration"
	}

	printBox(t, "POLYMARKET — TRADES METRICS", conditionID, []row{
		{"Trades fetched", fmt.Sprintf("%d", len(trades)), ""},
		{"Total volume", fmt.Sprintf("$%.2f", totalVol), ""},
		{"─────────────────", "", ""},
		{"Kyle's lambda (λ)", fmt.Sprintf("%.6f", lambda), lambdaNote},
		{"VPIN", fmt.Sprintf("%.4f", vpin), vpinNote},
		{"Lee-Ready buy %", fmt.Sprintf("%.1f%%", buyFrac*100), "fraction of buyer-initiated trades"},
		{"Wallet HHI", fmt.Sprintf("%.4f", hhi), hhiNote},
	})

	// Recent trades preview
	fmt.Printf("\n  %-20s  %-8s  %-10s  %s\n", "TIME (UTC)", "SIDE", "PRICE", "SIZE")
	fmt.Printf("  %s\n", "─────────────────────────────────────────────────")
	shown := trades
	if len(shown) > 5 {
		shown = shown[:5]
	}
	for _, tr := range shown {
		fmt.Printf("  %-20s  %-8s  %-10.4f  %.2f\n",
			tr.Timestamp.Format("2006-01-02 15:04:05"),
			tr.Side, tr.Price, tr.Size)
	}
	fmt.Println()
}

func totalVolume(trades []polymarket.Trade) float64 {
	var v float64
	for _, t := range trades {
		v += t.Size
	}
	return v
}
