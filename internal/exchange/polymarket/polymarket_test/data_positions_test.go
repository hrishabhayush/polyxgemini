package polymarket_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/hrishabhayush/polyxgemini/internal/config"
	polymarket "github.com/hrishabhayush/polyxgemini/internal/exchange/polymarket"
)

func TestDataPositions(t *testing.T) {
	cfg := config.PolymarketConfig{
		GammaBaseURL: "https://gamma-api.polymarket.com",
		DataBaseURL:  "https://data-api.polymarket.com",
		CLOBBaseURL:  "https://clob.polymarket.com",
	}
	client := polymarket.NewClient(cfg)
	ctx := context.Background()
	conditionID := testConditionID()

	// HHI is derived from trades data (wallet trading-volume concentration proxy)
	// since Polymarket has no public market-wide positions endpoint.
	trades, err := client.FetchTrades(ctx, conditionID, 500)
	if err != nil {
		t.Fatalf("FetchTrades (for HHI) failed: %v", err)
	}

	if len(trades) == 0 {
		t.Log("No trades returned — market may be inactive or condition ID may be wrong.")
		t.Log("Set POLYMARKET_CONDITION_ID env var to override.")
		return
	}

	hhi, top5 := polymarket.WalletConcentration(trades, 5)

	// Total volume
	var total float64
	for _, tr := range trades {
		total += tr.Size
	}

	// Unique wallets
	walletSet := make(map[string]struct{})
	for _, tr := range trades {
		if tr.ProxyWallet != "" {
			walletSet[tr.ProxyWallet] = struct{}{}
		}
	}

	// HHI interpretation
	hhiNote := "distributed — trading spread across many wallets"
	if hhi > 0.25 {
		hhiNote = "HIGH — whale risk (one wallet dominates trading volume)"
	} else if hhi > 0.10 {
		hhiNote = "moderate concentration"
	}

	printBox(t, "POLYMARKET — WALLET CONCENTRATION (HHI PROXY)", conditionID, []row{
		{"Trades analyzed", fmt.Sprintf("%d", len(trades)), ""},
		{"Unique wallets", fmt.Sprintf("%d", len(walletSet)), ""},
		{"Total volume", fmt.Sprintf("$%.2f", total), ""},
		{"─────────────────", "", ""},
		{"HHI (volume-based)", fmt.Sprintf("%.4f", hhi), hhiNote},
		{"HHI scale", "0 = distributed, 1 = one whale", ""},
		{"Note", "proxy via trade vol — not position size", "multi-wallet traders appear fragmented"},
	})

	if len(top5) == 0 {
		fmt.Println("  (no wallet data)")
		return
	}

	// Top wallets
	fmt.Printf("\n  %-8s  %-44s  %-10s  %s\n", "RANK", "WALLET", "VOLUME", "SHARE")
	fmt.Printf("  %s\n", "────────────────────────────────────────────────────────────────────")
	for i, w := range top5 {
		wallet := w.Wallet
		if len(wallet) > 42 {
			wallet = wallet[:6] + "..." + wallet[len(wallet)-4:]
		}
		fmt.Printf("  %-8d  %-44s  $%-9.2f  %.1f%%\n", i+1, wallet, w.Volume, w.SharePct)
	}
	fmt.Println()
}
