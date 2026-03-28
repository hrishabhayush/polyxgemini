package polymarket_test

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/hrishabhayush/polyxgemini/internal/config"
	polymarket "github.com/hrishabhayush/polyxgemini/internal/exchange/polymarket"
)

func TestGammaFetchMarkets(t *testing.T) {
	cfg := config.PolymarketConfig{
		GammaBaseURL: "https://gamma-api.polymarket.com",
		DataBaseURL:  "https://data-api.polymarket.com",
		CLOBBaseURL:  "https://clob.polymarket.com",
	}
	client := polymarket.NewClient(cfg)
	ctx := context.Background()

	// 1. Discover active markets with at least $10k volume
	markets, err := client.DiscoverMarkets(ctx, true, 10_000)
	if err != nil {
		t.Fatalf("DiscoverMarkets failed: %v", err)
	}

	printBox(t, "GAMMA API — MARKET DISCOVERY", "", []row{
		{"Markets returned", fmt.Sprintf("%d", len(markets)), "active markets with volume ≥ $10k"},
	})

	if len(markets) == 0 {
		t.Log("  (no markets returned — check Gamma API availability)")
		return
	}

	// Show top 5 from the discovery list
	fmt.Printf("\n  %-15s  %-20s  %s\n", "CONDITION ID", "SLUG", "QUESTION")
	fmt.Printf("  %s\n", strings.Repeat("─", 90))
	for i, m := range markets {
		if i >= 5 {
			break
		}
		question := m.Question
		if len(question) > 55 {
			question = question[:52] + "..."
		}
		slug := m.Slug
		if len(slug) > 18 {
			slug = slug[:18]
		}
		condID := m.ConditionID
		if len(condID) > 13 {
			condID = condID[:6] + "..." + condID[len(condID)-4:]
		}
		fmt.Printf("  %-15s  %-20s  %s\n", condID, slug, question)
	}
	fmt.Println()

	// 2. Single market lookup pinned to the standard test market (slug-based — reliable)
	detailed, err := client.ResolveMarket(ctx, testSlug())
	if err != nil {
		t.Fatalf("ResolveMarket failed: %v", err)
	}

	printBox(t, "GAMMA API — SINGLE MARKET LOOKUP", detailed.ConditionID, []row{
		{"Question", truncate(detailed.Question, 60), ""},
		{"Slug", detailed.Slug, ""},
		{"YES token", truncate(detailed.ClobTokenIDs[0], 20), "Utah State Aggies"},
		{"NO token", truncate(detailed.ClobTokenIDs[1], 20), "Arizona Wildcats"},
	})
}

// ── pretty-print helpers (shared across all tests in this package) ────────────

type row struct {
	label string
	value string
	note  string
}

func printBox(t *testing.T, title, subtitle string, rows []row) {
	t.Helper()
	width := 62
	fmt.Printf("\n%s\n", strings.Repeat("═", width))
	fmt.Printf("  %s\n", title)
	if subtitle != "" {
		fmt.Printf("  Market: %s\n", subtitle)
	}
	fmt.Printf("%s\n", strings.Repeat("═", width))
	for _, r := range rows {
		if r.note != "" {
			fmt.Printf("  %-24s %-16s %s\n", r.label, r.value, r.note)
		} else {
			fmt.Printf("  %-24s %s\n", r.label, r.value)
		}
	}
	fmt.Printf("%s\n", strings.Repeat("═", width))
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n-3] + "..."
}
