package gamma_test

import (
	"context"
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/hrishabhayush/polyxgemini/internal/config"
	"github.com/hrishabhayush/polyxgemini/internal/exchange/polymarket"
	"github.com/hrishabhayush/polyxgemini/internal/sentiment/gamma"
)

const defaultMarket = "will-israel-launch-a-major-ground-offensive-in-lebanon-by-march-31"

// TestFetchGammaComments is a live integration test that resolves a market,
// fetches its Gamma comments, filters them, and prints a summary.
func TestFetchGammaComments(t *testing.T) {
	marketName := defaultMarket
	if v := os.Getenv("MARKET_NAME"); v != "" {
		marketName = v
	}

	polyCfg := config.PolymarketConfig{
		CLOBBaseURL:  "https://clob.polymarket.com",
		GammaBaseURL: "https://gamma-api.polymarket.com",
		DataBaseURL:  "https://data-api.polymarket.com",
	}

	ctx := context.Background()
	polyClient := polymarket.NewClient(polyCfg)

	market, err := polyClient.FindMarket(ctx, marketName)
	if err != nil {
		t.Fatalf("FindMarket failed: %v", err)
	}

	t.Logf("Market: %s (Gamma ID: %d)", market.Question, market.GammaID)
	if market.GammaID == 0 {
		t.Skip("Market has no Gamma ID — cannot fetch comments")
	}

	client := gamma.NewClient(polyCfg.GammaBaseURL)
	comments, err := client.FetchComments(ctx, market.GammaID, 30)
	if err != nil {
		t.Fatalf("FetchComments failed: %v", err)
	}

	filtered := gamma.FilterRelevant(comments, 10)

	width := 56
	sep := strings.Repeat("═", width)
	thin := strings.Repeat("─", width)

	fmt.Printf("\n%s\n", sep)
	fmt.Printf("  GAMMA COMMENTS: %s\n", market.Question)
	fmt.Printf("%s\n", sep)
	fmt.Printf("  %-24s %d\n", "Total fetched", len(comments))
	fmt.Printf("  %-24s %d\n", "After filtering", len(filtered))
	fmt.Printf("%s\n", thin)

	for i, c := range filtered {
		if i >= 5 {
			fmt.Printf("  ... and %d more\n", len(filtered)-5)
			break
		}
		body := c.Body
		if len(body) > 80 {
			body = body[:80] + "..."
		}
		posTag := ""
		if c.PositionSize > 0 {
			posTag = fmt.Sprintf(" [pos=%.0f, w=%.2f]", c.PositionSize, gamma.PositionWeight(c.PositionSize, 500))
		}
		fmt.Printf("  %d. %s%s\n", i+1, body, posTag)
	}
	fmt.Printf("%s\n\n", sep)
}

// TestFilterRelevant verifies comment pre-filtering logic.
func TestFilterRelevant(t *testing.T) {
	comments := []gamma.Comment{
		{Body: "I think YES is the right call here, strong momentum"},
		{Body: "lol"},
		{Body: ""},
		{Body: "[removed]"},
		{Body: "[deleted]"},
		{Body: "🚀🚀🚀"},
		{Body: "Short but ok"},
		{Body: "   "},
		{Body: "This market is undervalued based on recent polling data"},
	}

	filtered := gamma.FilterRelevant(comments, 10)

	sep := strings.Repeat("─", 56)
	fmt.Printf("\n%s\n", sep)
	fmt.Printf("  FILTER RELEVANT SUMMARY\n")
	fmt.Printf("%s\n", sep)
	fmt.Printf("  Input comments:    %d\n", len(comments))
	fmt.Printf("  Kept comments:     %d\n", len(filtered))
	for i, c := range filtered {
		fmt.Printf("  %d) %q\n", i+1, c.Body)
	}
	fmt.Printf("%s\n\n", sep)

	if len(filtered) != 3 {
		t.Errorf("expected 3 comments after filtering, got %d", len(filtered))
		for i, c := range filtered {
			t.Logf("  [%d] %q", i, c.Body)
		}
	}
}

// TestPositionWeight verifies the bounded concave weighting function.
func TestPositionWeight(t *testing.T) {
	tests := []struct {
		name     string
		size     float64
		scale    float64
		wantMin  float64
		wantMax  float64
	}{
		{"zero position", 0, 500, 1.0, 1.0},
		{"negative position", -100, 500, 1.0, 1.0},
		{"small position", 50, 500, 1.0, 1.1},
		{"medium position", 500, 500, 1.2, 1.4},
		{"large position", 5000, 500, 1.45, 1.51},
		{"huge position", 1e6, 500, 1.49, 1.51},
		{"zero scale", 100, 0, 1.0, 1.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := gamma.PositionWeight(tt.size, tt.scale)
			fmt.Printf("  %-16s size=%8.0f scale=%5.0f -> weight=%.4f\n",
				tt.name, tt.size, tt.scale, got)
			if got < tt.wantMin || got > tt.wantMax {
				t.Errorf("PositionWeight(%.0f, %.0f) = %.4f, want [%.2f, %.2f]",
					tt.size, tt.scale, got, tt.wantMin, tt.wantMax)
			}
		})
	}

	fmt.Println()

	// Verify concavity: weight(200) - weight(100) > weight(1000) - weight(900)
	w100 := gamma.PositionWeight(100, 500)
	w200 := gamma.PositionWeight(200, 500)
	w900 := gamma.PositionWeight(900, 500)
	w1000 := gamma.PositionWeight(1000, 500)
	if (w200 - w100) <= (w1000 - w900) {
		t.Errorf("weight function not concave: delta(100→200)=%f, delta(900→1000)=%f",
			w200-w100, w1000-w900)
	}

	// Verify hard cap: never exceeds 2.0
	if got := gamma.PositionWeight(1e12, 1); got > 2.0 {
		t.Errorf("PositionWeight exceeded hard cap: %f", got)
	}
}
