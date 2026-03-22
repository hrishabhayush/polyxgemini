//go:build integration

package gdelt_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/hrishabhayush/polyxgemini/internal/config"
	"github.com/hrishabhayush/polyxgemini/internal/sentiment/gdelt"
)

func TestFetchArticles_Integration(t *testing.T) {
	cfg := config.SentimentConfig{
		GDELTBaseURL: "https://api.gdeltproject.org/api/v2/doc/doc",
	}
	client := gdelt.NewClient(cfg)

	articles, err := client.FetchArticles(context.Background(), "trump visit china", 5)
	if err != nil {
		t.Fatalf("FetchArticles error: %v", err)
	}
	if len(articles) == 0 {
		t.Fatal("expected at least one article, got none")
	}

	t.Logf("\n\n╔══════════════════════════════════════════════════════════════════╗")
	t.Logf("║  GDELT Results — query: \"trump visit china\"                      ║")
	t.Logf("╚══════════════════════════════════════════════════════════════════╝")
	for i, a := range articles {
		b, _ := json.MarshalIndent(a, "    ", "  ")
		t.Logf("\n  [%d] %s", i+1, b)
	}
	t.Logf("\n  Total articles returned: %d\n", len(articles))
}
