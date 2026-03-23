//go:build integration

package newsapi_test

import (
	"context"
	"encoding/json"
	"os"
	"testing"

	"github.com/hrishabhayush/polyxgemini/internal/config"
	"github.com/hrishabhayush/polyxgemini/internal/sentiment/newsapi"
)

func TestFetchHeadlines_Integration(t *testing.T) {
	key := os.Getenv("NEWSAPI_KEY")
	if key == "" {
		t.Skip("NEWSAPI_KEY not set — skipping integration test")
	}

	cfg := config.SentimentConfig{
		NewsAPIKey:     key,
		NewsAPIBaseURL: "https://newsapi.org/v2",
	}
	client := newsapi.NewClient(cfg)

	articles, err := client.FetchHeadlines(context.Background(), "trump visit china", 5)
	if err != nil {
		t.Fatalf("FetchHeadlines error: %v", err)
	}
	if len(articles) == 0 {
		t.Fatal("expected at least one article, got none")
	}

	t.Logf("\n\n╔══════════════════════════════════════════════════════════════════╗")
	t.Logf("║  NewsAPI Results — query: \"trump visit china\"                    ║")
	t.Logf("╚══════════════════════════════════════════════════════════════════╝")
	for i, a := range articles {
		b, _ := json.MarshalIndent(a, "    ", "  ")
		t.Logf("\n  [%d] %s", i+1, b)
	}
	t.Logf("\n  Total articles returned: %d\n", len(articles))
}
