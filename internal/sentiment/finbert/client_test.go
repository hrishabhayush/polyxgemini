//go:build integration

package finbert_test

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/hrishabhayush/polyxgemini/internal/config"
	"github.com/hrishabhayush/polyxgemini/internal/sentiment/finbert"
)

func TestScore_Integration(t *testing.T) {
	cfg := config.SentimentConfig{
		FinBERTServerURL: "http://localhost:8765",
	}
	client := finbert.NewClient(cfg)

	if err := client.Ping(context.Background()); err != nil {
		if errors.Is(err, finbert.ErrServerUnreachable) {
			t.Skipf("FinBERT server not reachable — run: uvicorn server:app --port 8765")
		}
		t.Fatalf("Ping error: %v", err)
	}

	texts := []string{
		"Trump announces historic trade deal with China, markets surge",
		"Markets crash as tariff war with China escalates sharply",
		"No significant developments reported in US-China talks today",
		"China and US agree to resume high-level diplomatic negotiations",
		"Analysts warn of recession risk if trade tensions continue",
	}

	scores, err := client.Score(context.Background(), texts)
	if err != nil {
		t.Fatalf("Score error: %v", err)
	}
	if len(scores) != len(texts) {
		t.Fatalf("expected %d scores, got %d", len(texts), len(scores))
	}

	t.Logf("\n\n╔══════════════════════════════════════════════════════════════════╗")
	t.Logf("║  FinBERT Sentiment Scores                                        ║")
	t.Logf("╚══════════════════════════════════════════════════════════════════╝")
	t.Logf("  %-10s  %-7s  %s", "LABEL", "CONF%", "TEXT")
	t.Logf("  %s", "──────────────────────────────────────────────────────────────")

	counts := map[string]int{"positive": 0, "negative": 0, "neutral": 0}
	for _, s := range scores {
		marker := " "
		switch s.Label {
		case "positive":
			marker = "+"
		case "negative":
			marker = "-"
		}
		text := s.Text
		if len(text) > 55 {
			text = text[:52] + "..."
		}
		t.Logf("  %s %-10s  %5.1f%%  %s", marker, string(s.Label), s.Score*100, text)
		counts[string(s.Label)]++
	}

	t.Logf("\n  Summary: +%d positive  -%d negative  ~%d neutral",
		counts["positive"], counts["negative"], counts["neutral"])
	t.Logf("  %s\n", fmt.Sprintf("Dominant: %s", dominant(counts)))
}

func dominant(counts map[string]int) string {
	best, bestN := "neutral", 0
	for label, n := range counts {
		if n > bestN {
			best, bestN = label, n
		}
	}
	return best
}
