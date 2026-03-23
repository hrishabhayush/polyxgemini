//go:build integration

package reddit_test

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/hrishabhayush/polyxgemini/internal/config"
	"github.com/hrishabhayush/polyxgemini/internal/sentiment/reddit"
)

func TestSearch_Integration(t *testing.T) {
	cfg := config.SentimentConfig{
		RedditBaseURL:    "https://www.reddit.com",
		RedditUserAgent:  "sentimentbot/1.0",
		RedditSubreddits: []string{"politics", "worldnews"},
	}
	client := reddit.NewClient(cfg)

	posts, err := client.Search(context.Background(), "trump visit china", 6)
	if err != nil {
		t.Fatalf("Search error: %v", err)
	}
	if len(posts) == 0 {
		t.Fatal("expected at least one post, got none")
	}

	t.Logf("\n\n╔══════════════════════════════════════════════════════════════════╗")
	t.Logf("║  Reddit Results — query: \"trump visit china\"                     ║")
	t.Logf("╚══════════════════════════════════════════════════════════════════╝")
	t.Logf("  %-12s  %-6s  %-8s  %s", "SUBREDDIT", "SCORE", "COMMENTS", "TITLE")
	t.Logf("  %s", fmt.Sprintf("%s", "──────────────────────────────────────────────────────────────"))
	for i, p := range posts {
		title := p.Title
		if len(title) > 60 {
			title = title[:57] + "..."
		}
		t.Logf("  %-12s  %-6d  %-8d  %s", p.Subreddit, p.Score, p.NumComments, title)
		if p.Selftext != "" {
			snippet := p.Selftext
			if len(snippet) > 120 {
				snippet = snippet[:117] + "..."
			}
			t.Logf("             └─ %s", snippet)
		}
		t.Logf("             Posted: %s | URL: %s", p.CreatedAt.Format(time.RFC1123), p.URL)
		if i < len(posts)-1 {
			t.Logf("")
		}
	}
	t.Logf("\n  Total posts returned: %d\n", len(posts))
}
