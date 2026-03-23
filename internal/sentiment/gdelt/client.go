package gdelt

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"time"

	"github.com/hrishabhayush/polyxgemini/internal/config"
)

// Article is a normalized GDELT article.
type Article struct {
	Title     string    `json:"title"`
	URL       string    `json:"url"`
	Domain    string    `json:"domain"`
	Language  string    `json:"language"`
	Published time.Time `json:"published"`
	Tone      float64   `json:"tone"`
}

type Client struct {
	cfg        config.SentimentConfig
	httpClient *http.Client
}

func NewClient(cfg config.SentimentConfig) *Client {
	return &Client{
		cfg:        cfg,
		httpClient: &http.Client{Timeout: 10 * time.Second},
	}
}

// FetchArticles calls the GDELT doc API in artlist mode.
// Returns at most maxResults articles ranked by HybridRel (relevance + recency)
// within the last 7 days. Pass keywords using GDELT boolean syntax, e.g.
// `trump AND impeach` or `"trump impeachment"` for tighter matching.
func (c *Client) FetchArticles(ctx context.Context, keywords string, maxResults int) ([]Article, error) {
	params := url.Values{}
	params.Set("query", keywords)
	params.Set("mode", "artlist")
	params.Set("format", "json")
	params.Set("maxrecords", strconv.Itoa(maxResults))
	params.Set("sort", "HybridRel") // relevance-aware ranking; DateDesc would push off-topic recency
	params.Set("timespan", "7d")    // limit to last 7 days to avoid stale noise

	endpoint := c.cfg.GDELTBaseURL + "?" + params.Encode()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("gdelt: build request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("gdelt: request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("gdelt: unexpected status %d", resp.StatusCode)
	}

	var body struct {
		Articles []struct {
			Title    string  `json:"title"`
			URL      string  `json:"url"`
			Domain   string  `json:"domain"`
			Language string  `json:"language"`
			Seendate string  `json:"seendate"`
			Tone     float64 `json:"tone"`
		} `json:"articles"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return nil, fmt.Errorf("gdelt: decode response: %w", err)
	}

	articles := make([]Article, 0, len(body.Articles))
	for _, a := range body.Articles {
		// GDELT seendate format: "20060102T150405Z"
		t, _ := time.Parse("20060102T150405Z", a.Seendate)
		articles = append(articles, Article{
			Title:     a.Title,
			URL:       a.URL,
			Domain:    a.Domain,
			Language:  a.Language,
			Published: t,
			Tone:      a.Tone,
		})
	}
	return articles, nil
}
