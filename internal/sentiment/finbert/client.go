package finbert

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/hrishabhayush/polyxgemini/internal/config"
	"github.com/hrishabhayush/polyxgemini/internal/sentiment"
)

// ErrServerUnreachable is returned when the FinBERT server is not running.
var ErrServerUnreachable = errors.New("finbert: server unreachable")

type scoreRequest struct {
	Texts []string `json:"texts"`
}

type scoreResponse struct {
	Results []struct {
		Label string  `json:"label"`
		Score float64 `json:"score"`
	} `json:"results"`
}

type Client struct {
	cfg        config.SentimentConfig
	httpClient *http.Client
}

func NewClient(cfg config.SentimentConfig) *Client {
	return &Client{
		cfg:        cfg,
		httpClient: &http.Client{Timeout: 30 * time.Second},
	}
}

// Ping checks if the FinBERT server is reachable and the model is loaded.
func (c *Client) Ping(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.cfg.FinBERTServerURL+"/health", nil)
	if err != nil {
		return fmt.Errorf("finbert: ping build request: %w", err)
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("%w: %v", ErrServerUnreachable, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("finbert: health check status %d", resp.StatusCode)
	}
	return nil
}

// Score sends a batch of texts to the local FinBERT server and returns one
// ArticleScore per input. The Source field must be set by the caller.
func (c *Client) Score(ctx context.Context, texts []string) ([]sentiment.ArticleScore, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	body, err := json.Marshal(scoreRequest{Texts: texts})
	if err != nil {
		return nil, fmt.Errorf("finbert: marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.cfg.FinBERTServerURL+"/score", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("finbert: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrServerUnreachable, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("finbert: unexpected status %d", resp.StatusCode)
	}

	var sr scoreResponse
	if err := json.NewDecoder(resp.Body).Decode(&sr); err != nil {
		return nil, fmt.Errorf("finbert: decode response: %w", err)
	}

	now := time.Now()
	scores := make([]sentiment.ArticleScore, len(sr.Results))
	for i, r := range sr.Results {
		scores[i] = sentiment.ArticleScore{
			Text:      texts[i],
			Label:     sentiment.Label(r.Label),
			Score:     r.Score,
			FetchedAt: now,
		}
	}
	return scores, nil
}
