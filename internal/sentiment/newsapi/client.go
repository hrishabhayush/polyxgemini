package newsapi

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"time"

	"github.com/hrishabhayush/polyxgemini/internal/config"
)

// ErrNoAPIKey is returned when no NewsAPI key is configured.
var ErrNoAPIKey = errors.New("newsapi: api key not configured")

// Article is a normalized NewsAPI article.
type Article struct {
	Title       string    `json:"title"`
	Description string    `json:"description"`
	Content     string    `json:"content"`
	Source      string    `json:"source"`
	URL         string    `json:"url"`
	PublishedAt time.Time `json:"publishedAt"`
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

// FetchHeadlines queries /v2/everything for the given keywords.
// Returns at most maxResults articles sorted by publishedAt descending.
func (c *Client) FetchHeadlines(ctx context.Context, keywords string, maxResults int) ([]Article, error) {
	if c.cfg.NewsAPIKey == "" {
		return nil, ErrNoAPIKey
	}

	params := url.Values{}
	params.Set("q", keywords)
	params.Set("sortBy", "publishedAt")
	params.Set("language", "en")
	params.Set("pageSize", strconv.Itoa(maxResults))

	endpoint := c.cfg.NewsAPIBaseURL + "/everything?" + params.Encode()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("newsapi: build request: %w", err)
	}
	req.Header.Set("X-Api-Key", c.cfg.NewsAPIKey)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("newsapi: request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("newsapi: unexpected status %d", resp.StatusCode)
	}

	var body struct {
		Status   string `json:"status"`
		Articles []struct {
			Title       string `json:"title"`
			Description string `json:"description"`
			Content     string `json:"content"`
			URL         string `json:"url"`
			PublishedAt string `json:"publishedAt"`
			Source      struct {
				Name string `json:"name"`
			} `json:"source"`
		} `json:"articles"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return nil, fmt.Errorf("newsapi: decode response: %w", err)
	}

	articles := make([]Article, 0, len(body.Articles))
	for _, a := range body.Articles {
		t, _ := time.Parse(time.RFC3339, a.PublishedAt)
		articles = append(articles, Article{
			Title:       a.Title,
			Description: a.Description,
			Content:     a.Content,
			Source:      a.Source.Name,
			URL:         a.URL,
			PublishedAt: t,
		})
	}
	return articles, nil
}
