package reddit

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

// Post is a normalized Reddit post.
type Post struct {
	Title       string    `json:"title"`
	Selftext    string    `json:"selftext"`
	Score       int       `json:"score"`
	Subreddit   string    `json:"subreddit"`
	URL         string    `json:"url"`
	CreatedAt   time.Time `json:"createdAt"`
	NumComments int       `json:"numComments"`
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

// redditResponse is the raw JSON structure returned by Reddit's search endpoint.
type redditResponse struct {
	Data struct {
		Children []struct {
			Data struct {
				Title      string  `json:"title"`
				Selftext   string  `json:"selftext"`
				Score      int     `json:"score"`
				Subreddit  string  `json:"subreddit"`
				URL        string  `json:"url"`
				CreatedUTC float64 `json:"created_utc"`
				NumComments int    `json:"num_comments"`
			} `json:"data"`
		} `json:"children"`
	} `json:"data"`
}

// Search queries Reddit for posts matching keywords.
// If RedditSubreddits is set, searches each subreddit individually and merges results.
// Otherwise does a global search. Returns at most maxResults posts total.
func (c *Client) Search(ctx context.Context, keywords string, maxResults int) ([]Post, error) {
	subs := c.cfg.RedditSubreddits
	if len(subs) == 0 {
		return c.searchGlobal(ctx, keywords, maxResults)
	}

	// Per-subreddit search, distribute limit evenly
	perSub := maxResults/len(subs) + 1
	seen := make(map[string]bool)
	var all []Post

	for _, sub := range subs {
		posts, err := c.searchSubreddit(ctx, keywords, sub, perSub)
		if err != nil {
			// Log and continue — one subreddit failing shouldn't abort all
			continue
		}
		for _, p := range posts {
			if !seen[p.URL] {
				seen[p.URL] = true
				all = append(all, p)
			}
			if len(all) >= maxResults {
				return all, nil
			}
		}
		// Respect rate limit: ~10 req/min unauthenticated
		time.Sleep(100 * time.Millisecond)
	}
	return all, nil
}

func (c *Client) searchGlobal(ctx context.Context, keywords string, limit int) ([]Post, error) {
	endpoint := c.buildURL("/search.json", keywords, limit, false)
	return c.fetch(ctx, endpoint)
}

func (c *Client) searchSubreddit(ctx context.Context, keywords, subreddit string, limit int) ([]Post, error) {
	endpoint := c.buildURL("/r/"+subreddit+"/search.json", keywords, limit, true)
	return c.fetch(ctx, endpoint)
}

func (c *Client) buildURL(path, keywords string, limit int, restrictSR bool) string {
	params := url.Values{}
	params.Set("q", keywords)
	params.Set("sort", "relevance") // was "new"; recency sort pushes off-topic recent posts
	params.Set("t", "week")         // only posts from the last 7 days
	params.Set("limit", strconv.Itoa(limit))
	if restrictSR {
		params.Set("restrict_sr", "true")
	}
	return c.cfg.RedditBaseURL + path + "?" + params.Encode()
}

func (c *Client) fetch(ctx context.Context, endpoint string) ([]Post, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("reddit: build request: %w", err)
	}
	// Reddit requires a descriptive User-Agent or it blocks the request
	ua := c.cfg.RedditUserAgent
	if ua == "" {
		ua = "sentimentbot/1.0"
	}
	req.Header.Set("User-Agent", ua)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("reddit: request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusTooManyRequests {
		return nil, fmt.Errorf("reddit: rate limited (429)")
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("reddit: unexpected status %d", resp.StatusCode)
	}

	var body redditResponse
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return nil, fmt.Errorf("reddit: decode response: %w", err)
	}

	posts := make([]Post, 0, len(body.Data.Children))
	for _, child := range body.Data.Children {
		d := child.Data
		posts = append(posts, Post{
			Title:       d.Title,
			Selftext:    d.Selftext,
			Score:       d.Score,
			Subreddit:   d.Subreddit,
			URL:         d.URL,
			CreatedAt:   time.Unix(int64(d.CreatedUTC), 0).UTC(),
			NumComments: d.NumComments,
		})
	}
	return posts, nil
}
