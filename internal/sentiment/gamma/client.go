package gamma

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
	"unicode"
)

// Comment is a normalized Polymarket Gamma comment.
type Comment struct {
	ID            string
	Body          string
	UserAddress   string
	CreatedAt     time.Time
	ReactionCount int
	PositionSize  float64 // total token position across YES+NO, parsed from profile.positions
}

// Client fetches market comments from the Gamma API.
type Client struct {
	baseURL    string
	httpClient *http.Client
}

// NewClient creates a Gamma comments client.
// baseURL should be "https://gamma-api.polymarket.com".
func NewClient(baseURL string) *Client {
	return &Client{
		baseURL:    strings.TrimRight(baseURL, "/"),
		httpClient: &http.Client{Timeout: 10 * time.Second},
	}
}

// rawComment mirrors the Gamma /comments JSON schema.
type rawComment struct {
	ID              string     `json:"id"`
	Body            string     `json:"body"`
	UserAddress     string     `json:"userAddress"`
	CreatedAt       *time.Time `json:"createdAt"`
	ReactionCount   *int       `json:"reactionCount"`
	Profile         *rawProfile `json:"profile"`
}

type rawProfile struct {
	Positions []rawPosition `json:"positions"`
}

type rawPosition struct {
	TokenID      string `json:"tokenId"`
	PositionSize string `json:"positionSize"`
}

// FetchComments retrieves up to limit comments for the given parent entity ID.
// For market discussions this is the Gamma Event ID (parent_entity_type=Event).
// Comments are ordered newest-first.
func (c *Client) FetchComments(ctx context.Context, parentEntityID int, limit int) ([]Comment, error) {
	params := url.Values{}
	// Gamma comments API currently expects PascalCase entity types.
	// "Event" works for market-level discussion threads tied to event IDs.
	params.Set("parent_entity_type", "Event")
	params.Set("parent_entity_id", strconv.Itoa(parentEntityID))
	params.Set("limit", strconv.Itoa(limit))
	params.Set("order", "createdAt")
	params.Set("ascending", "false")
	params.Set("get_positions", "true")

	endpoint := c.baseURL + "/comments?" + params.Encode()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("gamma/comments: build request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("gamma/comments: request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("gamma/comments: unexpected status %d", resp.StatusCode)
	}

	var raw []rawComment
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return nil, fmt.Errorf("gamma/comments: decode: %w", err)
	}

	comments := make([]Comment, 0, len(raw))
	for _, r := range raw {
		cm := Comment{
			ID:          r.ID,
			Body:        r.Body,
			UserAddress: r.UserAddress,
		}
		if r.CreatedAt != nil {
			cm.CreatedAt = *r.CreatedAt
		}
		if r.ReactionCount != nil {
			cm.ReactionCount = *r.ReactionCount
		}
		if r.Profile != nil {
			for _, p := range r.Profile.Positions {
				if sz, err := strconv.ParseFloat(p.PositionSize, 64); err == nil {
					cm.PositionSize += math.Abs(sz)
				}
			}
		}
		comments = append(comments, cm)
	}
	return comments, nil
}

// FilterRelevant removes low-signal comments before NLP scoring.
// Strips comments shorter than minLen, deleted/removed markers, and pure emoji/whitespace.
func FilterRelevant(comments []Comment, minLen int) []Comment {
	out := make([]Comment, 0, len(comments))
	for _, c := range comments {
		body := strings.TrimSpace(c.Body)
		if len(body) < minLen {
			continue
		}
		lower := strings.ToLower(body)
		if lower == "[removed]" || lower == "[deleted]" {
			continue
		}
		if isPureNonAlpha(body) {
			continue
		}
		out = append(out, c)
	}
	return out
}

func isPureNonAlpha(s string) bool {
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			return false
		}
	}
	return true
}

// PositionWeight computes a bounded, concave confidence modifier based on
// a commenter's total position size.
//
//	weight = 1.0 + alpha * tanh(positionSize / scale)
//
// alpha controls max boost (default 0.5 → max 1.5x), scale sets the
// half-saturation point (default 500). Hard-capped at maxWeight (2.0).
// Zero or negative position returns 1.0 (no penalty, no boost).
func PositionWeight(positionSize, scale float64) float64 {
	const alpha = 0.5
	const maxWeight = 2.0

	if positionSize <= 0 || scale <= 0 {
		return 1.0
	}
	w := 1.0 + alpha*math.Tanh(positionSize/scale)
	if w > maxWeight {
		w = maxWeight
	}
	if w < 1.0 {
		w = 1.0
	}
	return w
}
