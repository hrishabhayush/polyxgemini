package ml

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"time"
)

var ErrServerUnreachable = errors.New("ml: prediction server unreachable")

// PredictRequest mirrors the feature schema expected by the Python prediction server.
type PredictRequest struct {
	CurrentPrice          float64 `json:"current_price"`
	HurstExp              float64 `json:"hurst_exp"`
	VolRatio              float64 `json:"vol_ratio"`
	JumpResultEnc         int     `json:"jump_result_enc"`
	TradeCount            int     `json:"trade_count"`
	TotalVolume           float64 `json:"total_volume"`
	LogVolume             float64 `json:"log_volume"`
	KylesLambda           float64 `json:"kyles_lambda"`
	VPIN                  float64 `json:"vpin"`
	BuyFraction           float64 `json:"buy_fraction"`
	WalletHHI             float64 `json:"wallet_hhi"`
	ArticleCount          int     `json:"article_count"`
	BullishScore          float64 `json:"bullish_score"`
	BearishScore          float64 `json:"bearish_score"`
	SentimentNet          float64 `json:"sentiment_net"`
	ResolutionReliability float64 `json:"resolution_reliability"`
	MarketAgeDays         float64 `json:"market_age_days"`
	PriceDistanceFrom50   float64 `json:"price_distance_from_50"`
}

// Prediction is the ML server's response for a single market.
type Prediction struct {
	ProbYes float64 `json:"prob_yes"`
	ProbNo  float64 `json:"prob_no"`
	EdgeYes float64 `json:"edge_yes"`
	EdgeNo  float64 `json:"edge_no"`
}

// Client calls the Python ML prediction server.
type Client struct {
	baseURL    string
	httpClient *http.Client
}

func NewClient(baseURL string) *Client {
	return &Client{
		baseURL:    baseURL,
		httpClient: &http.Client{Timeout: 10 * time.Second},
	}
}

// Ping checks whether the prediction server is reachable and has a model loaded.
func (c *Client) Ping(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/health", nil)
	if err != nil {
		return fmt.Errorf("ml: ping build request: %w", err)
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("%w: %v", ErrServerUnreachable, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("ml: health check status %d", resp.StatusCode)
	}
	return nil
}

// Predict sends market features to the ML server and returns the prediction.
func (c *Client) Predict(ctx context.Context, req *PredictRequest) (*Prediction, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("ml: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.baseURL+"/predict", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("ml: build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrServerUnreachable, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ml: predict returned status %d", resp.StatusCode)
	}

	var pred Prediction
	if err := json.NewDecoder(resp.Body).Decode(&pred); err != nil {
		return nil, fmt.Errorf("ml: decode prediction: %w", err)
	}
	return &pred, nil
}

// EncodeJumpResult converts the string jump persistence result to the integer
// encoding expected by the ML model.
func EncodeJumpResult(result string) int {
	switch result {
	case "sustained":
		return 2
	case "reversed":
		return 1
	default:
		return 0
	}
}

// ComputeLogVolume returns log(1 + volume), matching the Python feature engineering.
func ComputeLogVolume(totalVolume float64) float64 {
	return math.Log1p(totalVolume)
}
