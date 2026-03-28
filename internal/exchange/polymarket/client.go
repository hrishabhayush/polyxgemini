package polymarket

import (
	"bytes"
	"context"
	"crypto/ecdsa"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	ethcrypto "github.com/ethereum/go-ethereum/crypto"

	"github.com/hrishabhayush/polyxgemini/internal/config"
	"github.com/hrishabhayush/polyxgemini/internal/exchange"
)

type Client struct {
	cfg        config.PolymarketConfig
	httpClient *http.Client
	privateKey *ecdsa.PrivateKey
}

func NewClient(cfg config.PolymarketConfig) *Client {
	c := &Client{cfg: cfg, httpClient: &http.Client{}}
	if cfg.PrivateKey != "" {
		pk := strings.TrimPrefix(cfg.PrivateKey, "0x")
		key, err := ethcrypto.HexToECDSA(pk)
		if err == nil {
			c.privateKey = key
		}
	}
	return c
}

func (c *Client) FetchMarkets(ctx context.Context) ([]exchange.Market, error) {
	// TODO: GET gamma_base_url/markets -> normalize to []Market
	return nil, nil
}

func (c *Client) FetchOrderBook(ctx context.Context, tokenID string) (*exchange.OrderBook, error) {
	// TODO: GET clob_base_url/order-book?token_id=tokenID
	return nil, nil
}

func (c *Client) PlaceOrder(ctx context.Context, order *exchange.OrderRequest) (*exchange.Order, error) {
	if c.privateKey == nil {
		return nil, fmt.Errorf("polymarket: private_key not configured — cannot sign orders")
	}

	signerAddr := ethcrypto.PubkeyToAddress(c.privateKey.PublicKey).Hex()
	funderAddr := c.cfg.WalletAddress
	if funderAddr == "" {
		funderAddr = signerAddr
	}

	side := SideBuy
	if order.Side == "sell" || order.Side == "SELL" {
		side = SideSell
	}

	makerAmt, takerAmt := ComputeAmounts(side, order.Price, order.Quantity)

	signedOrder, err := BuildSignedOrder(
		c.privateKey,
		funderAddr,
		signerAddr,
		order.MarketID,
		side,
		makerAmt,
		takerAmt,
		0,
		c.cfg.SignatureType,
		c.cfg.APIKey,
		"FOK",
		false,
	)
	if err != nil {
		return nil, fmt.Errorf("polymarket: build signed order: %w", err)
	}

	payload, err := json.Marshal(signedOrder)
	if err != nil {
		return nil, fmt.Errorf("polymarket: marshal order: %w", err)
	}

	timestamp := strconv.FormatInt(time.Now().Unix(), 10)
	requestPath := "/order"
	hmacSig := c.sign(timestamp, http.MethodPost, requestPath, string(payload))

	url := c.cfg.CLOBBaseURL + requestPath
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header["POLY_ADDRESS"] = []string{signerAddr}
	req.Header["POLY_SIGNATURE"] = []string{hmacSig}
	req.Header["POLY_TIMESTAMP"] = []string{timestamp}
	req.Header["POLY_API_KEY"] = []string{c.cfg.APIKey}
	req.Header["POLY_PASSPHRASE"] = []string{c.cfg.Passphrase}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("polymarket: order request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("polymarket: order returned status %d: %s", resp.StatusCode, string(respBody))
	}

	var result struct {
		OrderID   string  `json:"orderID"`
		Status    string  `json:"status"`
		FilledQty float64 `json:"filledSize"`
	}
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("polymarket: decode order response: %w", err)
	}

	return &exchange.Order{
		ID:        result.OrderID,
		Status:    result.Status,
		FilledQty: result.FilledQty,
	}, nil
}

func (c *Client) sign(timestamp, method, requestPath, body string) string {
	message := timestamp + method + requestPath
	if body != "" {
		message += body
	}
	secret, _ := base64.URLEncoding.DecodeString(c.cfg.APISecret)
	mac := hmac.New(sha256.New, secret)
	mac.Write([]byte(message))
	return base64.URLEncoding.EncodeToString(mac.Sum(nil))
}

func (c *Client) CancelOrder(ctx context.Context, orderID string) error {
	// TODO: DELETE clob_base_url/order/orderID (authenticated)
	return nil
}

// ResolvedMarket holds the Gamma API response fields needed for trading.
type ResolvedMarket struct {
	Slug         string
	Question     string
	ConditionID  string
	ClobTokenIDs [2]string // [YES, NO]
}

// gammaMarketResponse is the raw Gamma API response for a single market.
// Note: clobTokenIds comes back as a JSON-encoded string, not a native array.
type gammaMarketResponse struct {
	Slug         string `json:"slug"`
	Question     string `json:"question"`
	ConditionID  string `json:"conditionId"`
	ClobTokenIDs string `json:"clobTokenIds"`
}

// ResolveMarket looks up a market by slug via the Gamma API and returns its token IDs.
func (c *Client) ResolveMarket(ctx context.Context, slug string) (*ResolvedMarket, error) {
	url := fmt.Sprintf("%s/markets?slug=%s", c.cfg.GammaBaseURL, slug)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("gamma api request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("gamma api returned status %d", resp.StatusCode)
	}

	var markets []gammaMarketResponse
	if err := json.NewDecoder(resp.Body).Decode(&markets); err != nil {
		return nil, fmt.Errorf("failed to decode gamma response: %w", err)
	}

	if len(markets) == 0 {
		return nil, fmt.Errorf("no market found for slug: %s", slug)
	}

	m := markets[0]

	// clobTokenIds is a JSON string like '["token1", "token2"]' — parse it
	var tokenIDs []string
	if err := json.Unmarshal([]byte(m.ClobTokenIDs), &tokenIDs); err != nil {
		return nil, fmt.Errorf("failed to parse clobTokenIds for %s: %w", slug, err)
	}
	if len(tokenIDs) < 2 {
		return nil, fmt.Errorf("market %s has fewer than 2 token IDs", slug)
	}

	return &ResolvedMarket{
		Slug:         m.Slug,
		Question:     m.Question,
		ConditionID:  m.ConditionID,
		ClobTokenIDs: [2]string{tokenIDs[0], tokenIDs[1]},
	}, nil
}
