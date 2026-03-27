// cmd/predict: fetch a single Polymarket market by slug / URL / search query,
// compute all ML features, call the prediction server, and print a reasoned report.
//
// Usage:
//
//	go run ./cmd/predict -market "will bitcoin hit 100k by june"
//	go run ./cmd/predict -market "will-bitcoin-reach-100k-before-june-2025"
//	go run ./cmd/predict -market "https://polymarket.com/event/will-bitcoin-reach-100k"
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"net/http"
	"strings"
	"time"

	"github.com/hrishabhayush/polyxgemini/internal/config"
	"github.com/hrishabhayush/polyxgemini/internal/exchange/polymarket"
)

// ---------------------------------------------------------------------------
// ML server types (must match serve.py PredictRequest / PredictResponse)
// ---------------------------------------------------------------------------

type mlRequest struct {
	CurrentPrice    float64 `json:"current_price"`
	HurstExp        float64 `json:"hurst_exp"`
	HasHurst        int     `json:"has_hurst"`
	VolRatio        float64 `json:"vol_ratio"`
	JumpResultEnc   int     `json:"jump_result_enc"`
	TradeCount      int     `json:"trade_count"`
	LogTradeCount   float64 `json:"log_trade_count"`
	TotalVolume     float64 `json:"total_volume"`
	LogVolume       float64 `json:"log_volume"`
	AvgTradeSize    float64 `json:"avg_trade_size"`
	LogAvgTradeSize float64 `json:"log_avg_trade_size"`
	KylesLambda     float64 `json:"kyles_lambda"`
	VPIN            float64 `json:"vpin"`
	BuyFraction     float64 `json:"buy_fraction"`
	WalletHHI       float64 `json:"wallet_hhi"`
	MarketAgeDays   float64 `json:"market_age_days"`
	PriceDist50     float64 `json:"price_distance_from_50"`
}

type mlResponse struct {
	ProbYes float64 `json:"prob_yes"`
	ProbNo  float64 `json:"prob_no"`
	EdgeYes float64 `json:"edge_yes"`
	EdgeNo  float64 `json:"edge_no"`
}

// ---------------------------------------------------------------------------
// Feature extraction helpers
// ---------------------------------------------------------------------------

func jumpEnc(s string) int {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "sustained":
		return 2
	case "reversed":
		return 1
	default:
		return 0
	}
}

func ageDays(createdAt, endDate time.Time, now time.Time) float64 {
	if createdAt.IsZero() {
		return 0
	}
	ref := now
	if !endDate.IsZero() {
		ref = endDate
	}
	d := ref.Sub(createdAt).Hours() / 24.0
	if d < 0 {
		return 0
	}
	return d
}

// ---------------------------------------------------------------------------
// Reasoning engine: convert raw features into human-readable signals
// ---------------------------------------------------------------------------

type signal struct {
	name        string
	value       string
	direction   string // "bullish", "bearish", "neutral"
	explanation string
}

func reason(req mlRequest, pred mlResponse) []signal {
	var sigs []signal

	add := func(name, value, dir, expl string) {
		sigs = append(sigs, signal{name, value, dir, expl})
	}

	// Market price
	cp := req.CurrentPrice
	if cp > 0.65 {
		add("Market price", fmt.Sprintf("%.1f%%", cp*100), "bullish",
			"Crowd already pricing YES as likely — model agreement raises conviction")
	} else if cp < 0.35 {
		add("Market price", fmt.Sprintf("%.1f%%", cp*100), "bearish",
			"Crowd pricing NO as likely — contrarian YES call needs strong feature support")
	} else {
		add("Market price", fmt.Sprintf("%.1f%%", cp*100), "neutral",
			"Market near 50/50 — outcome genuinely uncertain per crowd")
	}

	// VPIN — order flow toxicity
	if req.VPIN > 0.75 {
		add("VPIN", fmt.Sprintf("%.3f", req.VPIN), "bearish",
			"High order-flow toxicity: informed traders dominating, often precedes adverse moves")
	} else if req.VPIN < 0.4 {
		add("VPIN", fmt.Sprintf("%.3f", req.VPIN), "bullish",
			"Low toxicity: uninformed retail flow, price likely near fair value")
	} else {
		add("VPIN", fmt.Sprintf("%.3f", req.VPIN), "neutral", "Moderate order-flow toxicity")
	}

	// Buy fraction — directional pressure
	if req.BuyFraction > 0.65 {
		add("Buy pressure", fmt.Sprintf("%.1f%% buys", req.BuyFraction*100), "bullish",
			"Majority of trades are buys — net positive flow toward YES")
	} else if req.BuyFraction < 0.35 {
		add("Buy pressure", fmt.Sprintf("%.1f%% buys", req.BuyFraction*100), "bearish",
			"Majority of trades are sells — net negative flow against YES")
	} else {
		add("Buy pressure", fmt.Sprintf("%.1f%% buys", req.BuyFraction*100), "neutral",
			"Balanced buy/sell flow — no clear directional pressure")
	}

	// Kyle's lambda — price impact
	if math.Abs(req.KylesLambda) > 0.001 {
		add("Price impact (λ)", fmt.Sprintf("%.5f", req.KylesLambda), "bearish",
			"High price impact per unit volume: thin liquidity, susceptible to manipulation")
	} else if math.Abs(req.KylesLambda) < 0.0001 && req.KylesLambda != 0 {
		add("Price impact (λ)", fmt.Sprintf("%.5f", req.KylesLambda), "bullish",
			"Very low price impact: deep liquidity, price discovery is efficient")
	} else {
		add("Price impact (λ)", fmt.Sprintf("%.5f", req.KylesLambda), "neutral",
			"Moderate price impact, typical for active markets")
	}

	// Wallet HHI — concentration
	if req.WalletHHI > 0.4 {
		add("Wallet concentration", fmt.Sprintf("HHI=%.3f", req.WalletHHI), "bearish",
			"Highly concentrated ownership: a few wallets dominate, higher manipulation risk")
	} else if req.WalletHHI < 0.1 {
		add("Wallet concentration", fmt.Sprintf("HHI=%.3f", req.WalletHHI), "bullish",
			"Widely distributed ownership: decentralized participation, robust price signal")
	} else {
		add("Wallet concentration", fmt.Sprintf("HHI=%.3f", req.WalletHHI), "neutral",
			"Moderate wallet concentration")
	}

	// Hurst exponent — mean-reversion vs trend
	if req.HasHurst == 1 {
		if req.HurstExp > 0.6 {
			add("Hurst exponent", fmt.Sprintf("%.3f", req.HurstExp), "bullish",
				"Trending behavior (H>0.5): price moves tend to persist in current direction")
		} else if req.HurstExp < 0.4 {
			add("Hurst exponent", fmt.Sprintf("%.3f", req.HurstExp), "bearish",
				"Mean-reverting behavior (H<0.5): price tends to reverse, current trend unreliable")
		} else {
			add("Hurst exponent", fmt.Sprintf("%.3f", req.HurstExp), "neutral",
				"Near-random walk (H≈0.5): no strong trend or mean-reversion signal")
		}
	} else {
		add("Hurst exponent", "N/A", "neutral",
			"Insufficient price history to compute — market too new or low-activity")
	}

	// Average trade size — institutional vs retail
	if req.AvgTradeSize > 500 {
		add("Avg trade size", fmt.Sprintf("$%.0f", req.AvgTradeSize), "bullish",
			"Large average trades suggest institutional/informed participation")
	} else if req.AvgTradeSize < 50 {
		add("Avg trade size", fmt.Sprintf("$%.0f", req.AvgTradeSize), "neutral",
			"Small retail-sized trades — price may lag true information")
	} else {
		add("Avg trade size", fmt.Sprintf("$%.0f", req.AvgTradeSize), "neutral",
			"Typical mixed retail/institutional participation")
	}

	// Market age
	if req.MarketAgeDays > 90 {
		add("Market age", fmt.Sprintf("%.0f days", req.MarketAgeDays), "bullish",
			"Mature market: price has had time to incorporate information")
	} else if req.MarketAgeDays < 14 {
		add("Market age", fmt.Sprintf("%.0f days", req.MarketAgeDays), "neutral",
			"Young market: price discovery may be incomplete, higher uncertainty")
	} else {
		add("Market age", fmt.Sprintf("%.0f days", req.MarketAgeDays), "neutral",
			"Moderate market age")
	}

	// Volume
	if req.TotalVolume > 100_000 {
		add("Total volume", fmt.Sprintf("$%.0f", req.TotalVolume), "bullish",
			"High liquidity: crowd wisdom well-established, model estimates more reliable")
	} else if req.TotalVolume < 5_000 {
		add("Total volume", fmt.Sprintf("$%.0f", req.TotalVolume), "bearish",
			"Low liquidity: sparse trading, prediction confidence is lower")
	} else {
		add("Total volume", fmt.Sprintf("$%.0f", req.TotalVolume), "neutral",
			"Moderate volume")
	}

	return sigs
}

// ---------------------------------------------------------------------------
// ML server call
// ---------------------------------------------------------------------------

func callServer(serverURL string, req mlRequest) (*mlResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	resp, err := http.Post(serverURL+"/predict", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("ML server unreachable at %s — is it running?\n  cd python/ml && uvicorn serve:app --host 127.0.0.1 --port 8766\n  error: %w", serverURL, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ML server returned HTTP %d", resp.StatusCode)
	}
	var pred mlResponse
	if err := json.NewDecoder(resp.Body).Decode(&pred); err != nil {
		return nil, err
	}
	return &pred, nil
}

// ---------------------------------------------------------------------------
// Report printer
// ---------------------------------------------------------------------------

func printReport(m polymarket.ResolvedMarket, req mlRequest, pred *mlResponse) {
	hr := strings.Repeat("─", 70)
	fmt.Println()
	fmt.Println(hr)
	fmt.Printf("  MARKET PREDICTION REPORT\n")
	fmt.Println(hr)
	fmt.Printf("  Question : %s\n", m.Question)
	fmt.Printf("  Slug     : %s\n", m.Slug)
	if !m.EndDate.IsZero() {
		fmt.Printf("  Closes   : %s\n", m.EndDate.Format("2006-01-02"))
	}
	fmt.Println(hr)

	// Verdict
	buyYes := pred.EdgeYes >= pred.EdgeNo
	side := "YES"
	edge := pred.EdgeYes
	if !buyYes {
		side = "NO"
		edge = pred.EdgeNo
	}

	fmt.Printf("\n  %-14s  P(YES) = %.1f%%   P(NO) = %.1f%%\n",
		"MODEL OUTPUT:", pred.ProbYes*100, pred.ProbNo*100)
	fmt.Printf("  %-14s  %+.1f%% vs market price of %.1f%%\n",
		"EDGE:", edge*100, req.CurrentPrice*100)

	verdict := "HOLD / SKIP"
	switch {
	case edge >= 0.15:
		verdict = fmt.Sprintf("STRONG BUY %s", side)
	case edge >= 0.07:
		verdict = fmt.Sprintf("BUY %s", side)
	case edge >= 0.03:
		verdict = fmt.Sprintf("LEAN %s", side)
	}
	fmt.Printf("  %-14s  %s\n", "SIGNAL:", verdict)

	// Features
	fmt.Printf("\n%s\n  RAW FEATURES\n%s\n", hr, hr)
	fmt.Printf("  %-22s  %10s\n", "Feature", "Value")
	fmt.Printf("  %-22s  %10s\n", strings.Repeat("-", 22), strings.Repeat("-", 10))
	fmt.Printf("  %-22s  %10.4f\n", "current_price", req.CurrentPrice)
	fmt.Printf("  %-22s  %10.0f\n", "trade_count", float64(req.TradeCount))
	fmt.Printf("  %-22s  %10.0f\n", "total_volume ($)", req.TotalVolume)
	fmt.Printf("  %-22s  %10.2f\n", "avg_trade_size ($)", req.AvgTradeSize)
	fmt.Printf("  %-22s  %10.4f\n", "vpin", req.VPIN)
	fmt.Printf("  %-22s  %10.4f\n", "buy_fraction", req.BuyFraction)
	fmt.Printf("  %-22s  %10.6f\n", "kyles_lambda", req.KylesLambda)
	fmt.Printf("  %-22s  %10.4f\n", "wallet_hhi", req.WalletHHI)
	if req.HasHurst == 1 {
		fmt.Printf("  %-22s  %10.4f\n", "hurst_exp", req.HurstExp)
		fmt.Printf("  %-22s  %10.4f\n", "vol_ratio", req.VolRatio)
	} else {
		fmt.Printf("  %-22s  %10s\n", "hurst_exp", "N/A")
	}
	fmt.Printf("  %-22s  %10.1f\n", "market_age_days", req.MarketAgeDays)

	// Reasoning
	fmt.Printf("\n%s\n  REASONING\n%s\n", hr, hr)
	sigs := reason(req, *pred)
	icons := map[string]string{"bullish": "▲", "bearish": "▼", "neutral": "●"}
	for _, s := range sigs {
		icon := icons[s.direction]
		fmt.Printf("  %s %-22s  %s\n", icon, s.name, s.value)
		fmt.Printf("    └ %s\n", s.explanation)
	}

	// Confidence note
	fmt.Printf("\n%s\n", hr)
	switch {
	case req.TotalVolume < 5000:
		fmt.Println("  ⚠  Low volume — treat prediction with caution.")
	case math.Abs(edge) < 0.05:
		fmt.Println("  ⚠  Edge too small to act on confidently.")
	case pred.ProbYes > 0.9 || pred.ProbYes < 0.1:
		fmt.Println("  ⚠  Extreme probability — verify market hasn't already resolved.")
	}
	fmt.Println(hr)
	fmt.Println()
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

func main() {
	configPath := flag.String("config", "config/config.yaml", "path to config file")
	marketQuery := flag.String("market", "", "market slug, URL, or search query (required)")
	serverURL := flag.String("server", "http://127.0.0.1:8766", "ML prediction server URL")
	flag.Duration("ws-timeout", 3*time.Second, "WebSocket collection timeout (reserved)")
	flag.Parse()

	if *marketQuery == "" {
		fmt.Println("Usage: go run ./cmd/predict -market \"<slug | URL | search query>\"")
		fmt.Println("Example: go run ./cmd/predict -market \"will bitcoin hit 100k by june\"")
		flag.Usage()
		return
	}

	cfg, err := config.Load(*configPath)
	if err != nil {
		log.Fatalf("config: %v", err)
	}

	ctx := context.Background()
	polyClient := polymarket.NewClient(cfg.Polymarket)

	fmt.Printf("Searching for market: %q\n", *marketQuery)

	// Resolution order:
	// 1. Try events?slug= (handles both single-market and multi-market events, any URL)
	// 2. Try markets?slug= (direct market slug)
	// 3. Try keyword search in active markets
	var markets []polymarket.ResolvedMarket

	eventMarkets, evErr := polyClient.ResolveEvent(ctx, *marketQuery)
	if evErr == nil {
		markets = eventMarkets
		if len(markets) == 1 {
			fmt.Printf("Found: %s\n", markets[0].Question)
		} else {
			fmt.Printf("Found event with %d markets\n", len(markets))
		}
	} else {
		m, err := polyClient.FindMarket(ctx, *marketQuery)
		if err != nil {
			log.Fatalf("market not found.\n  events API: %v\n  markets API: %v", evErr, err)
		}
		markets = []polymarket.ResolvedMarket{*m}
		fmt.Printf("Found: %s\n", m.Question)
	}

	now := time.Now().UTC()

	for i, mkt := range markets {
		if len(markets) > 1 {
			fmt.Printf("\n[%d/%d] Fetching features for: %s\n", i+1, len(markets), mkt.Question)
		} else {
			fmt.Printf("Found: %s\nFetching features...\n", mkt.Question)
		}

		var currentPrice, hurstExp, volRatio float64
		var jumpResult string
		hasHurst := 0

		points, err := polyClient.FetchPriceHistory(ctx, mkt.ClobTokenIDs[0], "max", 200)
		if err == nil && len(points) > 0 {
			prices := polymarket.Prices(points)
			currentPrice = prices[len(prices)-1]
			hurstExp = polymarket.HurstExponent(prices)
			if hurstExp != 0 {
				hasHurst = 1
			}
			curVol, histVol := polymarket.GARCHVolatility(prices)
			if histVol > 0 {
				volRatio = curVol / histVol
			}
			jumpResult = polymarket.JumpPersistence(prices, 0.03)
		}

		var tradeCount int
		var totalVolume, kylesLambda, vpin, buyFraction, walletHHI float64

		trades, err := polyClient.FetchTrades(ctx, mkt.ConditionID, 200)
		if err == nil && len(trades) > 0 {
			tradeCount = len(trades)
			for _, t := range trades {
				totalVolume += t.Size
			}
			kylesLambda = polymarket.KylesLambda(trades)
			if bs := totalVolume / 50; bs > 0 {
				vpin = polymarket.VPIN(trades, bs)
			}
			buyFraction = polymarket.LeeReadyStats(trades)
			walletHHI, _ = polymarket.WalletConcentration(trades, 0)
		}

		avgTradeSize := totalVolume / float64(tradeCount+1)
		marketAge := ageDays(mkt.CreatedAt, mkt.EndDate, now)

		req := mlRequest{
			CurrentPrice:    currentPrice,
			HurstExp:        hurstExp,
			HasHurst:        hasHurst,
			VolRatio:        volRatio,
			JumpResultEnc:   jumpEnc(jumpResult),
			TradeCount:      tradeCount,
			LogTradeCount:   math.Log1p(float64(tradeCount)),
			TotalVolume:     totalVolume,
			LogVolume:       math.Log1p(totalVolume),
			AvgTradeSize:    avgTradeSize,
			LogAvgTradeSize: math.Log1p(avgTradeSize),
			KylesLambda:     kylesLambda,
			VPIN:            vpin,
			BuyFraction:     buyFraction,
			WalletHHI:       walletHHI,
			MarketAgeDays:   marketAge,
			PriceDist50:     math.Abs(currentPrice - 0.5),
		}

		pred, err := callServer(*serverURL, req)
		if err != nil {
			log.Fatalf("prediction failed: %v", err)
		}

		printReport(mkt, req, pred)
	}
}
