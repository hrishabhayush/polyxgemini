package report_test

import (
	"context"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/hrishabhayush/polyxgemini/internal/config"
	"github.com/hrishabhayush/polyxgemini/internal/exchange/polymarket"
	"github.com/hrishabhayush/polyxgemini/internal/report"
)

const (
	defaultMarket      = "cs2-prv-vit-2026-03-23"
	defaultFinBERTURL  = "http://localhost:8765"
	defaultNewsAPIBase = "https://newsapi.org/v2"
	defaultGDELTBase   = "https://api.gdeltproject.org/api/v2/doc/doc"
	defaultRedditBase  = "https://www.reddit.com"
)

func TestMarketReport(t *testing.T) {
	marketName := defaultMarket
	if v := os.Getenv("MARKET_NAME"); v != "" {
		marketName = v
	}

	polyCfg := config.PolymarketConfig{
		CLOBBaseURL:  "https://clob.polymarket.com",
		GammaBaseURL: "https://gamma-api.polymarket.com",
		DataBaseURL:  "https://data-api.polymarket.com",
	}

	finbertURL := defaultFinBERTURL
	if v := os.Getenv("FINBERT_URL"); v != "" {
		finbertURL = v
	}

	sentCfg := config.SentimentConfig{
		NewsAPIKey:      os.Getenv("NEWSAPI_KEY"),
		NewsAPIBaseURL:  defaultNewsAPIBase,
		GDELTBaseURL:    defaultGDELTBase,
		RedditBaseURL:   defaultRedditBase,
		RedditUserAgent: "marketbot/1.0",
		FinBERTServerURL: finbertURL,
	}

	ctx := context.Background()
	gen := report.NewGenerator(polyCfg, sentCfg)

	rpt, err := gen.Generate(ctx, marketName)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	printReport(t, rpt)
}

// TestWSMetrics verifies WebSocket data collection independently.
// It resolves a market, connects to the CLOB WebSocket for 8 seconds,
// and prints the resulting mid-price, spread, and OBI.
func TestWSMetrics(t *testing.T) {
	marketName := defaultMarket
	if v := os.Getenv("MARKET_NAME"); v != "" {
		marketName = v
	}

	polyCfg := config.PolymarketConfig{
		CLOBBaseURL:  "https://clob.polymarket.com",
		GammaBaseURL: "https://gamma-api.polymarket.com",
		DataBaseURL:  "https://data-api.polymarket.com",
	}

	ctx := context.Background()
	client := polymarket.NewClient(polyCfg)

	market, err := client.FindMarket(ctx, marketName)
	if err != nil {
		t.Fatalf("FindMarket failed: %v", err)
	}

	t.Logf("Market: %s", market.Question)
	t.Logf("YES token: %s", market.ClobTokenIDs[0])
	t.Logf("Collecting WS data for 8 seconds...")

	metrics, err := polymarket.FetchWSMetrics(*market, 8*time.Second)
	if err != nil {
		t.Fatalf("FetchWSMetrics failed: %v", err)
	}

	width := 50
	sep := strings.Repeat("═", width)
	thin := strings.Repeat("─", width)

	fmt.Printf("\n%s\n", sep)
	fmt.Printf("  WEBSOCKET METRICS SNAPSHOT\n")
	fmt.Printf("  %s\n", market.Question)
	fmt.Printf("%s\n", sep)
	fmt.Printf("  %-22s %.4f  (%.1f¢)\n", "Mid-price", metrics.MidPrice, metrics.MidPrice*100)
	fmt.Printf("  %-22s %.4f  (%.1f¢)\n", "Best bid", metrics.BestBid, metrics.BestBid*100)
	fmt.Printf("  %-22s %.4f  (%.1f¢)\n", "Best ask", metrics.BestAsk, metrics.BestAsk*100)
	fmt.Printf("%s\n", thin)
	fmt.Printf("  %-22s %.4f  %s\n", "Bid-ask spread", metrics.BidAskSpread, spreadNote(metrics.BidAskSpread))
	fmt.Printf("  %-22s %+.4f  %s\n", "OBI", metrics.OBI, obiNote(metrics.OBI))
	fmt.Printf("%s\n", thin)
	fmt.Printf("  %-22s %.0f\n", "Total bid depth", metrics.TotalBidSize)
	fmt.Printf("  %-22s %.0f\n", "Total ask depth", metrics.TotalAskSize)
	fmt.Printf("%s\n\n", sep)

	if metrics.MidPrice == 0 {
		t.Log("WARNING: mid-price is 0 — WS may not have received PriceChanges data in time")
	}
	if metrics.BidAskSpread == 0 && metrics.OBI == 0 {
		t.Log("WARNING: spread and OBI are both 0 — WS may not have received AggOrderbook data in time")
	}
}

// TestWSMetricsComputation verifies the pure computation functions
// without requiring a live WebSocket connection.
func TestWSMetricsComputation(t *testing.T) {
	t.Run("MidPrice", func(t *testing.T) {
		if got := polymarket.MidPrice(0.45, 0.55); fmt.Sprintf("%.4f", got) != "0.5000" {
			t.Errorf("MidPrice(0.45, 0.55) = %f, want 0.5", got)
		}
		if got := polymarket.MidPrice(0, 0.55); got != 0 {
			t.Errorf("MidPrice(0, 0.55) = %f, want 0", got)
		}
	})

	t.Run("BidAskSpread", func(t *testing.T) {
		if got := polymarket.BidAskSpread(0.48, 0.52); fmt.Sprintf("%.4f", got) != "0.0400" {
			t.Errorf("BidAskSpread(0.48, 0.52) = %f, want 0.04", got)
		}
		if got := polymarket.BidAskSpread(0, 0.52); got != 0 {
			t.Errorf("BidAskSpread(0, 0.52) = %f, want 0", got)
		}
	})

	t.Run("OrderBookImbalance", func(t *testing.T) {
		if got := polymarket.OrderBookImbalance(100, 100); got != 0 {
			t.Errorf("OBI(100, 100) = %f, want 0", got)
		}
		if got := polymarket.OrderBookImbalance(150, 50); fmt.Sprintf("%.2f", got) != "0.50" {
			t.Errorf("OBI(150, 50) = %f, want 0.50", got)
		}
		if got := polymarket.OrderBookImbalance(50, 150); fmt.Sprintf("%.2f", got) != "-0.50" {
			t.Errorf("OBI(50, 150) = %f, want -0.50", got)
		}
		if got := polymarket.OrderBookImbalance(0, 0); got != 0 {
			t.Errorf("OBI(0, 0) = %f, want 0", got)
		}
	})
}

// ── pretty-print ─────────────────────────────────────────────────────────────

func printReport(t *testing.T, rpt *report.MarketReport) {
	t.Helper()
	width := 66
	sep := strings.Repeat("═", width)
	thin := strings.Repeat("─", width)

	nlpTag := "[NLP: off]"
	if rpt.NLPAvailable {
		nlpTag = "[NLP: on] "
	}

	condShort := rpt.ConditionID
	if len(condShort) > 20 {
		condShort = condShort[:8] + "..." + condShort[len(condShort)-6:]
	}

	fmt.Printf("\n%s\n", sep)
	fmt.Printf("  MARKET REPORT: %s\n", rpt.MarketName)
	fmt.Printf("  Condition ID: %-38s %s\n", condShort, nlpTag)
	fmt.Printf("%s\n", sep)

	// Microstructure
	fmt.Printf("  MICROSTRUCTURE\n")
	fmt.Printf("%s\n", thin)
	if rpt.TradeCount == 0 {
		fmt.Printf("  (no trade data)\n")
	} else {
		fmt.Printf("  %-24s %-12d $%.0f total vol\n", "Trades (200)", rpt.TradeCount, rpt.TotalVolume)
		fmt.Printf("  %-24s %-12s %s\n", "Kyle's λ",
			fmt.Sprintf("%.6f", rpt.KylesLambda), lambdaNote(rpt.KylesLambda))
		fmt.Printf("  %-24s %-12s %s\n", "VPIN",
			fmt.Sprintf("%.4f", rpt.VPIN), vpinNote(rpt.VPIN))
		fmt.Printf("  %-24s %-12s fraction buyer-initiated\n", "Lee-Ready buy %",
			fmt.Sprintf("%.1f%%", rpt.BuyFraction*100))
		fmt.Printf("  %-24s %-12s %s\n", "Wallet HHI",
			fmt.Sprintf("%.4f", rpt.WalletHHI), hhiNote(rpt.WalletHHI))
	}

	// Price dynamics
	fmt.Printf("%s\n", thin)
	fmt.Printf("  PRICE DYNAMICS (YES token)\n")
	fmt.Printf("%s\n", thin)
	if rpt.CurrentPrice == 0 {
		fmt.Printf("  (no price history)\n")
	} else {
		fmt.Printf("  %-24s %.4f\n", "Current price (YES)", rpt.CurrentPrice)
		fmt.Printf("  %-24s %-12s %s\n", "Hurst exponent",
			fmt.Sprintf("%.4f", rpt.HurstExp), hurstNote(rpt.HurstExp))
		fmt.Printf("  %-24s %-12s %s\n", "Vol ratio (cur/hist)",
			fmt.Sprintf("%.2fx", rpt.VolRatio), volNote(rpt.VolRatio))
		fmt.Printf("  %-24s %-12s %s\n", "Jump persistence",
			rpt.JumpResult, jumpNote(rpt.JumpResult))
	}

	// WebSocket real-time data
	fmt.Printf("%s\n", thin)
	fmt.Printf("  WEBSOCKET LIVE DATA (YES token)\n")
	fmt.Printf("%s\n", thin)
	if !rpt.WSAvailable {
		fmt.Printf("  (no WebSocket data — connection may have failed)\n")
	} else {
		fmt.Printf("  %-24s %.4f\n", "Mid-price", rpt.WSMidPrice)
		fmt.Printf("  %-24s %-12s (%s)\n", "Best bid",
			fmt.Sprintf("%.4f", rpt.WSBestBid),
			fmt.Sprintf("%.1f¢", rpt.WSBestBid*100))
		fmt.Printf("  %-24s %-12s (%s)\n", "Best ask",
			fmt.Sprintf("%.4f", rpt.WSBestAsk),
			fmt.Sprintf("%.1f¢", rpt.WSBestAsk*100))
		fmt.Printf("  %-24s %-12s %s\n", "Bid-ask spread",
			fmt.Sprintf("%.4f", rpt.WSBidAskSpread), spreadNote(rpt.WSBidAskSpread))
		fmt.Printf("  %-24s %-12s %s\n", "Order Book Imbalance",
			fmt.Sprintf("%+.4f", rpt.WSOBI), obiNote(rpt.WSOBI))
		fmt.Printf("  %-24s bid=%.0f  ask=%.0f\n", "Book depth (shares)",
			rpt.WSTotalBidSize, rpt.WSTotalAskSize)
	}

	// Sentiment
	fmt.Printf("%s\n", thin)
	newsapiN := rpt.SourceCounts["newsapi"]
	gdeltN := rpt.SourceCounts["gdelt"]
	redditN := rpt.SourceCounts["reddit"]
	fmt.Printf("  NEWS SENTIMENT  (%d articles, last 7d)\n", rpt.ArticleCount)
	fmt.Printf("%s\n", thin)
	if rpt.ArticleCount == 0 {
		fmt.Printf("  (no articles found)\n")
	} else if !rpt.NLPAvailable {
		fmt.Printf("  NLP scoring unavailable — start FinBERT server for scores\n")
		fmt.Printf("  %-24s NewsAPI(%-3d)  GDELT(%-3d)  Reddit(%-3d)\n",
			"Sources", newsapiN, gdeltN, redditN)
	} else {
		net := rpt.BullishScore - rpt.BearishScore
		netDir := "NEUTRAL"
		if net > 0.1 {
			netDir = "BULLISH"
		} else if net < -0.1 {
			netDir = "BEARISH"
		}
		fmt.Printf("  %-24s %.4f\n", "Bullish", rpt.BullishScore)
		fmt.Printf("  %-24s %.4f\n", "Bearish", rpt.BearishScore)
		fmt.Printf("  %-24s %-12s %s\n", "Net signal",
			fmt.Sprintf("%+.4f", net), netDir)
		fmt.Printf("  %-24s NewsAPI(%-3d)  GDELT(%-3d)  Reddit(%-3d)\n",
			"Sources", newsapiN, gdeltN, redditN)
	}

	fmt.Printf("%s\n\n", sep)
}

// ── interpretation helpers ────────────────────────────────────────────────────

func lambdaNote(λ float64) string {
	switch {
	case λ > 0.01:
		return "HIGH — informed flow moving prices"
	case λ > 0.005:
		return "moderate price impact"
	default:
		return "low price impact per unit flow"
	}
}

func vpinNote(v float64) string {
	switch {
	case v > 0.7:
		return "HIGH — strong informed trading signal"
	case v > 0.5:
		return "elevated — possible informed activity"
	default:
		return "normal — uninformed flow"
	}
}

func hhiNote(h float64) string {
	switch {
	case h > 0.25:
		return "HIGH — whale dominates trading"
	case h > 0.10:
		return "moderate concentration"
	default:
		return "distributed market"
	}
}

func hurstNote(h float64) string {
	switch {
	case h < 0.4:
		return "MEAN-REVERTING"
	case h < 0.45:
		return "slightly mean-reverting"
	case h > 0.6:
		return "TRENDING"
	case h > 0.55:
		return "slightly trending"
	default:
		return "random walk (H ≈ 0.5)"
	}
}

func volNote(r float64) string {
	switch {
	case r > 2.0:
		return "HIGH — elevated vol regime"
	case r > 1.5:
		return "elevated"
	case r < 0.5:
		return "suppressed vol regime"
	default:
		return "normal vol regime"
	}
}

func jumpNote(result string) string {
	switch result {
	case "sustained":
		return "moves tend to follow through"
	case "reversed":
		return "moves tend to snap back"
	default:
		return "no large moves (>3¢) detected"
	}
}

func spreadNote(s float64) string {
	switch {
	case s > 0.05:
		return "WIDE — illiquid market"
	case s > 0.02:
		return "moderate — typical for mid-cap"
	case s > 0:
		return "tight — healthy liquidity"
	default:
		return "n/a"
	}
}

func obiNote(obi float64) string {
	switch {
	case obi > 0.3:
		return "STRONG BUY pressure"
	case obi > 0.1:
		return "mild buy pressure"
	case obi < -0.3:
		return "STRONG SELL pressure"
	case obi < -0.1:
		return "mild sell pressure"
	default:
		return "balanced book"
	}
}
