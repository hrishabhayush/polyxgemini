package report_test

import (
	"context"
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/hrishabhayush/polyxgemini/internal/config"
	"github.com/hrishabhayush/polyxgemini/internal/report"
)

const (
	defaultMarket      = "will-seattle-seahawks-visit-the-white-house-in-2026"
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
