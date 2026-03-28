package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/hrishabhayush/polyxgemini/internal/config"
	"github.com/hrishabhayush/polyxgemini/internal/exchange/polymarket"
	"github.com/hrishabhayush/polyxgemini/internal/sentiment"
	"github.com/hrishabhayush/polyxgemini/internal/sentiment/finbert"
	"github.com/hrishabhayush/polyxgemini/internal/sentiment/gdelt"
	"github.com/hrishabhayush/polyxgemini/internal/sentiment/newsapi"
	"github.com/hrishabhayush/polyxgemini/internal/sentiment/reddit"
)

var vsRegexp = regexp.MustCompile(`(?i)\s+v(?:s\.?)?\s+`)

// MarketSnapshot is the JSONL row exported for each market.
type MarketSnapshot struct {
	ConditionID string  `json:"condition_id"`
	Slug        string  `json:"slug"`
	Question    string  `json:"question"`
	Outcome     string  `json:"outcome"`
	CreatedAt   string  `json:"created_at"`
	EndDate     string  `json:"end_date"`
	SnapshotAt  string  `json:"snapshot_at"`
	TruncPct    float64 `json:"truncation_pct"`

	// Price dynamics
	CurrentPrice float64 `json:"current_price"`
	HurstExp     float64 `json:"hurst_exp"`
	VolRatio     float64 `json:"vol_ratio"`
	JumpResult   string  `json:"jump_result"`

	// Microstructure
	TradeCount  int     `json:"trade_count"`
	TotalVolume float64 `json:"total_volume"`
	KylesLambda float64 `json:"kyles_lambda"`
	VPIN        float64 `json:"vpin"`
	BuyFraction float64 `json:"buy_fraction"`
	WalletHHI   float64 `json:"wallet_hhi"`

	// Sentiment
	ArticleCount int     `json:"article_count"`
	BullishScore float64 `json:"bullish_score"`
	BearishScore float64 `json:"bearish_score"`
	NLPAvailable bool    `json:"nlp_available"`

	// Resolution source
	ResolutionReliability float64 `json:"resolution_reliability"`
	ResolutionDomain      string  `json:"resolution_domain"`
}

func main() {
	configPath := flag.String("config", "config/config.yaml", "path to config file")
	outDir := flag.String("out", "data/snapshots", "output directory for JSONL files")
	truncPct := flag.Float64("truncate", 0.75, "fraction of price history to keep (avoids label leakage)")
	maxMarkets := flag.Int("max", 500, "maximum number of resolved markets to export")
	pageSize := flag.Int("page-size", 100, "Gamma API page size")
	delayMS := flag.Int("delay", 500, "delay between API calls in milliseconds")
	lookbackDays := flag.Int("lookback", 365, "only export markets that resolved within this many days (0 = no limit)")
	sentimentWindow := flag.String("sentiment-window", "1m", "GDELT/Reddit lookback window for sentiment (e.g. 7d, 1m, 3m, 6m, 1y)")
	flag.Parse()

	cfg, err := config.Load(*configPath)
	if err != nil {
		log.Fatalf("failed to load config: %v", err)
	}

	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		log.Fatalf("failed to create output dir: %v", err)
	}

	ctx := context.Background()
	polyClient := polymarket.NewClient(cfg.Polymarket)
	newsClient := newsapi.NewClient(cfg.Sentiment)
	gdeltClient := gdelt.NewClient(cfg.Sentiment)
	redditClient := reddit.NewClient(cfg.Sentiment)
	finbertClient := finbert.NewClient(cfg.Sentiment)

	finbertOK := finbertClient.Ping(ctx) == nil
	if finbertOK {
		log.Println("FinBERT server available — will score headlines")
	} else {
		log.Println("FinBERT server unavailable — sentiment scores will be zero")
	}

	// Compute the earliest end date we'll accept for CLOB data availability.
	var closedAfter time.Time
	if *lookbackDays > 0 {
		closedAfter = time.Now().UTC().AddDate(0, 0, -*lookbackDays)
		log.Printf("restricting to markets resolved after %s", closedAfter.Format("2006-01-02"))
	}

	// Paginate through resolved markets
	var allMarkets []polymarket.ResolvedMarket
	for offset := 0; len(allMarkets) < *maxMarkets; offset += *pageSize {
		batch, err := polyClient.DiscoverResolvedMarkets(ctx, *pageSize, offset, closedAfter)
		if err != nil {
			log.Printf("WARN: failed to fetch page at offset %d: %v", offset, err)
			break
		}
		if len(batch) == 0 {
			break
		}
		allMarkets = append(allMarkets, batch...)
		log.Printf("fetched %d resolved markets (total: %d)", len(batch), len(allMarkets))
		time.Sleep(time.Duration(*delayMS) * time.Millisecond)
	}

	if len(allMarkets) > *maxMarkets {
		allMarkets = allMarkets[:*maxMarkets]
	}
	log.Printf("exporting features for %d resolved markets", len(allMarkets))

	outPath := filepath.Join(*outDir, fmt.Sprintf("resolved_%s.jsonl", time.Now().Format("20060102")))
	f, err := os.Create(outPath)
	if err != nil {
		log.Fatalf("failed to create output file: %v", err)
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	exported := 0
	delay := time.Duration(*delayMS) * time.Millisecond

	skipped := 0
	for i, m := range allMarkets {
		snap := computeSnapshot(ctx, polyClient, newsClient, gdeltClient, redditClient, finbertClient, m, *truncPct, finbertOK, *sentimentWindow)
		if snap == nil {
			continue
		}
		// Skip rows with no price data — CLOB API returned nothing, all features are zero.
		// Training on these rows would introduce noise with no signal.
		if snap.CurrentPrice == 0 && snap.TradeCount == 0 && snap.HurstExp == 0 {
			skipped++
			log.Printf("SKIP %s: no price/trade data available", m.ConditionID)
			continue
		}
		if err := enc.Encode(snap); err != nil {
			log.Printf("WARN: failed to encode market %s: %v", m.ConditionID, err)
			continue
		}
		exported++

		if (i+1)%10 == 0 {
			log.Printf("progress: %d/%d markets (exported: %d, skipped: %d)", i+1, len(allMarkets), exported, skipped)
		}
		time.Sleep(delay)
	}
	log.Printf("skipped %d markets with no CLOB data", skipped)

	log.Printf("done — exported %d snapshots to %s", exported, outPath)
}

func computeSnapshot(
	ctx context.Context,
	poly *polymarket.Client,
	news *newsapi.Client,
	gdeltC *gdelt.Client,
	redditC *reddit.Client,
	finbertC *finbert.Client,
	m polymarket.ResolvedMarket,
	truncPct float64,
	finbertOK bool,
	sentimentWindow string,
) *MarketSnapshot {
	snap := &MarketSnapshot{
		ConditionID: m.ConditionID,
		Slug:        m.Slug,
		Question:    m.Question,
		Outcome:     m.Outcome,
		SnapshotAt:  time.Now().UTC().Format(time.RFC3339),
		TruncPct:    truncPct,
	}
	if !m.CreatedAt.IsZero() {
		snap.CreatedAt = m.CreatedAt.Format(time.RFC3339)
	}
	if !m.EndDate.IsZero() {
		snap.EndDate = m.EndDate.Format(time.RFC3339)
	}

	// Price dynamics: fetch full history, truncate to avoid label leakage
	points, err := poly.FetchPriceHistory(ctx, m.ClobTokenIDs[0], "max", 200)
	if err == nil && len(points) > 0 {
		cutoff := int(float64(len(points)) * truncPct)
		if cutoff < 2 {
			cutoff = 2
		}
		if cutoff > len(points) {
			cutoff = len(points)
		}
		truncated := polymarket.Prices(points[:cutoff])
		snap.CurrentPrice = truncated[len(truncated)-1]
		snap.HurstExp = polymarket.HurstExponent(truncated)
		curVol, histVol := polymarket.GARCHVolatility(truncated)
		if histVol > 0 {
			snap.VolRatio = curVol / histVol
		}
		snap.JumpResult = polymarket.JumpPersistence(truncated, 0.03)
	}

	// Microstructure from trades
	trades, err := poly.FetchTrades(ctx, m.ConditionID, 200)
	if err == nil && len(trades) > 0 {
		snap.TradeCount = len(trades)
		for _, t := range trades {
			snap.TotalVolume += t.Size
		}
		snap.KylesLambda = polymarket.KylesLambda(trades)
		if bs := snap.TotalVolume / 50; bs > 0 {
			snap.VPIN = polymarket.VPIN(trades, bs)
		}
		snap.BuyFraction = polymarket.LeeReadyStats(trades)
		snap.WalletHHI, _ = polymarket.WalletConcentration(trades, 0)
	}

	// Resolution source metadata
	meta := polymarket.ClassifySource(m.ResolutionSource)
	snap.ResolutionReliability = meta.ReliabilityScore
	snap.ResolutionDomain = meta.Domain

	// Sentiment from news sources
	newsapiQ, broadQ := newsKeywords(m.Question)
	var allTitles []string

	if articles, err := news.FetchHeadlines(ctx, newsapiQ, 20); err == nil {
		for _, a := range articles {
			allTitles = append(allTitles, a.Title)
		}
	} else {
		log.Printf("WARN [%s] newsapi: %v", m.ConditionID[:8], err)
	}
	if articles, err := gdeltC.FetchArticles(ctx, broadQ, 20, sentimentWindow); err == nil {
		for _, a := range articles {
			allTitles = append(allTitles, a.Title)
		}
	} else {
		log.Printf("WARN [%s] gdelt: %v", m.ConditionID[:8], err)
	}
	// Reddit t= param: map GDELT-style "1m"/"3m" to Reddit's "month"/"year"
	redditPeriod := gdeltToRedditPeriod(sentimentWindow)
	if posts, err := redditC.Search(ctx, broadQ, 15, redditPeriod); err == nil {
		for _, p := range posts {
			allTitles = append(allTitles, p.Title)
		}
	} else {
		log.Printf("WARN [%s] reddit: %v", m.ConditionID[:8], err)
	}

	snap.ArticleCount = len(allTitles)

	if finbertOK && len(allTitles) > 0 {
		scores, err := finbertC.Score(ctx, allTitles)
		if err == nil {
			snap.NLPAvailable = true
			agg := sentiment.NewAggregator(len(scores))
			for _, s := range scores {
				s.Source = "mixed"
				agg.Add(m.ConditionID, s)
			}
			sig := agg.Signal(m.ConditionID)
			snap.BullishScore = sig.BullishScore
			snap.BearishScore = sig.BearishScore
		}
	}

	return snap
}

// gdeltToRedditPeriod maps GDELT timespan strings to Reddit's t= parameter values.
func gdeltToRedditPeriod(gdeltSpan string) string {
	switch gdeltSpan {
	case "7d", "":
		return "week"
	case "1m":
		return "month"
	case "3m", "6m":
		return "year"
	case "1y":
		return "year"
	default:
		return "month"
	}
}

func newsKeywords(question string) (newsapiQ, broadQ string) {
	parts := vsRegexp.Split(question, 2)
	if len(parts) == 2 {
		a := strings.TrimSpace(parts[0])
		b := strings.TrimSpace(parts[1])
		return a + " AND " + b, a + " " + b
	}
	return question, question
}
