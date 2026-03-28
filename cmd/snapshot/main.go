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

// ActiveSnapshot captures features for an active (unresolved) market at a point in time.
// outcome is omitted — it will be back-filled once the market resolves.
type ActiveSnapshot struct {
	ConditionID string `json:"condition_id"`
	Slug        string `json:"slug"`
	Question    string `json:"question"`
	CreatedAt   string `json:"created_at"`
	EndDate     string `json:"end_date"`
	SnapshotAt  string `json:"snapshot_at"`

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

	// WebSocket order book
	WSMidPrice     float64 `json:"ws_mid_price"`
	WSBidAskSpread float64 `json:"ws_bid_ask_spread"`
	WSOBI          float64 `json:"ws_obi"`
	WSAvailable    bool    `json:"ws_available"`

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
	minVolume := flag.Float64("min-volume", 1000, "minimum market volume to include")
	maxMarkets := flag.Int("max", 200, "maximum number of active markets to snapshot")
	maxAgeDays := flag.Int("max-age-days", 730, "maximum market age in days to keep (filters stale legacy markets)")
	endWithinHours := flag.Int("end-within-hours", 24*45, "skip markets ending after this many hours (far-future inactive listings)")
	delayMS := flag.Int("delay", 500, "delay between API calls in milliseconds")
	wsTimeout := flag.Duration("ws-timeout", 5*time.Second, "WebSocket collection duration per market")
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
		log.Println("FinBERT server available")
	} else {
		log.Println("FinBERT server unavailable — sentiment scores will be zero")
	}

	markets, err := polyClient.DiscoverMarkets(ctx, true, *minVolume)
	if err != nil {
		log.Fatalf("failed to discover active markets: %v", err)
	}
	usedFallback := false
	if len(markets) > 0 {
		allClosed := true
		for _, m := range markets {
			if !m.Closed {
				allClosed = false
				break
			}
		}
		if allClosed {
			// Over-fetch fallback candidates; downstream filters and no-data checks can drop many.
			target := *maxMarkets * 20
			if target < 100 {
				target = 100
			}
			log.Printf("WARN: active=true returned %d/%d closed markets; falling back to closed=false pagination (target=%d)",
				len(markets), len(markets), target)
			markets, err = polyClient.DiscoverOpenMarkets(ctx, *minVolume, target)
			if err != nil {
				log.Fatalf("fallback open-market discovery failed: %v", err)
			}
			usedFallback = true
		}
	}
	now := time.Now().UTC()
	maxAge := time.Duration(*maxAgeDays) * 24 * time.Hour
	maxEndWindow := time.Duration(*endWithinHours) * time.Hour

	// Gamma active=true can include stale legacy entries; enforce stricter live filters.
	filtered := make([]polymarket.ResolvedMarket, 0, len(markets))
	skippedClosed := 0
	skippedMissingEnd := 0
	skippedExpired := 0
	skippedTooFarEnd := 0
	skippedTooOld := 0
	for _, m := range markets {
		if m.Closed {
			skippedClosed++
			continue
		}
		if m.EndDate.IsZero() {
			skippedMissingEnd++
			// Some open markets arrive without end_date in Gamma; allow them.
		} else {
			if m.EndDate.Before(now) {
				skippedExpired++
				continue
			}
			if m.EndDate.After(now.Add(maxEndWindow)) {
				skippedTooFarEnd++
				continue
			}
		}
		if !m.CreatedAt.IsZero() && m.CreatedAt.Before(now.Add(-maxAge)) {
			// When fallback is active, Gamma "open" rows can have stale created_at.
			// Prefer keeping and letting CLOB data availability decide.
			if !usedFallback {
				skippedTooOld++
				continue
			}
		}
		filtered = append(filtered, m)
	}
	log.Printf("live filter: kept=%d (skipped closed=%d missing_end=%d expired=%d too_far_end=%d too_old=%d)",
		len(filtered), skippedClosed, skippedMissingEnd, skippedExpired, skippedTooFarEnd, skippedTooOld)

	markets = filtered
	if len(markets) > *maxMarkets {
		markets = markets[:*maxMarkets]
	}
	log.Printf("snapshotting %d active markets", len(markets))

	outPath := filepath.Join(*outDir, fmt.Sprintf("active_%s.jsonl", time.Now().Format("20060102_150405")))
	f, err := os.Create(outPath)
	if err != nil {
		log.Fatalf("failed to create output file: %v", err)
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	exported := 0
	delay := time.Duration(*delayMS) * time.Millisecond

	for i, m := range markets {
		snap := snapshotActive(ctx, polyClient, newsClient, gdeltClient, redditClient, finbertClient, m, finbertOK, *wsTimeout)
		if snap.CurrentPrice == 0 && snap.TradeCount == 0 {
			// No CLOB signal available; treat as non-live/noisy and skip output row.
			continue
		}
		if err := enc.Encode(snap); err != nil {
			log.Printf("WARN: encode %s: %v", m.ConditionID, err)
			continue
		}
		exported++

		if (i+1)%10 == 0 {
			log.Printf("progress: %d/%d markets", i+1, len(markets))
		}
		time.Sleep(delay)
	}

	log.Printf("done — exported %d active snapshots to %s", exported, outPath)
}

func snapshotActive(
	ctx context.Context,
	poly *polymarket.Client,
	news *newsapi.Client,
	gdeltC *gdelt.Client,
	redditC *reddit.Client,
	finbertC *finbert.Client,
	m polymarket.ResolvedMarket,
	finbertOK bool,
	wsTimeout time.Duration,
) *ActiveSnapshot {
	snap := &ActiveSnapshot{
		ConditionID: m.ConditionID,
		Slug:        m.Slug,
		Question:    m.Question,
		SnapshotAt:  time.Now().UTC().Format(time.RFC3339),
	}
	if !m.CreatedAt.IsZero() {
		snap.CreatedAt = m.CreatedAt.Format(time.RFC3339)
	}
	if !m.EndDate.IsZero() {
		snap.EndDate = m.EndDate.Format(time.RFC3339)
	}

	// Price dynamics (full history — no truncation for active markets)
	points, err := poly.FetchPriceHistory(ctx, m.ClobTokenIDs[0], "max", 200)
	if err == nil && len(points) > 0 {
		prices := polymarket.Prices(points)
		snap.CurrentPrice = prices[len(prices)-1]
		snap.HurstExp = polymarket.HurstExponent(prices)
		curVol, histVol := polymarket.GARCHVolatility(prices)
		if histVol > 0 {
			snap.VolRatio = curVol / histVol
		}
		snap.JumpResult = polymarket.JumpPersistence(prices, 0.03)
	}

	// Microstructure
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

	// WebSocket order book snapshot (active markets only)
	if wsMetrics, err := polymarket.FetchWSMetrics(m, wsTimeout); err == nil && wsMetrics != nil {
		snap.WSAvailable = true
		snap.WSMidPrice = wsMetrics.MidPrice
		snap.WSBidAskSpread = wsMetrics.BidAskSpread
		snap.WSOBI = wsMetrics.OBI
	}

	// Resolution source
	meta := polymarket.ClassifySource(m.ResolutionSource)
	snap.ResolutionReliability = meta.ReliabilityScore
	snap.ResolutionDomain = meta.Domain

	// Sentiment
	newsapiQ, broadQ := newsKeywords(m.Question)
	var allTitles []string

	if articles, err := news.FetchHeadlines(ctx, newsapiQ, 20); err == nil {
		for _, a := range articles {
			allTitles = append(allTitles, a.Title)
		}
	}
	if articles, err := gdeltC.FetchArticles(ctx, broadQ, 20, "7d"); err == nil {
		for _, a := range articles {
			allTitles = append(allTitles, a.Title)
		}
	}
	if posts, err := redditC.Search(ctx, broadQ, 15, "week"); err == nil {
		for _, p := range posts {
			allTitles = append(allTitles, p.Title)
		}
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

func newsKeywords(question string) (newsapiQ, broadQ string) {
	parts := vsRegexp.Split(question, 2)
	if len(parts) == 2 {
		a := strings.TrimSpace(parts[0])
		b := strings.TrimSpace(parts[1])
		return a + " AND " + b, a + " " + b
	}
	return question, question
}
