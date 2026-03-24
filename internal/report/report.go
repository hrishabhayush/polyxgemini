package report

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/hrishabhayush/polyxgemini/internal/config"
	"github.com/hrishabhayush/polyxgemini/internal/exchange/polymarket"
	"github.com/hrishabhayush/polyxgemini/internal/sentiment"
	"github.com/hrishabhayush/polyxgemini/internal/sentiment/finbert"
	"github.com/hrishabhayush/polyxgemini/internal/sentiment/gdelt"
	"github.com/hrishabhayush/polyxgemini/internal/sentiment/newsapi"
	"github.com/hrishabhayush/polyxgemini/internal/sentiment/reddit"
)

// vsRegexp matches common "versus" separators in market questions.
var vsRegexp = regexp.MustCompile(`(?i)\s+v(?:s\.?)?\s+`)

// MarketReport holds all computed metrics for a single prediction market.
type MarketReport struct {
	MarketName  string
	ConditionID string
	YESToken    string
	FetchedAt   time.Time

	// Microstructure (from trade data)
	TradeCount  int
	TotalVolume float64
	KylesLambda float64
	VPIN        float64
	BuyFraction float64 // Lee-Ready fraction of buyer-initiated trades
	WalletHHI   float64

	// Price dynamics (from CLOB price history)
	CurrentPrice float64
	HurstExp     float64
	VolRatio     float64 // current 7-bar variance / historical variance
	JumpResult   string

	// WebSocket real-time data (from CLOB WebSocket)
	WSMidPrice     float64 // (best_bid + best_ask) / 2 from PriceChanges
	WSBidAskSpread float64 // best_ask - best_bid from AggOrderbook
	WSOBI          float64 // Order Book Imbalance from AggOrderbook
	WSBestBid      float64
	WSBestAsk      float64
	WSTotalBidSize float64
	WSTotalAskSize float64
	WSAvailable    bool

	// Resolution source metadata (Tier 1-3 enrichment)
	ResolutionSource      string
	ResolutionDomain      string
	ResolutionReliability float64
	ResolutionSnippet     string
	ResolutionEnrichStatus string

	// Sentiment (from news sources + optional FinBERT NLP)
	ArticleCount int
	BullishScore float64
	BearishScore float64
	SourceCounts map[string]int // {"newsapi": N, "gdelt": N, "reddit": N}
	NLPAvailable bool
}

// Generator orchestrates Polymarket API calls and sentiment fetching for a single market.
type Generator struct {
	poly    *polymarket.Client
	news    *newsapi.Client
	gdelt   *gdelt.Client
	reddit  *reddit.Client
	finbert *finbert.Client
}

// NewGenerator constructs a Generator from the given configs.
func NewGenerator(polyCfg config.PolymarketConfig, sentCfg config.SentimentConfig) *Generator {
	return &Generator{
		poly:    polymarket.NewClient(polyCfg),
		news:    newsapi.NewClient(sentCfg),
		gdelt:   gdelt.NewClient(sentCfg),
		reddit:  reddit.NewClient(sentCfg),
		finbert: finbert.NewClient(sentCfg),
	}
}

// Generate fetches all metrics for the named market and returns a MarketReport.
// marketName is matched case-insensitively against the Polymarket market question or slug.
func (g *Generator) Generate(ctx context.Context, marketName string) (*MarketReport, error) {
	// 1. Find the market on Polymarket
	market, err := g.poly.FindMarket(ctx, marketName)
	if err != nil {
		return nil, fmt.Errorf("report: %w", err)
	}

	rpt := &MarketReport{
		MarketName:   market.Question,
		ConditionID:  market.ConditionID,
		YESToken:     market.ClobTokenIDs[0],
		FetchedAt:    time.Now().UTC(),
		SourceCounts: make(map[string]int),
	}

	// 2. Microstructure: fetch and compute from trades
	trades, err := g.poly.FetchTrades(ctx, market.ConditionID, 200) //TODO: make this dynamic
	if err != nil {
		return nil, fmt.Errorf("report: fetch trades: %w", err)
	}
	if len(trades) > 0 {
		rpt.TradeCount = len(trades)
		for _, t := range trades {
			rpt.TotalVolume += t.Size
		}
		rpt.KylesLambda = polymarket.KylesLambda(trades)
		if bucketSize := rpt.TotalVolume / 50; bucketSize > 0 {
			rpt.VPIN = polymarket.VPIN(trades, bucketSize)
		}
		rpt.BuyFraction = polymarket.LeeReadyStats(trades)
		rpt.WalletHHI, _ = polymarket.WalletConcentration(trades, 0)
	}

	// 3. Price dynamics: fetch CLOB price history for YES token
	points, err := g.poly.FetchPriceHistory(ctx, market.ClobTokenIDs[0], "max", 100)
	if err == nil && len(points) > 0 {
		prices := polymarket.Prices(points)
		rpt.CurrentPrice = prices[len(prices)-1]
		rpt.HurstExp = polymarket.HurstExponent(prices)
		curVol, histVol := polymarket.GARCHVolatility(prices)
		if histVol > 0 {
			rpt.VolRatio = curVol / histVol
		}
		rpt.JumpResult = polymarket.JumpPersistence(prices, 0.03)
	}
	// Price history failure is non-fatal: fields stay zero

	// 4. Launch all concurrent fetches: WS, news, and resolution enrichment
	var wsMetrics *polymarket.WSMetrics
	var wsErr error
	var resMeta polymarket.SourceMeta
	var wg sync.WaitGroup
	var mu sync.Mutex

	// 4a. WebSocket snapshot
	wg.Add(1)
	go func() {
		defer wg.Done()
		wsMetrics, wsErr = polymarket.FetchWSMetrics(*market, 5*time.Second)
	}()

	// 4b. Resolution source enrichment (Tier 2 inline, Tier 3 async)
	resMeta = polymarket.ClassifySource(market.ResolutionSource)
	wg.Add(1)
	go func() {
		defer wg.Done()
		polymarket.EnrichSource(ctx, &resMeta)
	}()

	// 5. Derive news search keywords from market question
	newsapiQ, broadQ := newsKeywords(market.Question)

	// 6. Fetch articles from all news sources concurrently
	var allTitles []string

	fetch := func(source string, fn func() ([]string, int)) {
		defer wg.Done()
		titles, count := fn()
		mu.Lock()
		rpt.SourceCounts[source] = count
		rpt.ArticleCount += count
		allTitles = append(allTitles, titles...)
		mu.Unlock()
	}

	wg.Add(3)
	go fetch("newsapi", func() ([]string, int) {
		articles, err := g.news.FetchHeadlines(ctx, newsapiQ, 20)
		if err != nil {
			return nil, 0
		}
		titles := make([]string, len(articles))
		for i, a := range articles {
			titles[i] = a.Title
		}
		return titles, len(articles)
	})
	go fetch("gdelt", func() ([]string, int) {
		articles, err := g.gdelt.FetchArticles(ctx, broadQ, 20, "7d")
		if err != nil {
			return nil, 0
		}
		titles := make([]string, len(articles))
		for i, a := range articles {
			titles[i] = a.Title
		}
		return titles, len(articles)
	})
	go fetch("reddit", func() ([]string, int) {
		posts, err := g.reddit.Search(ctx, broadQ, 15, "week")
		if err != nil {
			return nil, 0
		}
		titles := make([]string, len(posts))
		for i, p := range posts {
			titles[i] = p.Title
		}
		return titles, len(posts)
	})
	wg.Wait()

	// 7. Populate WebSocket metrics (non-fatal if WS fails)
	if wsErr == nil && wsMetrics != nil {
		rpt.WSAvailable = true
		rpt.WSMidPrice = wsMetrics.MidPrice
		rpt.WSBidAskSpread = wsMetrics.BidAskSpread
		rpt.WSOBI = wsMetrics.OBI
		rpt.WSBestBid = wsMetrics.BestBid
		rpt.WSBestAsk = wsMetrics.BestAsk
		rpt.WSTotalBidSize = wsMetrics.TotalBidSize
		rpt.WSTotalAskSize = wsMetrics.TotalAskSize
	}

	// 8. Populate resolution source metadata
	rpt.ResolutionSource = resMeta.RawSource
	rpt.ResolutionDomain = resMeta.Domain
	rpt.ResolutionReliability = resMeta.ReliabilityScore
	rpt.ResolutionSnippet = resMeta.ExtractedSnippet
	rpt.ResolutionEnrichStatus = resMeta.EnrichmentStatus

	// 9. Optional NLP scoring via FinBERT (news only)
	if pingErr := g.finbert.Ping(ctx); pingErr == nil {
		rpt.NLPAvailable = true

		// Score news titles
		if len(allTitles) > 0 {
			scores, err := g.finbert.Score(ctx, allTitles)
			if err == nil {
				agg := sentiment.NewAggregator(len(scores))
				for _, s := range scores {
					s.Source = "mixed"
					agg.Add(market.ConditionID, s)
				}
				sig := agg.Signal(market.ConditionID)
				rpt.BullishScore = sig.BullishScore
				rpt.BearishScore = sig.BearishScore
			}
		}

	}

	return rpt, nil
}

// newsKeywords derives search query strings from a market question.
// "Team A vs Team B" → newsapiQ: "Team A AND Team B", broadQ: "Team A Team B"
// If no versus separator is found, both queries equal the full question.
func newsKeywords(question string) (newsapiQ, broadQ string) {
	parts := vsRegexp.Split(question, 2)
	if len(parts) == 2 {
		a := strings.TrimSpace(parts[0])
		b := strings.TrimSpace(parts[1])
		return a + " AND " + b, a + " " + b
	}
	return question, question
}
