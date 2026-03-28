package engine

import (
	"context"
	"log"
	"math"
	"time"

	"github.com/hrishabhayush/polyxgemini/internal/config"
	"github.com/hrishabhayush/polyxgemini/internal/exchange/polymarket"
	"github.com/hrishabhayush/polyxgemini/internal/metrics"
	"github.com/hrishabhayush/polyxgemini/internal/ml"
	"github.com/hrishabhayush/polyxgemini/internal/report"
)

// MLSignal represents a model-predicted edge on a single market.
type MLSignal struct {
	Market   polymarket.ResolvedMarket
	ProbYes  float64
	ProbNo   float64
	EdgeYes  float64
	EdgeNo   float64
	BuySide  string // "YES" or "NO" — the side where the model sees positive edge
	BuyEdge  float64
	Price    float64
}

// MLScanner periodically generates reports for active markets, feeds features
// to the ML prediction server, and logs markets with positive edge.
type MLScanner struct {
	reportGen *report.Generator
	mlClient  *ml.Client
	poly      *polymarket.Client
	cfg       config.EngineConfig
}

func NewMLScanner(
	reportGen *report.Generator,
	mlClient *ml.Client,
	poly *polymarket.Client,
	cfg config.EngineConfig,
) *MLScanner {
	return &MLScanner{
		reportGen: reportGen,
		mlClient:  mlClient,
		poly:      poly,
		cfg:       cfg,
	}
}

// ScanOnce evaluates all provided markets and returns signals with positive edge.
func (s *MLScanner) ScanOnce(ctx context.Context, markets []polymarket.ResolvedMarket) []MLSignal {
	var signals []MLSignal

	for _, m := range markets {
		pred, err := s.evaluateMarket(ctx, m)
		if err != nil {
			log.Printf("[ml] skip %s: %v", m.Slug, err)
			continue
		}
		if pred.BuyEdge >= s.cfg.MinEdge {
			signals = append(signals, *pred)
			log.Printf("[ml] SIGNAL %s: buy %s @ %.2f¢ (model=%.1f%%, edge=+%.1f%%)",
				m.Slug, pred.BuySide, pred.Price*100,
				prob(pred.BuySide, pred)*100, pred.BuyEdge*100)
		}
	}
	return signals
}

// Run starts a polling loop that scans markets at the configured interval.
func (s *MLScanner) Run(ctx context.Context, markets []polymarket.ResolvedMarket) error {
	interval := time.Duration(s.cfg.PollIntervalMS) * time.Millisecond
	if interval <= 0 {
		interval = 60 * time.Second
	}

	log.Printf("[ml] scanner started — polling every %s, min edge %.1f%%",
		interval, s.cfg.MinEdge*100)

	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	// Scan immediately on start, then on each tick.
	s.ScanOnce(ctx, markets)
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			s.ScanOnce(ctx, markets)
		}
	}
}

func (s *MLScanner) evaluateMarket(ctx context.Context, m polymarket.ResolvedMarket) (*MLSignal, error) {
	start := time.Now()

	rpt, err := s.reportGen.Generate(ctx, m.Slug)
	if err != nil {
		return nil, err
	}

	jumpEnc := ml.EncodeJumpResult(rpt.JumpResult)
	ageDays := 0.0
	if !m.CreatedAt.IsZero() && !m.EndDate.IsZero() {
		ageDays = m.EndDate.Sub(m.CreatedAt).Hours() / 24
	}

	req := &ml.PredictRequest{
		CurrentPrice:          rpt.CurrentPrice,
		HurstExp:              rpt.HurstExp,
		VolRatio:              rpt.VolRatio,
		JumpResultEnc:         jumpEnc,
		TradeCount:            rpt.TradeCount,
		TotalVolume:           rpt.TotalVolume,
		LogVolume:             ml.ComputeLogVolume(rpt.TotalVolume),
		KylesLambda:           rpt.KylesLambda,
		VPIN:                  rpt.VPIN,
		BuyFraction:           rpt.BuyFraction,
		WalletHHI:             rpt.WalletHHI,
		ArticleCount:          rpt.ArticleCount,
		BullishScore:          rpt.BullishScore,
		BearishScore:          rpt.BearishScore,
		SentimentNet:          rpt.BullishScore - rpt.BearishScore,
		ResolutionReliability: rpt.ResolutionReliability,
		MarketAgeDays:         ageDays,
		PriceDistanceFrom50:   math.Abs(rpt.CurrentPrice - 0.5),
	}

	pred, err := s.mlClient.Predict(ctx, req)
	if err != nil {
		return nil, err
	}

	elapsed := time.Since(start)
	metrics.ConfidenceInferenceSeconds.Observe(elapsed.Seconds())

	sig := &MLSignal{
		Market:  m,
		ProbYes: pred.ProbYes,
		ProbNo:  pred.ProbNo,
		EdgeYes: pred.EdgeYes,
		EdgeNo:  pred.EdgeNo,
		Price:   rpt.CurrentPrice,
	}

	if pred.EdgeYes >= pred.EdgeNo {
		sig.BuySide = "YES"
		sig.BuyEdge = pred.EdgeYes
	} else {
		sig.BuySide = "NO"
		sig.BuyEdge = pred.EdgeNo
	}

	metrics.ConfidenceScore.WithLabelValues(m.Slug, "prediction").Set(sig.BuyEdge)

	return sig, nil
}

func prob(side string, sig *MLSignal) float64 {
	if side == "YES" {
		return sig.ProbYes
	}
	return sig.ProbNo
}
