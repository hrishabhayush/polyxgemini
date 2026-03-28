package main

import (
	"context"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/hrishabhayush/polyxgemini/internal/arb"
	"github.com/hrishabhayush/polyxgemini/internal/budget"
	"github.com/hrishabhayush/polyxgemini/internal/config"
	"github.com/hrishabhayush/polyxgemini/internal/engine"
	"github.com/hrishabhayush/polyxgemini/internal/exchange/gemini"
	"github.com/hrishabhayush/polyxgemini/internal/exchange/polymarket"
	"github.com/hrishabhayush/polyxgemini/internal/ml"
	"github.com/hrishabhayush/polyxgemini/internal/report"
)

func main() {
	configPath := flag.String("config", "config/config.yaml", "path to config file")
	flag.Parse()

	cfg, err := config.Load(*configPath)
	if err != nil {
		log.Fatalf("failed to load config: %v", err)
	}
	log.Printf("config loaded from %s", *configPath)

	go func() {
		mux := http.NewServeMux()
		mux.Handle("/metrics", promhttp.Handler())
		log.Printf("[metrics] serving Prometheus on :9090/metrics")
		if err := http.ListenAndServe(":9090", mux); err != nil {
			log.Fatalf("metrics server: %v", err)
		}
	}()

	wl, err := config.LoadWatchlist(cfg.Polymarket.WatchlistPath)
	if err != nil {
		log.Fatalf("failed to load watchlist: %v", err)
	}
	log.Printf("watchlist loaded: %d polymarket, %d gemini markets",
		len(wl.Markets), len(wl.GeminiMarkets))

	ctx := context.Background()
	botCtx, botCancel := context.WithCancel(ctx)
	defer botCancel()

	maxUSD := cfg.Engine.MaxPositionUSD
	if maxUSD <= 0 {
		maxUSD = 1e12
	}
	bud := budget.New(maxUSD, botCancel)

	polyClient := polymarket.NewClient(cfg.Polymarket)
	geminiClient := gemini.NewClient(cfg.Gemini)

	detector := arb.NewDetector(1024, bud, cfg.Engine.DryRun, polyClient, geminiClient)
	updates := detector.Updates()

	var polyResolved []polymarket.ResolvedMarket
	for _, entry := range wl.Markets {
		m, err := polyClient.ResolveMarket(ctx, entry.Slug)
		if err != nil {
			log.Printf("WARN: failed to resolve polymarket %q: %v", entry.Slug, err)
			continue
		}
		log.Printf("[poly] resolved %q -> %s", m.Question, m.Slug)
		polyResolved = append(polyResolved, *m)
	}

	var allPolyMarkets []polymarket.ResolvedMarket
	var allGeminiEvents []gemini.ResolvedEvent
	var polyPairMappings []polymarket.PairMapping
	var geminiPairMappings []gemini.PairMapping

	for _, entry := range wl.GeminiMarkets {
		e, err := geminiClient.ResolveEvent(ctx, entry.Ticker)
		if err != nil {
			log.Printf("WARN: failed to resolve gemini %q: %v", entry.Ticker, err)
			continue
		}
		log.Printf("[gemi] resolved %q -> %d contracts", e.Title, len(e.Contracts))
		allGeminiEvents = append(allGeminiEvents, *e)
	}

	for _, pair := range wl.Pairs {
		log.Printf("[pair] resolving %q...", pair.Name)

		polyM, err := polyClient.ResolveMarket(ctx, pair.PolymarketSlug)
		if err != nil {
			log.Printf("WARN: pair %q: polymarket resolve failed: %v", pair.Name, err)
			continue
		}
		allPolyMarkets = append(allPolyMarkets, *polyM)

		geminiE, err := geminiClient.ResolveEvent(ctx, pair.GeminiTicker)
		if err != nil {
			log.Printf("WARN: pair %q: gemini resolve failed: %v", pair.Name, err)
			continue
		}
		allGeminiEvents = append(allGeminiEvents, *geminiE)

		polyPairMappings = append(polyPairMappings, polymarket.PairMapping{
			PairID:     pair.Name,
			YesTokenID: polyM.ClobTokenIDs[0],
			NoTokenID:  polyM.ClobTokenIDs[1],
			Category:   pair.Category,
		})

		mids := &arb.MarketIDs{
			PolyYesTokenID: polyM.ClobTokenIDs[0],
			PolyNoTokenID:  polyM.ClobTokenIDs[1],
		}

		for _, om := range pair.Mapping {
			for _, c := range geminiE.Contracts {
				if strings.EqualFold(c.Label, om.GeminiLabel) {
					outcome := om.PolyOutcome
					geminiPairMappings = append(geminiPairMappings, gemini.PairMapping{
						PairID:           pair.Name,
						InstrumentSymbol: c.InstrumentSymbol,
						Outcome:          outcome,
						Category:         pair.Category,
					})
					if outcome == "yes" {
						mids.GeminiYesSymbol = c.InstrumentSymbol
					} else {
						mids.GeminiNoSymbol = c.InstrumentSymbol
					}
				}
			}
		}

		mappedSymbols := make(map[string]bool)
		for _, gpm := range geminiPairMappings {
			if gpm.PairID == pair.Name {
				mappedSymbols[gpm.InstrumentSymbol] = true
			}
		}
		for _, c := range geminiE.Contracts {
			if !mappedSymbols[c.InstrumentSymbol] {
				opposite := "no"
				for _, om := range pair.Mapping {
					if om.PolyOutcome == "no" {
						opposite = "yes"
					}
				}
				geminiPairMappings = append(geminiPairMappings, gemini.PairMapping{
					PairID:           pair.Name,
					InstrumentSymbol: c.InstrumentSymbol,
					Outcome:          opposite,
					Category:         pair.Category,
				})
				if opposite == "yes" && mids.GeminiYesSymbol == "" {
					mids.GeminiYesSymbol = c.InstrumentSymbol
				} else if opposite == "no" && mids.GeminiNoSymbol == "" {
					mids.GeminiNoSymbol = c.InstrumentSymbol
				}
			}
		}

		detector.RegisterPair(pair.Name, mids)

		log.Printf("[pair] %q: poly=%s, gemini=%s (%d contracts)",
			pair.Name, polyM.Slug, geminiE.Ticker, len(geminiE.Contracts))
	}

	go detector.Run()

	var hedgeMonitor *engine.HedgeMonitor
	if cfg.Hedge.Mode != "" && len(allGeminiEvents) > 0 {
		hedgeCfg := engine.HedgeConfig{
			Mode:                 cfg.Hedge.Mode,
			CurveAlpha:           cfg.Hedge.CurveAlpha,
			CurveBeta:            cfg.Hedge.CurveBeta,
			MaxHedgeQty:          cfg.Hedge.MaxHedgeQty,
			LossThresholdPct:     cfg.Hedge.LossThresholdPct,
			MildLossThresholdPct: cfg.Hedge.MildLossThresholdPct,
			MaxTotalExposure:     cfg.Hedge.MaxTotalExposure,
			GeminiHalfSpread:     cfg.Hedge.GeminiHalfSpread,
			RebalanceThreshold:   cfg.Hedge.RebalanceThreshold,
			PollIntervalMS:       cfg.Hedge.PollIntervalMS,
		}
		gameDurMin := cfg.Hedge.GameDurationMin
		if gameDurMin <= 0 {
			gameDurMin = 40
		}
		hedgeMonitor = engine.NewHedgeMonitor(hedgeCfg, time.Duration(gameDurMin)*time.Minute)
	}

	if hedgeMonitor != nil && len(polyPairMappings) > 0 && len(geminiPairMappings) > 0 {
		pm := polyPairMappings[0]
		var gemSymbols []string
		for _, gm := range geminiPairMappings {
			if gm.PairID == pm.PairID {
				gemSymbols = append(gemSymbols, gm.InstrumentSymbol)
			}
		}
		hedgeMonitor.SetActivePair(engine.PairBookMapping{
			PairName:      pm.PairID,
			PolyTokenIDs:  []string{pm.YesTokenID, pm.NoTokenID},
			GeminiSymbols: gemSymbols,
		})
	}

	polyWSMarkets := mergeResolvedMarkets(polyResolved, allPolyMarkets)
	if len(polyWSMarkets) > 0 {
		polyWS := polymarket.NewWSClient(polyWSMarkets, updates, polyPairMappings)
		if hedgeMonitor != nil {
			polyWS.SetBookHook(hedgeMonitor.UpdatePolyBook)
		}
		if err := polyWS.Connect(); err != nil {
			log.Printf("WARN: polymarket ws failed: %v", err)
		} else {
			defer polyWS.Close()
			log.Printf("[poly] subscribed to %d markets", len(polyWSMarkets))
		}
	}

	geminiWSMarkets := dedupeGeminiEvents(allGeminiEvents)
	if len(geminiWSMarkets) > 0 {
		geminiWS := gemini.NewWSClient(cfg.Gemini, geminiWSMarkets, updates, geminiPairMappings)
		if hedgeMonitor != nil {
			geminiWS.SetBookHook(hedgeMonitor.UpdateGeminiBook)
		}
		if err := geminiWS.Connect(); err != nil {
			log.Printf("WARN: gemini ws failed: %v", err)
		} else {
			defer geminiWS.Close()
			log.Printf("[gemi] subscribed to %d events", len(geminiWSMarkets))
		}
	}

	if hedgeMonitor != nil {
		go hedgeMonitor.Run()
		defer hedgeMonitor.Stop()
	}

	if len(polyWSMarkets) == 0 && len(geminiWSMarkets) == 0 {
		log.Fatal("no markets resolved on either exchange")
	}

	scanCtx, scanCancel := context.WithCancel(botCtx)
	defer scanCancel()

	if cfg.Engine.MLServerURL != "" && len(polyResolved) > 0 {
		mlClient := ml.NewClient(cfg.Engine.MLServerURL)
		if err := mlClient.Ping(ctx); err != nil {
			log.Printf("[ml] prediction server not available at %s: %v", cfg.Engine.MLServerURL, err)
		} else {
			reportGen := report.NewGenerator(cfg.Polymarket, cfg.Sentiment)
			scanner := engine.NewMLScanner(reportGen, mlClient, polyClient, cfg.Engine)
			go func() {
				if err := scanner.Run(scanCtx, polyResolved); err != nil && scanCtx.Err() == nil {
					log.Printf("[ml] scanner stopped: %v", err)
				}
			}()
			log.Printf("[ml] scanner running for %d markets", len(polyResolved))
		}
	}

	log.Println("listening for updates... (Ctrl+C to stop)")

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	<-sig
	scanCancel()
	log.Println("shutting down...")
}

func mergeResolvedMarkets(a, b []polymarket.ResolvedMarket) []polymarket.ResolvedMarket {
	seen := make(map[string]struct{})
	out := make([]polymarket.ResolvedMarket, 0, len(a)+len(b))
	for _, m := range append(append([]polymarket.ResolvedMarket{}, a...), b...) {
		if m.ConditionID == "" {
			continue
		}
		if _, ok := seen[m.ConditionID]; ok {
			continue
		}
		seen[m.ConditionID] = struct{}{}
		out = append(out, m)
	}
	return out
}

func dedupeGeminiEvents(in []gemini.ResolvedEvent) []gemini.ResolvedEvent {
	seen := make(map[string]struct{})
	out := make([]gemini.ResolvedEvent, 0, len(in))
	for _, e := range in {
		if e.Ticker == "" {
			continue
		}
		if _, ok := seen[e.Ticker]; ok {
			continue
		}
		seen[e.Ticker] = struct{}{}
		out = append(out, e)
	}
	return out
}
