package main

import (
	"context"
	"flag"
	"log"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"strings"
	"syscall"

	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/hrishabhayush/polyxgemini/internal/arb"
	"github.com/hrishabhayush/polyxgemini/internal/budget"
	"github.com/hrishabhayush/polyxgemini/internal/config"
	"github.com/hrishabhayush/polyxgemini/internal/exchange/gemini"
	"github.com/hrishabhayush/polyxgemini/internal/exchange/polymarket"
	_ "github.com/hrishabhayush/polyxgemini/internal/metrics"
)

func main() {
	configPath := flag.String("config", "config/config.yaml", "path to config file")
	flag.Parse()

	cfg, err := config.Load(*configPath)
	if err != nil {
		log.Fatalf("failed to load config: %v", err)
	}
	log.Printf("config loaded from %s", *configPath)

	// Validate API keys when running live
	if !cfg.Engine.DryRun {
		if cfg.Polymarket.APIKey == "" || cfg.Polymarket.APISecret == "" {
			log.Fatal("polymarket api_key and api_secret are required when dry_run is false")
		}
		if cfg.Gemini.APIKey == "" || cfg.Gemini.APISecret == "" {
			log.Fatal("gemini api_key and api_secret are required when dry_run is false")
		}
		log.Println("[MODE] LIVE — real orders will be placed")
	} else {
		log.Println("[MODE] DRY-RUN — no real orders will be placed")
	}

	wl, err := config.LoadWatchlist(cfg.Polymarket.WatchlistPath)
	if err != nil {
		log.Fatalf("failed to load watchlist: %v", err)
	}
	log.Printf("watchlist: %d polymarket, %d gemini, %d pairs",
		len(wl.Markets), len(wl.GeminiMarkets), len(wl.Pairs))

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	polyClient := polymarket.NewClient(cfg.Polymarket)
	geminiClient := gemini.NewClient(cfg.Gemini)

	// Budget: $10 global spending cap
	budgetTracker := budget.New(10.0, cancel)

	// Arb detector
	detector := arb.NewDetector(256, budgetTracker, cfg.Engine.DryRun, polyClient, geminiClient)
	updates := detector.Updates()

	// Collect all polymarket markets to subscribe to (standalone + pairs)
	var allPolyMarkets []polymarket.ResolvedMarket
	var polyPairMappings []polymarket.PairMapping

	// Standalone polymarket markets
	for _, entry := range wl.Markets {
		m, err := polyClient.ResolveMarket(ctx, entry.Slug)
		if err != nil {
			log.Printf("WARN: failed to resolve polymarket %q: %v", entry.Slug, err)
			continue
		}
		log.Printf("[poly] resolved %q -> %s", m.Question, m.Slug)
		allPolyMarkets = append(allPolyMarkets, *m)
	}

	// Collect all gemini markets (standalone + pairs)
	var allGeminiEvents []gemini.ResolvedEvent
	var geminiPairMappings []gemini.PairMapping

	// Standalone gemini markets
	for _, entry := range wl.GeminiMarkets {
		e, err := geminiClient.ResolveEvent(ctx, entry.Ticker)
		if err != nil {
			log.Printf("WARN: failed to resolve gemini %q: %v", entry.Ticker, err)
			continue
		}
		log.Printf("[gemi] resolved %q -> %d contracts", e.Title, len(e.Contracts))
		allGeminiEvents = append(allGeminiEvents, *e)
	}

	// Resolve pairs
	for _, pair := range wl.Pairs {
		log.Printf("[pair] resolving %q...", pair.Name)

		// Resolve polymarket side
		polyM, err := polyClient.ResolveMarket(ctx, pair.PolymarketSlug)
		if err != nil {
			log.Printf("WARN: pair %q: polymarket resolve failed: %v", pair.Name, err)
			continue
		}
		allPolyMarkets = append(allPolyMarkets, *polyM)

		// Resolve gemini side
		geminiE, err := geminiClient.ResolveEvent(ctx, pair.GeminiTicker)
		if err != nil {
			log.Printf("WARN: pair %q: gemini resolve failed: %v", pair.Name, err)
			continue
		}
		allGeminiEvents = append(allGeminiEvents, *geminiE)

		// Build mappings from the pair config
		// poly YES token = ClobTokenIDs[0], NO = ClobTokenIDs[1]
		polyPairMappings = append(polyPairMappings, polymarket.PairMapping{
			PairID:     pair.Name,
			YesTokenID: polyM.ClobTokenIDs[0],
			NoTokenID:  polyM.ClobTokenIDs[1],
			Category:   pair.Category,
		})

		// Build MarketIDs for the arb detector
		mids := &arb.MarketIDs{
			PolyYesTokenID: polyM.ClobTokenIDs[0],
			PolyNoTokenID:  polyM.ClobTokenIDs[1],
		}

		// Map gemini contracts to outcomes based on the mapping config
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

		// Map remaining gemini contracts as the opposite outcome
		mappedSymbols := make(map[string]bool)
		for _, gpm := range geminiPairMappings {
			if gpm.PairID == pair.Name {
				mappedSymbols[gpm.InstrumentSymbol] = true
			}
		}
		for _, c := range geminiE.Contracts {
			if !mappedSymbols[c.InstrumentSymbol] {
				// This contract wasn't explicitly mapped — it's the opposite
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

		// Register market IDs with the arb detector
		detector.RegisterPair(pair.Name, mids)

		log.Printf("[pair] %q: poly=%s, gemini=%s (%d contracts)",
			pair.Name, polyM.Slug, geminiE.Ticker, len(geminiE.Contracts))
	}

	// Start arb detector
	go detector.Run()

	// Start Polymarket WS
	if len(allPolyMarkets) > 0 {
		polyWS := polymarket.NewWSClient(allPolyMarkets, updates, polyPairMappings)
		if err := polyWS.Connect(); err != nil {
			log.Printf("WARN: polymarket ws failed: %v", err)
		} else {
			defer polyWS.Close()
			log.Printf("[poly] subscribed to %d markets", len(allPolyMarkets))
		}
	}

	// Start Gemini WS
	if len(allGeminiEvents) > 0 {
		geminiWS := gemini.NewWSClient(cfg.Gemini, allGeminiEvents, updates, geminiPairMappings)
		if err := geminiWS.Connect(); err != nil {
			log.Printf("WARN: gemini ws failed: %v", err)
		} else {
			defer geminiWS.Close()
			log.Printf("[gemi] subscribed to %d events", len(allGeminiEvents))
		}
	}

	if len(allPolyMarkets) == 0 && len(allGeminiEvents) == 0 {
		log.Fatal("no markets resolved on either exchange")
	}

	// Prometheus metrics endpoint
	go func() {
		http.Handle("/metrics", promhttp.Handler())
		log.Println("metrics server listening on :9090")
		if err := http.ListenAndServe(":9090", nil); err != nil {
			log.Printf("metrics server error: %v", err)
		}
	}()

	// Start Prometheus + Grafana via docker-compose
	startObservability()

	log.Println("listening for updates... (Ctrl+C to stop)")

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	select {
	case <-sig:
		log.Println("shutting down (signal)...")
	case <-ctx.Done():
		log.Printf("shutting down (budget exhausted: spent $%.2f)...", budgetTracker.Spent())
	}

	stopObservability()
}

func startObservability() {
	cmd := exec.Command("docker", "compose", "up", "-d")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		log.Printf("WARN: docker compose up failed: %v (dashboards won't be available)", err)
		return
	}
	log.Println("[observability] prometheus + grafana started (grafana at http://localhost:3000)")
}

func stopObservability() {
	log.Println("[observability] stopping prometheus + grafana...")
	cmd := exec.Command("docker", "compose", "down")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		log.Printf("WARN: docker compose down failed: %v", err)
	}
}
