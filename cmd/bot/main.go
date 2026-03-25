package main

import (
	"context"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"

	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/hrishabhayush/polyxgemini/internal/config"
	"github.com/hrishabhayush/polyxgemini/internal/exchange/gemini"
	"github.com/hrishabhayush/polyxgemini/internal/exchange/polymarket"
	_ "github.com/hrishabhayush/polyxgemini/internal/metrics"
)

func main() {
	configPath := flag.String("config", "config/config.yaml", "path to config file")
	flag.Parse()

	// Load config
	cfg, err := config.Load(*configPath)
	if err != nil {
		log.Fatalf("failed to load config: %v", err)
	}
	log.Printf("config loaded from %s", *configPath)

	// Load watchlist
	wl, err := config.LoadWatchlist(cfg.Polymarket.WatchlistPath)
	if err != nil {
		log.Fatalf("failed to load watchlist: %v", err)
	}
	log.Printf("watchlist loaded: %d polymarket, %d gemini markets",
		len(wl.Markets), len(wl.GeminiMarkets))

	ctx := context.Background()

	// --- Polymarket ---
	polyClient := polymarket.NewClient(cfg.Polymarket)
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

	if len(polyResolved) > 0 {
		polyWS := polymarket.NewWSClient(polyResolved)
		if err := polyWS.Connect(); err != nil {
			log.Printf("WARN: polymarket ws failed: %v", err)
		} else {
			defer polyWS.Close()
			log.Printf("[poly] subscribed to %d markets", len(polyResolved))
		}
	}

	// --- Gemini ---
	geminiClient := gemini.NewClient(cfg.Gemini)
	var geminiResolved []gemini.ResolvedEvent
	for _, entry := range wl.GeminiMarkets {
		e, err := geminiClient.ResolveEvent(ctx, entry.Ticker)
		if err != nil {
			log.Printf("WARN: failed to resolve gemini %q: %v", entry.Ticker, err)
			continue
		}
		log.Printf("[gemi] resolved %q -> %d contracts", e.Title, len(e.Contracts))
		for _, c := range e.Contracts {
			log.Printf("[gemi]   %s (%s) ask=%s¢", c.Label, c.InstrumentSymbol, c.BestAsk)
		}
		geminiResolved = append(geminiResolved, *e)
	}

	if len(geminiResolved) > 0 {
		geminiWS := gemini.NewWSClient(cfg.Gemini.WSURL, geminiResolved)
		if err := geminiWS.Connect(); err != nil {
			log.Printf("WARN: gemini ws failed: %v", err)
		} else {
			defer geminiWS.Close()
			log.Printf("[gemi] subscribed to %d events", len(geminiResolved))
		}
	}

	if len(polyResolved) == 0 && len(geminiResolved) == 0 {
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

	log.Println("listening for updates... (Ctrl+C to stop)")

	// Wait for interrupt
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	<-sig
	log.Println("shutting down...")
}
