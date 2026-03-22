package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/hrishabhayush/polyxgemini/internal/config"
	"github.com/hrishabhayush/polyxgemini/internal/exchange/polymarket"
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
	log.Printf("watchlist loaded: %d markets", len(wl.Markets))

	// Resolve slugs to token IDs via Gamma API
	ctx := context.Background()
	polyClient := polymarket.NewClient(cfg.Polymarket)

	var resolved []polymarket.ResolvedMarket
	for _, entry := range wl.Markets {
		m, err := polyClient.ResolveMarket(ctx, entry.Slug)
		if err != nil {
			log.Printf("WARN: failed to resolve market %q: %v", entry.Slug, err)
			continue
		}
		log.Printf("resolved %q -> %s", m.Question, m.Slug)
		resolved = append(resolved, *m)
	}

	if len(resolved) == 0 {
		log.Fatal("no markets resolved, nothing to subscribe to")
	}

	// Connect WebSocket and subscribe
	ws := polymarket.NewWSClient(resolved)
	if err := ws.Connect(); err != nil {
		log.Fatalf("failed to connect polymarket ws: %v", err)
	}
	defer ws.Close()

	log.Printf("subscribed to %d markets, listening for updates...", len(resolved))

	// Wait for interrupt
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	<-sig
	log.Println("shutting down...")
}
