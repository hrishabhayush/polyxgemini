package polymarket

import (
	"fmt"
	"log"
	"time"

	polymarketrealtime "github.com/ivanzzeth/polymarket-go-real-time-data-client"
	"github.com/shopspring/decimal"
)

// WSClient wraps the Polymarket real-time data client for CLOB market subscriptions.
type WSClient struct {
	client   *polymarketrealtime.Client
	markets  []ResolvedMarket
	tokenIDs []string
	// Maps asset ID -> label like "BTC Up YES" / "BTC Up NO"
	assetLabels map[string]string
}

// NewWSClient creates a WebSocket client for the given resolved markets.
func NewWSClient(markets []ResolvedMarket) *WSClient {
	client := polymarketrealtime.New(
		polymarketrealtime.WithAutoReconnect(true),
		polymarketrealtime.WithPingInterval(10*time.Second),
		polymarketrealtime.WithOnConnect(func() {
			log.Println("polymarket ws: connected")
		}),
		polymarketrealtime.WithOnDisconnect(func(err error) {
			log.Printf("polymarket ws: disconnected: %v", err)
		}),
	)

	var tokenIDs []string
	labels := make(map[string]string)
	for _, m := range markets {
		tokenIDs = append(tokenIDs, m.ClobTokenIDs[0], m.ClobTokenIDs[1])
		short := m.Question
		if len(short) > 30 {
			short = short[:30] + "..."
		}
		labels[m.ClobTokenIDs[0]] = fmt.Sprintf("%s YES", short)
		labels[m.ClobTokenIDs[1]] = fmt.Sprintf("%s NO", short)
	}

	return &WSClient{client: client, markets: markets, tokenIDs: tokenIDs, assetLabels: labels}
}

func (w *WSClient) label(assetID string) string {
	if l, ok := w.assetLabels[assetID]; ok {
		return l
	}
	if len(assetID) > 10 {
		return assetID[:10]
	}
	return assetID
}

// Connect establishes the WebSocket connection and subscribes to CLOB market data.
func (w *WSClient) Connect() error {
	if err := w.client.Connect(); err != nil {
		return err
	}

	filter := polymarketrealtime.NewCLOBMarketFilter(w.tokenIDs...)

	// Subscribe to orderbook updates
	err := w.client.SubscribeToCLOBMarketAggOrderbook(
		filter,
		func(ob polymarketrealtime.AggOrderbook) error {
			bestAsk := decimal.NewFromInt(999)
			for _, a := range ob.Asks {
				if a.Price.LessThan(bestAsk) {
					bestAsk = a.Price
				}
			}
			cents := bestAsk.Mul(decimal.NewFromInt(100))
			log.Printf("[book] %-40s | buy @ %s¢ | asks: %d levels",
				w.label(ob.AssetID), cents.StringFixed(1), len(ob.Asks))
			return nil
		},
	)
	if err != nil {
		return err
	}

	// Subscribe to price changes (skip empty heartbeats)
	err = w.client.SubscribeToCLOBMarketPriceChanges(
		filter,
		func(pc polymarketrealtime.PriceChanges) error {
			if pc.Market == "" || len(pc.PriceChange) == 0 {
				return nil
			}
			for _, c := range pc.PriceChange {
				askCents := c.BestAsk.Mul(decimal.NewFromInt(100))
				log.Printf("[price] %-40s | buy @ %s¢",
					w.label(c.AssetID), askCents.StringFixed(1))
			}
			return nil
		},
	)
	if err != nil {
		return err
	}

	return nil
}

// Close disconnects the WebSocket.
func (w *WSClient) Close() error {
	return w.client.Disconnect()
}
