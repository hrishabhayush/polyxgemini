package polymarket

import (
	"log"
	"time"

	polymarketrealtime "github.com/ivanzzeth/polymarket-go-real-time-data-client"
)

// WSClient wraps the Polymarket real-time data client for CLOB market subscriptions.
type WSClient struct {
	client   *polymarketrealtime.Client
	tokenIDs []string
}

// NewWSClient creates a WebSocket client that will subscribe to the given token IDs.
func NewWSClient(wsURL string, tokenIDs []string) *WSClient {
	client := polymarketrealtime.New(
		polymarketrealtime.WithClobHost(wsURL),
		polymarketrealtime.WithAutoReconnect(true),
		polymarketrealtime.WithPingInterval(10*time.Second),
		polymarketrealtime.WithOnConnect(func() {
			log.Println("polymarket ws: connected")
		}),
		polymarketrealtime.WithOnDisconnect(func(err error) {
			log.Printf("polymarket ws: disconnected: %v", err)
		}),
	)
	return &WSClient{client: client, tokenIDs: tokenIDs}
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
		func(orderbook polymarketrealtime.AggOrderbook) error {
			// TODO: Forward to engine via channel
			log.Printf("polymarket book: market=%s bids=%d asks=%d",
				orderbook.Market, len(orderbook.Bids), len(orderbook.Asks))
			return nil
		},
	)
	if err != nil {
		return err
	}

	// Subscribe to price changes
	err = w.client.SubscribeToCLOBMarketPriceChanges(
		filter,
		func(pc polymarketrealtime.PriceChanges) error {
			// TODO: Forward to engine via channel
			log.Printf("polymarket price_change: market=%s changes=%d",
				pc.Market, len(pc.PriceChange))
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
