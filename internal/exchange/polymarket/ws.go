package polymarket

import (
	"fmt"
	"log"
	"sync"
	"time"

	polymarketrealtime "github.com/ivanzzeth/polymarket-go-real-time-data-client"
	"github.com/shopspring/decimal"
)

// WSMetrics holds real-time microstructure metrics derived from CLOB WebSocket data.
type WSMetrics struct {
	MidPrice     float64 // (best_bid + best_ask) / 2
	BidAskSpread float64 // best_ask - best_bid
	OBI          float64 // Order Book Imbalance: (bid_size - ask_size) / (bid_size + ask_size)
	BestBid      float64
	BestAsk      float64
	TotalBidSize float64
	TotalAskSize float64
}

// MidPrice computes the mid-price from best bid and best ask.
func MidPrice(bestBid, bestAsk float64) float64 {
	if bestBid <= 0 || bestAsk <= 0 {
		return 0
	}
	return (bestBid + bestAsk) / 2
}

// BidAskSpread computes the absolute bid-ask spread.
func BidAskSpread(bestBid, bestAsk float64) float64 {
	if bestBid <= 0 || bestAsk <= 0 {
		return 0
	}
	return bestAsk - bestBid
}

// OrderBookImbalance computes OBI from total bid and ask volumes.
// Range: -1.0 (all selling pressure) to +1.0 (all buying pressure).
func OrderBookImbalance(totalBidSize, totalAskSize float64) float64 {
	total := totalBidSize + totalAskSize
	if total == 0 {
		return 0
	}
	return (totalBidSize - totalAskSize) / total
}

// FetchWSMetrics creates a temporary WebSocket connection, subscribes to
// AggOrderbook and PriceChanges for the YES token of the given market,
// collects data for collectDuration, computes metrics, and disconnects.
func FetchWSMetrics(market ResolvedMarket, collectDuration time.Duration) (*WSMetrics, error) {
	var mu sync.Mutex
	var latestOB *polymarketrealtime.AggOrderbook
	var latestBid, latestAsk decimal.Decimal

	gotData := make(chan struct{}, 1)
	yesToken := market.ClobTokenIDs[0]

	client := polymarketrealtime.New(
		polymarketrealtime.WithAutoReconnect(false),
		polymarketrealtime.WithPingInterval(10*time.Second),
	)

	if err := client.Connect(); err != nil {
		return nil, fmt.Errorf("ws: connect: %w", err)
	}
	defer client.Disconnect()

	filter := polymarketrealtime.NewCLOBMarketFilter(yesToken, market.ClobTokenIDs[1])

	err := client.SubscribeToCLOBMarketAggOrderbook(
		filter,
		func(ob polymarketrealtime.AggOrderbook) error {
			if ob.AssetID == yesToken {
				mu.Lock()
				cp := ob
				latestOB = &cp
				mu.Unlock()
				select {
				case gotData <- struct{}{}:
				default:
				}
			}
			return nil
		},
	)
	if err != nil {
		return nil, fmt.Errorf("ws: subscribe orderbook: %w", err)
	}

	err = client.SubscribeToCLOBMarketPriceChanges(
		filter,
		func(pc polymarketrealtime.PriceChanges) error {
			if pc.Market == "" || len(pc.PriceChange) == 0 {
				return nil
			}
			for _, c := range pc.PriceChange {
				if c.AssetID == yesToken {
					mu.Lock()
					latestBid = c.BestBid
					latestAsk = c.BestAsk
					mu.Unlock()
					select {
					case gotData <- struct{}{}:
					default:
					}
					break
				}
			}
			return nil
		},
	)
	if err != nil {
		return nil, fmt.Errorf("ws: subscribe price changes: %w", err)
	}

	time.Sleep(collectDuration)

	mu.Lock()
	defer mu.Unlock()

	metrics := &WSMetrics{}

	pcBid, _ := latestBid.Float64()
	pcAsk, _ := latestAsk.Float64()
	if pcBid > 0 && pcAsk > 0 {
		metrics.BestBid = pcBid
		metrics.BestAsk = pcAsk
		metrics.MidPrice = MidPrice(pcBid, pcAsk)
	}

	if latestOB != nil {
		obBestBid := decimal.Zero
		for _, b := range latestOB.Bids {
			if b.Price.GreaterThan(obBestBid) {
				obBestBid = b.Price
			}
		}
		obBestAsk := decimal.New(999, 0)
		for _, a := range latestOB.Asks {
			if a.Price.LessThan(obBestAsk) {
				obBestAsk = a.Price
			}
		}

		bid, _ := obBestBid.Float64()
		ask, _ := obBestAsk.Float64()

		if bid > 0 && ask < 999 {
			metrics.BidAskSpread = BidAskSpread(bid, ask)
			if metrics.MidPrice == 0 {
				metrics.BestBid = bid
				metrics.BestAsk = ask
				metrics.MidPrice = MidPrice(bid, ask)
			}
		}

		var totalBid, totalAsk float64
		for _, b := range latestOB.Bids {
			s, _ := b.Size.Float64()
			totalBid += s
		}
		for _, a := range latestOB.Asks {
			s, _ := a.Size.Float64()
			totalAsk += s
		}
		metrics.TotalBidSize = totalBid
		metrics.TotalAskSize = totalAsk
		metrics.OBI = OrderBookImbalance(totalBid, totalAsk)
	}

	return metrics, nil
}

// ── Streaming WSClient (used by the live bot) ────────────────────────────────

// WSClient wraps the Polymarket real-time data client for CLOB market subscriptions.
type WSClient struct {
	client   *polymarketrealtime.Client
	markets  []ResolvedMarket
	tokenIDs []string
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
			log.Printf("[POLY book]  %-40s | buy @ %s¢ | asks: %d levels",
				w.label(ob.AssetID), cents.StringFixed(1), len(ob.Asks))
			return nil
		},
	)
	if err != nil {
		return err
	}

	err = w.client.SubscribeToCLOBMarketPriceChanges(
		filter,
		func(pc polymarketrealtime.PriceChanges) error {
			if pc.Market == "" || len(pc.PriceChange) == 0 {
				return nil
			}
			for _, c := range pc.PriceChange {
				askCents := c.BestAsk.Mul(decimal.NewFromInt(100))
				log.Printf("[POLY price] %-40s | buy @ %s¢",
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
