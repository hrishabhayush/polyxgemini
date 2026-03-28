package polymarket

import (
	"fmt"
	"sync"
	"time"

	polymarketrealtime "github.com/ivanzzeth/polymarket-go-real-time-data-client"
	"github.com/shopspring/decimal"
)

// WSMetrics holds a snapshot of order book metrics from the CLOB WebSocket.
type WSMetrics struct {
	MidPrice       float64
	BidAskSpread   float64
	OBI            float64
	BestBid        float64
	BestAsk        float64
	TotalBidSize   float64
	TotalAskSize   float64
}

// MidPrice returns the mid-price from best bid and ask (0–1 scale).
func MidPrice(bestBid, bestAsk float64) float64 {
	if bestBid <= 0 || bestAsk <= 0 {
		return 0
	}
	return (bestBid + bestAsk) / 2
}

// BidAskSpread returns ask minus bid.
func BidAskSpread(bestBid, bestAsk float64) float64 {
	if bestBid <= 0 || bestAsk <= 0 {
		return 0
	}
	return bestAsk - bestBid
}

// OrderBookImbalance (OBI) is (bidSize - askSize) / (bidSize + askSize).
func OrderBookImbalance(totalBid, totalAsk float64) float64 {
	s := totalBid + totalAsk
	if s == 0 {
		return 0
	}
	return (totalBid - totalAsk) / s
}

// FetchWSMetrics connects to the Polymarket CLOB WebSocket, subscribes to the YES
// token aggregated order book, and returns metrics from the first snapshot within timeout.
func FetchWSMetrics(m ResolvedMarket, timeout time.Duration) (*WSMetrics, error) {
	if len(m.ClobTokenIDs) < 1 || m.ClobTokenIDs[0] == "" {
		return nil, fmt.Errorf("polymarket: market has no YES token id")
	}
	tokenID := m.ClobTokenIDs[0]

	client := polymarketrealtime.New(
		polymarketrealtime.WithAutoReconnect(false),
		polymarketrealtime.WithReadTimeout(timeout+30*time.Second),
	)
	if err := client.Connect(); err != nil {
		return nil, fmt.Errorf("polymarket ws: connect: %w", err)
	}
	defer client.Disconnect()

	var once sync.Once
	result := make(chan *WSMetrics, 1)

	filter := polymarketrealtime.NewCLOBMarketFilter(tokenID)
	err := client.SubscribeToCLOBMarketAggOrderbook(filter, func(ob polymarketrealtime.AggOrderbook) error {
		if ob.AssetID != "" && ob.AssetID != tokenID {
			return nil
		}
		metrics := aggOrderbookToMetrics(ob)
		once.Do(func() { result <- metrics })
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("polymarket ws: subscribe: %w", err)
	}

	select {
	case met := <-result:
		return met, nil
	case <-time.After(timeout):
		return nil, fmt.Errorf("polymarket ws: no orderbook snapshot within %v", timeout)
	}
}

func aggOrderbookToMetrics(ob polymarketrealtime.AggOrderbook) *WSMetrics {
	var bestAsk, bestBid decimal.Decimal
	var hasAsk, hasBid bool
	var totalBid, totalAsk decimal.Decimal

	for _, a := range ob.Asks {
		if !hasAsk || a.Price.LessThan(bestAsk) {
			bestAsk = a.Price
			hasAsk = true
		}
		totalAsk = totalAsk.Add(a.Size)
	}
	for _, b := range ob.Bids {
		if !hasBid || b.Price.GreaterThan(bestBid) {
			bestBid = b.Price
			hasBid = true
		}
		totalBid = totalBid.Add(b.Size)
	}

	tb, _ := totalBid.Float64()
	ta, _ := totalAsk.Float64()

	if !hasAsk || !hasBid {
		return &WSMetrics{
			TotalBidSize: tb,
			TotalAskSize: ta,
		}
	}

	bidF, _ := bestBid.Float64()
	askF, _ := bestAsk.Float64()

	return &WSMetrics{
		MidPrice:       MidPrice(bidF, askF),
		BidAskSpread:   BidAskSpread(bidF, askF),
		OBI:            OrderBookImbalance(tb, ta),
		BestBid:        bidF,
		BestAsk:        askF,
		TotalBidSize:   tb,
		TotalAskSize:   ta,
	}
}
