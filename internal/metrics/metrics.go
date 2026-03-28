package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// ---- WebSocket Health ----

var WSConnected = promauto.NewGaugeVec(prometheus.GaugeOpts{
	Namespace: "polyxgemini",
	Subsystem: "ws",
	Name:      "connected",
	Help:      "Whether the WebSocket connection is alive (1) or dead (0).",
}, []string{"exchange"})

var WSDisconnectsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
	Namespace: "polyxgemini",
	Subsystem: "ws",
	Name:      "disconnects_total",
	Help:      "Total number of WebSocket disconnections.",
}, []string{"exchange"})

var WSMessagesTotal = promauto.NewCounterVec(prometheus.CounterOpts{
	Namespace: "polyxgemini",
	Subsystem: "ws",
	Name:      "messages_total",
	Help:      "Total WebSocket messages received.",
}, []string{"exchange", "type"})

var WSLastMessageTimestamp = promauto.NewGaugeVec(prometheus.GaugeOpts{
	Namespace: "polyxgemini",
	Subsystem: "ws",
	Name:      "last_message_timestamp",
	Help:      "Unix timestamp of the last message received.",
}, []string{"exchange"})

var WSReconnectsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
	Namespace: "polyxgemini",
	Subsystem: "ws",
	Name:      "reconnects_total",
	Help:      "Total number of WebSocket reconnection attempts.",
}, []string{"exchange"})

// ---- Order Latency ----

var OrderRoundtripSeconds = promauto.NewHistogramVec(prometheus.HistogramOpts{
	Namespace: "polyxgemini",
	Subsystem: "order",
	Name:      "roundtrip_seconds",
	Help:      "Round-trip latency from order submission to confirmation.",
	Buckets:   []float64{0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10},
}, []string{"exchange", "side", "outcome"})

var OrderCancelRoundtripSeconds = promauto.NewHistogramVec(prometheus.HistogramOpts{
	Namespace: "polyxgemini",
	Subsystem: "order",
	Name:      "cancel_roundtrip_seconds",
	Help:      "Round-trip latency for order cancellation.",
	Buckets:   []float64{0.05, 0.1, 0.25, 0.5, 1, 2.5, 5},
}, []string{"exchange"})

var OrdersPlacedTotal = promauto.NewCounterVec(prometheus.CounterOpts{
	Namespace: "polyxgemini",
	Subsystem: "order",
	Name:      "placed_total",
	Help:      "Total orders placed.",
}, []string{"exchange", "side", "outcome"})

var OrdersFailedTotal = promauto.NewCounterVec(prometheus.CounterOpts{
	Namespace: "polyxgemini",
	Subsystem: "order",
	Name:      "failed_total",
	Help:      "Total orders that failed.",
}, []string{"exchange", "reason"})

var OrdersFilled = promauto.NewCounterVec(prometheus.CounterOpts{
	Namespace: "polyxgemini",
	Subsystem: "order",
	Name:      "filled_total",
	Help:      "Total orders filled.",
}, []string{"exchange"})

// ---- Arbitrage ----

var ArbSpreadBPS = promauto.NewGaugeVec(prometheus.GaugeOpts{
	Namespace: "polyxgemini",
	Subsystem: "arb",
	Name:      "spread_bps",
	Help:      "Current arbitrage spread in basis points per market pair.",
}, []string{"pair", "direction"})

var ArbOpportunitiesDetected = promauto.NewCounterVec(prometheus.CounterOpts{
	Namespace: "polyxgemini",
	Subsystem: "arb",
	Name:      "opportunities_detected_total",
	Help:      "Total arbitrage opportunities detected.",
}, []string{"pair"})

var ArbOpportunitiesExecuted = promauto.NewCounterVec(prometheus.CounterOpts{
	Namespace: "polyxgemini",
	Subsystem: "arb",
	Name:      "opportunities_executed_total",
	Help:      "Total arbitrage opportunities executed.",
}, []string{"pair"})

var ArbRoundtripSeconds = promauto.NewHistogramVec(prometheus.HistogramOpts{
	Namespace: "polyxgemini",
	Subsystem: "arb",
	Name:      "roundtrip_seconds",
	Help:      "Full arb round-trip: detect spread, place both legs, confirm or cancel.",
	Buckets:   []float64{0.1, 0.25, 0.5, 1, 2, 5, 10},
}, []string{"pair"})

// ---- Capital & PnL ----

var CapitalDeployedUSD = promauto.NewGaugeVec(prometheus.GaugeOpts{
	Namespace: "polyxgemini",
	Subsystem: "capital",
	Name:      "deployed_usd",
	Help:      "Current capital deployed in USD.",
}, []string{"exchange"})

var CapitalTotalUSD = promauto.NewGauge(prometheus.GaugeOpts{
	Namespace: "polyxgemini",
	Subsystem: "capital",
	Name:      "total_usd",
	Help:      "Total capital across all exchanges in USD.",
})

var CapitalRatio = promauto.NewGauge(prometheus.GaugeOpts{
	Namespace: "polyxgemini",
	Subsystem: "capital",
	Name:      "gemini_polymarket_ratio",
	Help:      "Ratio of Gemini capital to Polymarket capital. Used for rebalancing triggers.",
})

var PnLRealizedUSD = promauto.NewCounterVec(prometheus.CounterOpts{
	Namespace: "polyxgemini",
	Subsystem: "pnl",
	Name:      "realized_usd",
	Help:      "Cumulative realized PnL in USD.",
}, []string{"pair"})

var PnLUnrealizedUSD = promauto.NewGaugeVec(prometheus.GaugeOpts{
	Namespace: "polyxgemini",
	Subsystem: "pnl",
	Name:      "unrealized_usd",
	Help:      "Current unrealized PnL in USD.",
}, []string{"pair"})

// ---- Fees ----

var FeesTotalUSD = promauto.NewCounterVec(prometheus.CounterOpts{
	Namespace: "polyxgemini",
	Subsystem: "fees",
	Name:      "total_usd",
	Help:      "Cumulative fees paid in USD.",
}, []string{"exchange"})

// ---- Active Positions ----

var ActivePositions = promauto.NewGaugeVec(prometheus.GaugeOpts{
	Namespace: "polyxgemini",
	Subsystem: "positions",
	Name:      "active",
	Help:      "Number of active positions (contracts held).",
}, []string{"exchange", "pair", "outcome"})

var ActiveTickers = promauto.NewGauge(prometheus.GaugeOpts{
	Namespace: "polyxgemini",
	Subsystem: "positions",
	Name:      "active_tickers",
	Help:      "Number of unique market pairs the bot is actively trading.",
})

// ---- Market Data ----

var BookDepthLevels = promauto.NewGaugeVec(prometheus.GaugeOpts{
	Namespace: "polyxgemini",
	Subsystem: "book",
	Name:      "depth_levels",
	Help:      "Number of price levels in the order book.",
}, []string{"exchange", "market", "side"})

var BookBestBid = promauto.NewGaugeVec(prometheus.GaugeOpts{
	Namespace: "polyxgemini",
	Subsystem: "book",
	Name:      "best_bid_cents",
	Help:      "Best bid price in cents.",
}, []string{"exchange", "market"})

var BookBestAsk = promauto.NewGaugeVec(prometheus.GaugeOpts{
	Namespace: "polyxgemini",
	Subsystem: "book",
	Name:      "best_ask_cents",
	Help:      "Best ask price in cents.",
}, []string{"exchange", "market"})

var PriceStalenessSeconds = promauto.NewGaugeVec(prometheus.GaugeOpts{
	Namespace: "polyxgemini",
	Subsystem: "book",
	Name:      "staleness_seconds",
	Help:      "Seconds since the last price update for a market.",
}, []string{"exchange", "market"})

// ---- Confidence Model ----

var ConfidenceScore = promauto.NewGaugeVec(prometheus.GaugeOpts{
	Namespace: "polyxgemini",
	Subsystem: "confidence",
	Name:      "score",
	Help:      "ML model confidence score (0-1) for a market.",
}, []string{"market", "category"})

var ConfidenceInferenceSeconds = promauto.NewHistogram(prometheus.HistogramOpts{
	Namespace: "polyxgemini",
	Subsystem: "confidence",
	Name:      "inference_seconds",
	Help:      "Time taken for ML model inference.",
	Buckets:   []float64{0.01, 0.05, 0.1, 0.25, 0.5, 1},
})

var NewsSourceLastFetch = promauto.NewGaugeVec(prometheus.GaugeOpts{
	Namespace: "polyxgemini",
	Subsystem: "news",
	Name:      "last_fetch_timestamp",
	Help:      "Unix timestamp of last successful fetch from a news source.",
}, []string{"source"})

var NewsSourceErrors = promauto.NewCounterVec(prometheus.CounterOpts{
	Namespace: "polyxgemini",
	Subsystem: "news",
	Name:      "errors_total",
	Help:      "Total errors fetching from a news source.",
}, []string{"source"})

// ---- Slippage ----

var SlippageBPS = promauto.NewHistogramVec(prometheus.HistogramOpts{
	Namespace: "polyxgemini",
	Subsystem: "order",
	Name:      "slippage_bps",
	Help:      "Slippage in basis points (actual fill vs expected price).",
	Buckets:   []float64{1, 5, 10, 25, 50, 100, 250, 500},
}, []string{"exchange"})

// ---- Fill Rate ----

var FillRate = promauto.NewGaugeVec(prometheus.GaugeOpts{
	Namespace: "polyxgemini",
	Subsystem: "order",
	Name:      "fill_rate",
	Help:      "Ratio of filled orders to placed orders (0-1).",
}, []string{"exchange"})

// ---- Hedge Layer ----

var HedgeTNorm = promauto.NewGauge(prometheus.GaugeOpts{
	Namespace: "polyxgemini",
	Subsystem: "hedge",
	Name:      "t_norm",
	Help:      "Normalised game time [0,1] used by the hedge monitor.",
})

var HedgeCurveValue = promauto.NewGauge(prometheus.GaugeOpts{
	Namespace: "polyxgemini",
	Subsystem: "hedge",
	Name:      "curve_value",
	Help:      "Current Gemini Beta curve value [0,1].",
})

var HedgeTargetQty = promauto.NewGauge(prometheus.GaugeOpts{
	Namespace: "polyxgemini",
	Subsystem: "hedge",
	Name:      "target_qty",
	Help:      "Desired Gemini hedge contract count from curve.",
})

var HedgeGeminiQty = promauto.NewGauge(prometheus.GaugeOpts{
	Namespace: "polyxgemini",
	Subsystem: "hedge",
	Name:      "gemini_qty",
	Help:      "Actual Gemini paper portfolio contract count.",
})

var HedgeGeminiRealised = promauto.NewGauge(prometheus.GaugeOpts{
	Namespace: "polyxgemini",
	Subsystem: "hedge",
	Name:      "gemini_realised_pnl",
	Help:      "Gemini paper portfolio cumulative realised PnL.",
})

var HedgeGeminiUnrealised = promauto.NewGauge(prometheus.GaugeOpts{
	Namespace: "polyxgemini",
	Subsystem: "hedge",
	Name:      "gemini_unrealised_pnl",
	Help:      "Gemini paper portfolio mark-to-market unrealised PnL.",
})

var HedgeCombinedPnL = promauto.NewGauge(prometheus.GaugeOpts{
	Namespace: "polyxgemini",
	Subsystem: "hedge",
	Name:      "combined_pnl",
	Help:      "Total PnL across Poly + Gemini positions.",
})

var HedgePolyPnL = promauto.NewGauge(prometheus.GaugeOpts{
	Namespace: "polyxgemini",
	Subsystem: "hedge",
	Name:      "poly_pnl",
	Help:      "Polymarket unrealised PnL tracked by hedge monitor.",
})

var HedgeNewsActive = promauto.NewGauge(prometheus.GaugeOpts{
	Namespace: "polyxgemini",
	Subsystem: "hedge",
	Name:      "news_active",
	Help:      "Whether a news event has been detected (1=yes, 0=no).",
})

var HedgeActionsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
	Namespace: "polyxgemini",
	Subsystem: "hedge",
	Name:      "actions_total",
	Help:      "Total hedge actions taken by type.",
}, []string{"action"})
