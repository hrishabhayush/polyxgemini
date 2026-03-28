"""
Prometheus metrics for the Python trading loop.
Metric names match the Go definitions in internal/metrics/metrics.go
so both code paths feed the same Grafana dashboards.
"""

from prometheus_client import Counter, Gauge, Histogram, start_http_server

# ---- Hedge Layer ----
hedge_t_norm = Gauge(
    "polyxgemini_hedge_t_norm",
    "Normalised game time [0,1].",
)
hedge_curve_value = Gauge(
    "polyxgemini_hedge_curve_value",
    "Current Beta curve value [0,1].",
)
hedge_target_qty = Gauge(
    "polyxgemini_hedge_target_qty",
    "Desired contract count from curve.",
)
hedge_actual_qty = Gauge(
    "polyxgemini_hedge_gemini_qty",
    "Actual paper portfolio contract count.",
)
hedge_realised_pnl = Gauge(
    "polyxgemini_hedge_gemini_realised_pnl",
    "Paper portfolio cumulative realised PnL.",
)
hedge_unrealised_pnl = Gauge(
    "polyxgemini_hedge_gemini_unrealised_pnl",
    "Paper portfolio mark-to-market unrealised PnL.",
)
hedge_combined_pnl = Gauge(
    "polyxgemini_hedge_combined_pnl",
    "Total realised + unrealised PnL.",
)
hedge_poly_pnl = Gauge(
    "polyxgemini_hedge_poly_pnl",
    "Polymarket unrealised PnL tracked by hedge monitor.",
)
hedge_news_active = Gauge(
    "polyxgemini_hedge_news_active",
    "Whether a news event has been detected (1=yes, 0=no).",
)
hedge_actions_total = Counter(
    "polyxgemini_hedge_actions_total",
    "Total hedge actions taken by type.",
    ["action"],
)

# ---- Confidence / ML ----
confidence_score = Gauge(
    "polyxgemini_confidence_score",
    "ML model confidence score (0-1) for a market.",
    ["market", "category"],
)
confidence_inference_seconds = Histogram(
    "polyxgemini_confidence_inference_seconds",
    "Time taken for ML model inference.",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1],
)

# ---- Trading state ----
trading_state = Gauge(
    "polyxgemini_trading_state",
    "Current trading engine state (0=flat,1=sensing,2=armed,3=execute,4=cooldown).",
)
trading_ema_edge = Gauge(
    "polyxgemini_trading_ema_edge",
    "EMA-smoothed edge vs market.",
)
trading_poly_price = Gauge(
    "polyxgemini_trading_poly_price",
    "Current Polymarket price fed to ML.",
)
trading_score_diff = Gauge(
    "polyxgemini_trading_score_diff",
    "Score differential (home - away).",
)
trading_time_remaining = Gauge(
    "polyxgemini_trading_time_remaining_sec",
    "Game clock seconds remaining.",
)

STATE_MAP = {
    "FLAT": 0,
    "SENSING": 1,
    "ARMED": 2,
    "EXECUTE": 3,
    "COOLDOWN": 4,
}


def start_metrics_server(port: int = 9090) -> None:
    """Start the Prometheus /metrics HTTP endpoint."""
    start_http_server(port)
