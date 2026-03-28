"""
Prometheus metrics exporter for the NCAA live trading loop.

Starts an HTTP server on the given port (default 9200) exposing /metrics
for Prometheus to scrape.  All gauges are updated once per live-loop
iteration via update().
"""

from __future__ import annotations

from typing import Any

from prometheus_client import Gauge, Enum, start_http_server

_NAMESPACE = "ncaa"

# ---- Model ----
model_prob_home = Gauge(
    f"{_NAMESPACE}_model_prob_home",
    "ML model P(home win)",
)
model_prob_away = Gauge(
    f"{_NAMESPACE}_model_prob_away",
    "ML model P(away win)",
)
model_edge = Gauge(
    f"{_NAMESPACE}_model_edge",
    "ML model edge vs market (signed)",
)
model_side = Enum(
    f"{_NAMESPACE}_model_side",
    "ML model recommended side",
    states=["HOME", "AWAY", "UNKNOWN"],
)

# ---- Engine ----
engine_state = Enum(
    f"{_NAMESPACE}_engine_state",
    "Trading engine state machine state",
    states=["IDLE", "WATCHING", "ARMED", "EXECUTE", "COOLDOWN", "OFF"],
)
engine_ema_edge = Gauge(
    f"{_NAMESPACE}_engine_ema_edge",
    "EMA-smoothed edge from trading engine",
)
engine_epsilon_eff = Gauge(
    f"{_NAMESPACE}_engine_epsilon_eff",
    "Effective epsilon threshold",
)
engine_norm_lead = Gauge(
    f"{_NAMESPACE}_engine_norm_lead",
    "Normalised lead (score_diff / sigma)",
)
engine_persistence = Gauge(
    f"{_NAMESPACE}_engine_persistence_ticks",
    "Current persistence tick count",
)

# ---- Paper Portfolio ----
paper_position = Enum(
    f"{_NAMESPACE}_paper_position",
    "Paper portfolio position side",
    states=["FLAT", "HOME_YES", "AWAY_YES"],
)
paper_qty = Gauge(
    f"{_NAMESPACE}_paper_qty",
    "Paper portfolio open contract count",
)
paper_realised_pnl = Gauge(
    f"{_NAMESPACE}_paper_realised_pnl_usd",
    "Paper portfolio cumulative realised PnL (USD)",
)
paper_unrealised_pnl = Gauge(
    f"{_NAMESPACE}_paper_unrealised_pnl_usd",
    "Paper portfolio mark-to-market unrealised PnL (USD)",
)
paper_entry_price = Gauge(
    f"{_NAMESPACE}_paper_entry_price",
    "Paper portfolio VWAP entry price",
)

# ---- Risk Engine ----
risk_target_qty = Gauge(
    f"{_NAMESPACE}_risk_target_qty",
    "Risk engine target contract quantity from curve",
)
risk_actual_qty = Gauge(
    f"{_NAMESPACE}_risk_actual_qty",
    "Risk engine actual contract quantity",
)
risk_curve_value = Gauge(
    f"{_NAMESPACE}_risk_curve_value",
    "Risk engine base curve value [0,1]",
)
risk_news_boost = Gauge(
    f"{_NAMESPACE}_risk_news_boost",
    "Risk engine news boost multiplier",
)

# ---- Game Clock ----
game_time_remaining = Gauge(
    f"{_NAMESPACE}_game_time_remaining_sec",
    "Seconds remaining in the game",
)
game_period = Gauge(
    f"{_NAMESPACE}_game_period",
    "Current game period (quarter/half)",
)

# ---- Market ----
market_poly_price = Gauge(
    f"{_NAMESPACE}_market_poly_price",
    "Current Polymarket YES price",
)

_started = False


def start(port: int = 9200) -> None:
    """Start the Prometheus HTTP server (idempotent)."""
    global _started
    if _started:
        return
    start_http_server(port)
    _started = True


def update(
    *,
    snapshot: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    tick_result: Any | None = None,
    portfolio: Any | None = None,
    risk_status: dict[str, Any] | None = None,
) -> None:
    """Push one iteration's worth of data into Prometheus gauges."""

    if snapshot:
        game_time_remaining.set(float(snapshot.get("time_remaining_sec", 0)))
        game_period.set(float(snapshot.get("period", 0)))

    if payload:
        market_poly_price.set(float(payload.get("poly_price", 0)))

    if result:
        model_prob_home.set(float(result.get("prob_home_win", 0)))
        model_prob_away.set(float(result.get("prob_away_win", 0)))
        model_edge.set(float(result.get("edge_vs_market", 0)))
        side = result.get("model_side", "UNKNOWN").upper()
        if side not in ("HOME", "AWAY"):
            side = "UNKNOWN"
        model_side.state(side)

    if tick_result is not None:
        state_val = getattr(tick_result, "state", None)
        if state_val is not None:
            engine_state.state(state_val.value if hasattr(state_val, "value") else str(state_val))
        engine_ema_edge.set(float(getattr(tick_result, "ema_edge", 0)))
        engine_epsilon_eff.set(float(getattr(tick_result, "epsilon_eff", 0)))
        engine_norm_lead.set(float(getattr(tick_result, "normalised_lead", 0)))
        engine_persistence.set(float(getattr(tick_result, "persistence_ticks", 0)))
    else:
        engine_state.state("OFF")

    if portfolio is not None:
        pos = "FLAT"
        if getattr(portfolio, "position", None) == "HOME":
            pos = "HOME_YES"
        elif getattr(portfolio, "position", None) == "AWAY":
            pos = "AWAY_YES"
        paper_position.state(pos)
        paper_qty.set(float(getattr(portfolio, "qty", 0)))
        paper_realised_pnl.set(float(getattr(portfolio, "realised_pnl", 0)))
        paper_unrealised_pnl.set(float(getattr(portfolio, "unrealised_pnl", 0)))
        ep = getattr(portfolio, "entry_price", 0)
        paper_entry_price.set(float(ep) if ep else 0.0)

    if risk_status:
        risk_target_qty.set(float(risk_status.get("target_qty", 0)))
        risk_actual_qty.set(float(risk_status.get("actual_qty", 0)))
        risk_curve_value.set(float(risk_status.get("base_curve", 0)))
        risk_news_boost.set(float(risk_status.get("news_boost", 0)))
