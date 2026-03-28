"""
Pricing-loss risk engine — continuous position sizing via bell curve.

Determines **how many** contracts to hold at any point in a market's lifetime.
The trading engine decides *when* to trade; this engine decides *how much*.

Architecture:
    TradingEngine.tick() → TickResult(action, side)
        ↓
    RiskEngine.evaluate() → OrderDirective(action, delta_qty, target_qty)
        ↓
    PaperPortfolio.on_execute(qty=delta_qty)

Base curve: Beta distribution PDF on [0,1] — zero at both endpoints,
peak tunable via alpha/beta. Default alpha=3.0, beta=2.0 peaks at t_norm≈0.667.

News handler: consolidation freeze → exponential-decay boost multiplied onto base.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CurveConfig:
    alpha: float = 3.0               # ramp-up speed (Beta dist)
    beta: float = 2.0                # ramp-down speed (Beta dist)
    max_contracts: float = 10.0      # peak position at curve mode
    news_boost_max: float = 2.0      # multiplier ceiling on news
    news_decay_halflife_sec: float = 300.0   # 5 min half-life
    consolidation_pause_sec: float = 30.0    # freeze after news
    rebalance_threshold: float = 0.5         # min |delta| to act


@dataclass
class OrderDirective:
    action: str          # "buy" | "reduce" | "hold" | "none"
    delta_qty: float     # contracts to add (positive) or remove (negative)
    target_qty: float    # desired total position
    reason: str = ""


@dataclass
class _NewsEvent:
    magnitude: float
    wall_time: float     # time.monotonic() when news arrived


class RiskEngine:
    """Continuous position-sizing engine driven by a Beta-distribution bell curve."""

    def __init__(self, cfg: CurveConfig | None = None) -> None:
        self.cfg = cfg or CurveConfig()
        # Actual tracked position (absolute contracts, side-agnostic)
        self._actual_qty: float = 0.0
        self._actual_side: str = ""  # "HOME" | "AWAY" | ""
        # News events (kept for boost calculation)
        self._news_events: list[_NewsEvent] = []
        # Pre-compute Beta normaliser so peak of base_curve = 1.0
        self._beta_norm = self._beta_pdf_raw(self._beta_mode())

    # ------------------------------------------------------------------
    # Beta helpers
    # ------------------------------------------------------------------

    def _beta_mode(self) -> float:
        a, b = self.cfg.alpha, self.cfg.beta
        if a <= 1 and b <= 1:
            return 0.5
        return (a - 1.0) / (a + b - 2.0)

    @staticmethod
    def _log_beta(a: float, b: float) -> float:
        return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

    def _beta_pdf_raw(self, t: float) -> float:
        """Un-normalised Beta PDF value (we normalise separately)."""
        a, b = self.cfg.alpha, self.cfg.beta
        if t <= 0.0 or t >= 1.0:
            return 0.0
        log_val = (a - 1.0) * math.log(t) + (b - 1.0) * math.log(1.0 - t) - self._log_beta(a, b)
        return math.exp(log_val)

    # ------------------------------------------------------------------
    # Public curve methods
    # ------------------------------------------------------------------

    def base_curve(self, t_norm: float) -> float:
        """
        Return [0, 1] multiplier from Beta PDF evaluated at t_norm ∈ [0, 1].
        Normalised so peak = 1.0.
        """
        t = max(0.0, min(1.0, t_norm))
        if self._beta_norm <= 0:
            return 0.0
        return self._beta_pdf_raw(t) / self._beta_norm

    def news_boost(self, wall_time: float) -> float:
        """
        Compute multiplicative boost from all active news events.
        Returns ≥ 1.0 (1.0 = no boost).
        """
        if not self._news_events:
            return 1.0

        best_boost = 1.0
        hl = self.cfg.news_decay_halflife_sec
        pause = self.cfg.consolidation_pause_sec

        for ev in self._news_events:
            elapsed = wall_time - ev.wall_time
            if elapsed < 0:
                continue
            # Consolidation freeze: during pause window, boost is 0 (no trading)
            if elapsed < pause:
                continue
            # After consolidation: exponential decay
            decay_elapsed = elapsed - pause
            decay = 0.5 ** (decay_elapsed / hl) if hl > 0 else 0.0
            boost = 1.0 + (self.cfg.news_boost_max - 1.0) * ev.magnitude * decay
            best_boost = max(best_boost, boost)

        return best_boost

    def in_consolidation(self, wall_time: float) -> bool:
        """True if any news event is still within its consolidation freeze."""
        for ev in self._news_events:
            elapsed = wall_time - ev.wall_time
            if 0 <= elapsed < self.cfg.consolidation_pause_sec:
                return True
        return False

    def target_position(self, t_norm: float, wall_time: float) -> float:
        """Desired absolute contract count at this point in time."""
        base = self.base_curve(t_norm)
        boost = self.news_boost(wall_time)
        return base * boost * self.cfg.max_contracts

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        tick_action: str,
        tick_side: str,
        t_norm: float,
        wall_time: float,
    ) -> OrderDirective:
        """
        Called after TradingEngine.tick() returns an execute signal.

        Parameters
        ----------
        tick_action : "execute" | "none"
        tick_side : "HOME" | "AWAY"
        t_norm : normalised game time in [0, 1]
        wall_time : time.monotonic()
        """
        if tick_action != "execute":
            return OrderDirective("none", 0.0, self._actual_qty, reason="no_signal")

        if self.in_consolidation(wall_time):
            return OrderDirective("hold", 0.0, self._actual_qty, reason="consolidation_freeze")

        target = self.target_position(t_norm, wall_time)
        target = max(0.0, target)

        # Side flip → close everything first, then open toward new side
        if self._actual_side and self._actual_side != tick_side.upper() and self._actual_qty > 0:
            delta = -self._actual_qty
            return OrderDirective("reduce", delta, 0.0, reason=f"side_flip_{self._actual_side}_to_{tick_side}")

        delta = target - self._actual_qty
        if abs(delta) < self.cfg.rebalance_threshold:
            return OrderDirective("hold", 0.0, self._actual_qty, reason="within_threshold")

        if delta > 0:
            return OrderDirective("buy", delta, target, reason="curve_ramp")
        else:
            return OrderDirective("reduce", delta, target, reason="curve_trim")

    def check_rebalance(self, t_norm: float, wall_time: float) -> OrderDirective:
        """
        Called every tick (even without execute signal) for curve-driven trimming.
        Only produces reduce orders when the curve has moved below current position.
        """
        if self._actual_qty <= 0:
            return OrderDirective("none", 0.0, 0.0, reason="flat")

        if self.in_consolidation(wall_time):
            return OrderDirective("hold", 0.0, self._actual_qty, reason="consolidation_freeze")

        target = self.target_position(t_norm, wall_time)
        target = max(0.0, target)
        delta = target - self._actual_qty

        # Only trim down (never buy on rebalance — buys come from evaluate())
        if delta >= -self.cfg.rebalance_threshold:
            return OrderDirective("hold", 0.0, self._actual_qty, reason="no_trim_needed")

        return OrderDirective("reduce", delta, target, reason="curve_trim_rebalance")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_news(self, magnitude: float, wall_time: float) -> None:
        """Register a news/edge-shift event. magnitude ∈ [0, 1]."""
        mag = max(0.0, min(1.0, magnitude))
        self._news_events.append(_NewsEvent(magnitude=mag, wall_time=wall_time))
        # Prune old events (>10 min past full decay)
        cutoff = wall_time - (self.cfg.consolidation_pause_sec + self.cfg.news_decay_halflife_sec * 10)
        self._news_events = [e for e in self._news_events if e.wall_time > cutoff]

    def on_fill(self, qty: float, side: str) -> None:
        """Update internal position tracker after a fill."""
        side_up = side.upper()
        if self._actual_side and self._actual_side != side_up:
            # Side flip — reset
            self._actual_qty = abs(qty)
            self._actual_side = side_up if qty > 0 else ""
        else:
            self._actual_qty += qty
            if self._actual_qty <= 0:
                self._actual_qty = 0.0
                self._actual_side = ""
            else:
                self._actual_side = side_up

    # ------------------------------------------------------------------
    # Diagnostic
    # ------------------------------------------------------------------

    def status(self, t_norm: float, wall_time: float) -> dict[str, Any]:
        return {
            "target_qty": round(self.target_position(t_norm, wall_time), 2),
            "actual_qty": round(self._actual_qty, 2),
            "side": self._actual_side or "FLAT",
            "base_curve": round(self.base_curve(t_norm), 4),
            "news_boost": round(self.news_boost(wall_time), 3),
            "in_consolidation": self.in_consolidation(wall_time),
        }
