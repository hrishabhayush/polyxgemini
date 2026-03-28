"""
Basketball live-trading decision engine.

Implements the three-layer architecture from the strategy document:
  Signal (model + market) → Decision rule (filters) → Execution trigger

Layers inside the engine:
  1. Normalised lead  (score_diff / sigma)
  2. Regime gate      (foul storm, end-of-quarter, early blowout, timeout cluster)
  3. EMA-smoothed edge
  4. K-tick persistence counter
  5. Five-state machine  IDLE → WATCHING → ARMED → EXECUTE → COOLDOWN
  6. Time-scaled epsilon threshold with hysteresis

Usage:
  engine = TradingEngine(interval_sec=5, ...)
  for each tick:
      result = engine.tick(snapshot, payload, predict_result, events)
      # result.state, result.action, result.log_entry
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

class State(str, Enum):
    IDLE = "IDLE"
    WATCHING = "WATCHING"
    ARMED = "ARMED"
    EXECUTE = "EXECUTE"
    COOLDOWN = "COOLDOWN"


# ---------------------------------------------------------------------------
# Tick result returned to the caller
# ---------------------------------------------------------------------------

@dataclass
class TickResult:
    state: State
    action: str  # "none" | "execute"
    ema_edge: float
    normalised_lead: float
    dead_zone: str  # "" when clean, else reason
    persistence_ticks: int
    epsilon_eff: float
    log_entry: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Sigma lookup — empirical score-diff standard deviation by game state
# From Section 3.1 of the strategy document, with interpolation for gaps.
# Keys: (quarter, seconds_remaining_in_quarter).
# We linearly interpolate within each quarter.
# ---------------------------------------------------------------------------

_SIGMA_ANCHORS: list[tuple[int, float, float]] = [
    # (quarter, seconds_left_in_quarter, sigma_pts)
    (1, 600.0, 16.0),
    (1, 0.0, 14.0),
    (2, 600.0, 14.0),
    (2, 300.0, 12.0),
    (2, 0.0, 10.0),
    (3, 600.0, 12.0),
    (3, 0.0, 10.0),
    (4, 600.0, 9.0),
    (4, 180.0, 7.5),
    (4, 45.0, 3.0),
    (4, 0.0, 2.0),
]

# Pre-index by quarter for fast lookup
_SIGMA_BY_Q: dict[int, list[tuple[float, float]]] = {}
for _q, _s, _sig in _SIGMA_ANCHORS:
    _SIGMA_BY_Q.setdefault(_q, []).append((_s, _sig))
for _q in _SIGMA_BY_Q:
    _SIGMA_BY_Q[_q].sort(key=lambda x: -x[0])  # descending by seconds


def sigma_at(quarter: int, time_remaining_sec: float, regulation_secs: float = 2400.0) -> float:
    """Return empirical sigma for a game state. Interpolates within the quarter anchors."""
    if quarter < 1:
        quarter = 1

    # For men's halves, map half → quarter equivalent
    # Half 1 (1200–2400 s remaining) → Q1/Q2; Half 2 (0–1200 s) → Q3/Q4
    # The caller already passes period; for two-half games period=1 maps to ~Q1/Q2.
    # We keep it simple: if quarter > 4 (OT), use Q4 anchors.
    q = min(quarter, 4)

    # Seconds remaining in this quarter (approximate for interpolation)
    if q in _SIGMA_BY_Q:
        anchors = _SIGMA_BY_Q[q]
    else:
        anchors = _SIGMA_BY_Q[4]

    # time_remaining_in_quarter: for a 4-quarter game, each quarter is 600 s.
    # For a 2-half game each half is 1200 s; we normalise to 600 s equivalent.
    quarter_len = 600.0
    if regulation_secs <= 2400.0 and quarter <= 2 and regulation_secs >= 2000:
        # Two-half format: half is 1200 s; map remaining within the half to 0–600 for anchors
        quarter_len = 1200.0

    secs_in_q = time_remaining_sec % quarter_len if quarter_len > 0 else time_remaining_sec
    secs_in_q = max(0.0, min(secs_in_q, 600.0))

    # Interpolate
    if len(anchors) == 1:
        return anchors[0][1]
    for i in range(len(anchors) - 1):
        s_hi, sig_hi = anchors[i]
        s_lo, sig_lo = anchors[i + 1]
        if secs_in_q >= s_lo:
            if s_hi == s_lo:
                return sig_hi
            frac = (secs_in_q - s_lo) / (s_hi - s_lo)
            return sig_lo + frac * (sig_hi - sig_lo)
    return anchors[-1][1]


def normalised_lead(score_diff: float, sigma: float) -> float:
    if sigma <= 0:
        return 0.0
    return score_diff / sigma


# ---------------------------------------------------------------------------
# Regime gate helpers
# ---------------------------------------------------------------------------

_FOUL_RE = re.compile(r"\b(foul|personal|technical|flagrant)\b", re.IGNORECASE)
_TIMEOUT_RE = re.compile(r"\b(timeout|time.?out)\b", re.IGNORECASE)


def _detect_foul_storm(
    events: list[dict],
    time_remaining_sec: float,
    foul_count: int = 3,
    foul_window_sec: float = 120.0,
    proximity_sec: float = 90.0,
) -> bool:
    """N+ fouls within a sliding game-clock window near current time."""
    foul_times: list[float] = []
    for ev in events:
        if _FOUL_RE.search(ev.get("event_text", "")):
            foul_times.append(ev["time_remaining_sec"])
    foul_times.sort(reverse=True)
    for i in range(len(foul_times)):
        window_start = foul_times[i]
        window_end = window_start - foul_window_sec
        count = 0
        for ft in foul_times[i:]:
            if ft >= window_end:
                count += 1
            else:
                break
        if count >= foul_count and window_start >= time_remaining_sec >= max(window_end - proximity_sec, 0.0):
            return True
    return False


def _detect_timeout_cluster(
    events: list[dict],
    time_remaining_sec: float,
    gap_sec: float = 90.0,
    proximity_sec: float = 90.0,
) -> bool:
    """Back-to-back timeouts within gap_sec of game clock near current time."""
    to_times: list[float] = []
    for ev in events:
        if _TIMEOUT_RE.search(ev.get("event_text", "")):
            to_times.append(ev["time_remaining_sec"])
    to_times.sort(reverse=True)
    for i in range(len(to_times) - 1):
        gap = to_times[i] - to_times[i + 1]
        if gap <= gap_sec:
            if to_times[i] >= time_remaining_sec >= to_times[i + 1] - proximity_sec:
                return True
    return False


def _end_of_quarter(
    period: int,
    time_remaining_sec: float,
    regulation_secs: float = 2400.0,
    window_sec: float = 60.0,
) -> bool:
    """Last window_sec seconds of any quarter (or half for men's)."""
    if regulation_secs >= 2000 and period <= 2:
        quarter_len = 1200.0  # half
    else:
        quarter_len = 600.0
    secs_in_q = time_remaining_sec % quarter_len if quarter_len > 0 else time_remaining_sec
    return secs_in_q <= window_sec


def _early_blowout(
    period: int,
    score_diff: int,
    time_remaining_sec: float,
    regulation_secs: float = 2400.0,
    blowout_q1: int = 10,
    blowout_q2: int = 18,
) -> bool:
    """Lead exceeds threshold for current quarter."""
    ad = abs(score_diff)
    if regulation_secs >= 2000 and period <= 2:
        # Men's halves: period 1 ≈ Q1+Q2; map by remaining time
        if time_remaining_sec > 1200:
            return ad >= blowout_q1
        return ad >= blowout_q2

    if period == 1:
        return ad >= blowout_q1
    if period == 2:
        return ad >= blowout_q2
    return False


def _has_recent_disruptions(
    events: list[dict],
    time_remaining_sec: float,
    window_sec: float,
) -> bool:
    """Check if any fouls or timeouts occurred within window_sec of current clock."""
    for ev in events:
        text = ev.get("event_text", "")
        if _FOUL_RE.search(text) or _TIMEOUT_RE.search(text):
            ev_time = ev["time_remaining_sec"]
            if abs(ev_time - time_remaining_sec) <= window_sec:
                return True
    return False


# ---------------------------------------------------------------------------
# Trading Engine
# ---------------------------------------------------------------------------

@dataclass
class DeadZoneConfig:
    """Configurable thresholds for regime gate dead zones."""
    # End-of-quarter: last N seconds of each quarter/half
    end_of_quarter_sec: float = 60.0
    # Early blowout: lead thresholds per quarter (or mapped half)
    blowout_q1: int = 10
    blowout_q2: int = 18
    # Foul storm: N fouls within a sliding window
    foul_count: int = 3
    foul_window_sec: float = 120.0
    foul_proximity_sec: float = 90.0  # how close to current clock the window must be
    # Timeout cluster: back-to-back timeouts within N seconds
    timeout_gap_sec: float = 90.0
    timeout_proximity_sec: float = 90.0
    # Clean play re-entry: seconds of foul/timeout-free play after dead zone
    clean_play_sec: float = 120.0
    # Whether clean play window must be truly clean (no fouls/TOs) vs just elapsed time
    clean_play_strict: bool = True


@dataclass
class _EngineConfig:
    epsilon_base: float = 0.04
    epsilon_time_k: float = 1.0
    persistence_sec: float = 45.0
    cooldown_sec: float = 180.0  # 3 minutes
    ema_span: int = 25
    regulation_secs: float = 2400.0
    trade_log_path: str = "data/trade_log.jsonl"
    # Min wall seconds between execute signals (None = no limit).
    min_seconds_between_executes: float | None = None
    # Dead zone configuration
    dead_zone: DeadZoneConfig | None = None


class TradingEngine:
    """Stateful decision engine fed one tick at a time from the live loop."""

    def __init__(
        self,
        interval_sec: float = 5.0,
        epsilon_base: float = 0.04,
        epsilon_time_k: float = 1.0,
        persistence_sec: float = 45.0,
        cooldown_min: float = 3.0,
        ema_span: int = 25,
        trade_log_path: str = "data/trade_log.jsonl",
        regulation_secs: float = 2400.0,
        min_seconds_between_executes: float | None = None,
        dead_zone_config: DeadZoneConfig | None = None,
    ):
        dz_cfg = dead_zone_config or DeadZoneConfig()
        self._cfg = _EngineConfig(
            epsilon_base=epsilon_base,
            epsilon_time_k=epsilon_time_k,
            persistence_sec=persistence_sec,
            cooldown_sec=cooldown_min * 60.0,
            ema_span=ema_span,
            regulation_secs=regulation_secs,
            trade_log_path=trade_log_path,
            min_seconds_between_executes=min_seconds_between_executes,
            dead_zone=dz_cfg,
        )
        self._interval = interval_sec
        self._k_ticks = max(1, math.ceil(persistence_sec / interval_sec))

        # State
        self._state = State.IDLE
        self._last_execute_mono: float | None = None
        self._now_mono: float = 0.0
        self._ema_edge: float = 0.0
        self._ema_alpha: float = 2.0 / (ema_span + 1)
        self._ema_initialised = False
        self._side_streak: int = 0
        self._last_side: str = ""
        self._cooldown_remaining: float = 0.0
        self._dead_zone_clear_elapsed: float = 0.0
        self._last_dead_zone: bool = False
        self._in_hysteresis: bool = False

        self._log_path = Path(trade_log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def last_side(self) -> str:
        """Expose the last seen model side cleanly."""
        return self._last_side

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tick(
        self,
        snapshot: dict,
        payload: dict,
        predict_result: dict,
        events: list[dict],
        *,
        clock_delta: float | None = None,
    ) -> TickResult:
        """Process one tick. Returns state, action, and optional log entry.

        clock_delta: if set, overrides self._interval for dead-zone and cooldown
                     timers (used by demo replay where wall-clock != game-clock).
        """
        self._now_mono = time.monotonic()
        effective_interval = clock_delta if clock_delta is not None else self._interval

        period = int(snapshot.get("period", 1))
        t_rem = float(snapshot.get("time_remaining_sec", 2400))
        score_diff = int(snapshot.get("score_diff", 0))
        edge_raw = float(predict_result.get("edge_vs_market", 0.0))
        side = str(predict_result.get("model_side", ""))

        # 1. Normalised lead
        sig = sigma_at(period, t_rem, self._cfg.regulation_secs)
        norm_lead = normalised_lead(score_diff, sig)

        # 2–5. EMA, persistence, epsilon (always updated for logging / display)
        self._update_ema(edge_raw)
        self._update_persistence(side)
        eps_eff = self._epsilon_effective(t_rem)

        prev_state = self._state
        action = "none"
        reason = ""

        # Regime gate
        dz = self._check_dead_zones(events, period, t_rem, score_diff)
        dz_cfg = self._cfg.dead_zone

        # State machine transition
        if dz:
            if self._state != State.IDLE:
                reason = f"dead_zone:{dz}"
                self._transition(State.IDLE, reason, snapshot, payload, predict_result)
            self._last_dead_zone = True
            self._dead_zone_clear_elapsed = 0.0
        else:
            if self._last_dead_zone:
                # In strict mode, any foul/timeout during the clean play window resets the counter
                if dz_cfg.clean_play_strict and events and _has_recent_disruptions(
                    events, t_rem, self._dead_zone_clear_elapsed
                ):
                    self._dead_zone_clear_elapsed = 0.0
                else:
                    self._dead_zone_clear_elapsed += effective_interval

                if self._dead_zone_clear_elapsed < dz_cfg.clean_play_sec:
                    dz = f"clean_play_wait({self._dead_zone_clear_elapsed:.0f}/{dz_cfg.clean_play_sec:.0f}s)"
                else:
                    self._last_dead_zone = False
                    self._dead_zone_clear_elapsed = 0.0

            if not dz:
                action, reason = self._step_state_machine(eps_eff, side, self._now_mono)

        # Decrement cooldown
        if self._state == State.COOLDOWN:
            self._cooldown_remaining -= effective_interval
            if self._cooldown_remaining <= 0:
                self._cooldown_remaining = 0
                self._in_hysteresis = True
                reason = "cooldown_expired"
                self._transition(State.IDLE, reason, snapshot, payload, predict_result)

        # Build log entry on any state change
        log_entry: dict[str, Any] | None = None
        if self._state != prev_state or action == "execute":
            log_entry = self._build_log(
                prev_state, self._state, reason, snapshot, payload, predict_result,
                norm_lead, self._ema_edge, eps_eff, dz,
            )
            self._write_log(log_entry)

        return TickResult(
            state=self._state,
            action=action,
            ema_edge=self._ema_edge,
            normalised_lead=norm_lead,
            dead_zone=dz,
            persistence_ticks=self._side_streak,
            epsilon_eff=eps_eff,
            log_entry=log_entry,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_dead_zones(
        self, events: list[dict], period: int, t_rem: float, score_diff: int,
    ) -> str:
        dz = self._cfg.dead_zone
        if _end_of_quarter(period, t_rem, self._cfg.regulation_secs, window_sec=dz.end_of_quarter_sec):
            return "end_of_quarter"
        if _early_blowout(
            period, score_diff, t_rem, self._cfg.regulation_secs,
            blowout_q1=dz.blowout_q1, blowout_q2=dz.blowout_q2,
        ):
            return "early_blowout"
        if events:
            if _detect_foul_storm(
                events, t_rem,
                foul_count=dz.foul_count,
                foul_window_sec=dz.foul_window_sec,
                proximity_sec=dz.foul_proximity_sec,
            ):
                return "foul_storm"
            if _detect_timeout_cluster(
                events, t_rem,
                gap_sec=dz.timeout_gap_sec,
                proximity_sec=dz.timeout_proximity_sec,
            ):
                return "timeout_cluster"
        return ""

    def _update_ema(self, edge_raw: float) -> None:
        if not self._ema_initialised:
            self._ema_edge = edge_raw
            self._ema_initialised = True
        else:
            self._ema_edge = self._ema_alpha * edge_raw + (1 - self._ema_alpha) * self._ema_edge

    def _update_persistence(self, side: str) -> None:
        if side == self._last_side:
            self._side_streak += 1
        else:
            self._side_streak = 1
            self._last_side = side

    def _epsilon_effective(self, t_rem: float) -> float:
        frac = t_rem / self._cfg.regulation_secs if self._cfg.regulation_secs > 0 else 0.0
        eps = self._cfg.epsilon_base * (1.0 + self._cfg.epsilon_time_k * frac)
        if self._in_hysteresis:
            eps *= 1.5
        return eps

    def _execute_cadence_allows(self) -> bool:
        c = self._cfg.min_seconds_between_executes
        if c is None or c <= 0:
            return True
        if self._last_execute_mono is None:
            return True
        return (self._now_mono - self._last_execute_mono) >= c

    def _record_execute_time(self) -> None:
        self._last_execute_mono = self._now_mono

    def _step_state_machine(
        self, eps_eff: float, side: str, now_mono: float
    ) -> tuple[str, str]:
        """Returns (action, reason). May call _transition internally."""
        self._now_mono = now_mono
        action = "none"
        reason = ""

        if self._state == State.IDLE:
            if abs(self._ema_edge) > eps_eff * 0.5:
                reason = "edge_building"
                self._transition(State.WATCHING, reason)
                self._in_hysteresis = False

        elif self._state == State.WATCHING:
            if abs(self._ema_edge) <= eps_eff * 0.3:
                reason = "edge_collapsed"
                self._transition(State.IDLE, reason)
            elif self._last_side != side:
                reason = "side_flip"
                self._transition(State.IDLE, reason)
            elif self._side_streak >= self._k_ticks and abs(self._ema_edge) > eps_eff:
                reason = "persistence_met"
                self._transition(State.ARMED, reason)

        elif self._state == State.ARMED:
            if self._last_side != side:
                reason = "side_flip"
                self._transition(State.IDLE, reason)
            elif abs(self._ema_edge) <= eps_eff * 0.5:
                reason = "edge_dropped"
                self._transition(State.IDLE, reason)
            elif abs(self._ema_edge) > eps_eff:
                if self._execute_cadence_allows():
                    reason = "all_filters_passed"
                    self._transition(State.EXECUTE, reason)
                    action = "execute"
                    self._record_execute_time()
                    self._transition(State.COOLDOWN, "post_execute")
                    self._cooldown_remaining = self._cfg.cooldown_sec
                else:
                    reason = "trade_interval_wait"

        elif self._state == State.EXECUTE:
            # Transient — immediately moves to COOLDOWN in the ARMED branch above
            pass

        elif self._state == State.COOLDOWN:
            # Handled in tick() after this call
            pass

        return action, reason

    def _transition(
        self,
        new_state: State,
        reason: str,
        snapshot: dict | None = None,
        payload: dict | None = None,
        predict_result: dict | None = None,
    ) -> None:
        self._state = new_state
        if new_state == State.IDLE:
            self._side_streak = 0

    def _build_log(
        self,
        state_from: State,
        state_to: State,
        reason: str,
        snapshot: dict,
        payload: dict,
        predict_result: dict,
        norm_lead: float,
        ema_edge: float,
        eps_eff: float,
        dead_zone: str,
    ) -> dict[str, Any]:
        return {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "state_from": state_from.value,
            "state_to": state_to.value,
            "reason": reason,
            "min_seconds_between_executes": self._cfg.min_seconds_between_executes,
            "edge_ema": round(ema_edge, 5),
            "edge_raw": round(float(predict_result.get("edge_vs_market", 0)), 5),
            "norm_lead": round(norm_lead, 3),
            "epsilon_eff": round(eps_eff, 4),
            "persistence_ticks": self._side_streak,
            "k_ticks_required": self._k_ticks,
            "poly": round(float(payload.get("poly_price", 0)), 4),
            "score": f"{snapshot.get('away_score', '?')}-{snapshot.get('home_score', '?')}",
            "clock": f"{snapshot.get('time_remaining_sec', 0):.0f}s P{snapshot.get('period', '?')}",
            "dead_zone": dead_zone,
            "side": str(predict_result.get("model_side", "")),
            "prob_home": round(float(predict_result.get("prob_home_win", 0)), 4),
        }

    def _write_log(self, entry: dict[str, Any]) -> None:
        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass
