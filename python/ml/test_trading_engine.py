"""Tests for trading engine regime gate (dead zones) and state machine."""

import json
import tempfile
from pathlib import Path

from trading_engine import (
    DeadZoneConfig,
    State,
    TradingEngine,
    _detect_foul_storm,
    _detect_timeout_cluster,
    _early_blowout,
    _end_of_quarter,
    _has_recent_disruptions,
    sigma_at,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_events(descriptions: list[tuple[float, str]]) -> list[dict]:
    """Create events list from (time_remaining_sec, event_text) tuples."""
    return [
        {"time_remaining_sec": t, "event_text": txt}
        for t, txt in descriptions
    ]


def _make_engine(**kwargs) -> TradingEngine:
    tmp = tempfile.mktemp(suffix=".jsonl")
    defaults = dict(interval_sec=5.0, trade_log_path=tmp)
    defaults.update(kwargs)
    return TradingEngine(**defaults)


def _tick(engine, period=1, t_rem=1200.0, score_diff=0, edge=0.05, side="HOME",
          events=None, poly_price=0.5):
    snapshot = {"period": period, "time_remaining_sec": t_rem, "score_diff": score_diff,
                "away_score": max(0, -score_diff), "home_score": max(0, score_diff)}
    payload = {"poly_price": poly_price}
    predict = {"edge_vs_market": edge, "model_side": side, "prob_home_win": 0.5 + edge}
    return engine.tick(snapshot, payload, predict, events or [])


# ---------------------------------------------------------------------------
# End-of-quarter dead zone
# ---------------------------------------------------------------------------

class TestEndOfQuarter:
    def test_last_60_seconds_triggers(self):
        assert _end_of_quarter(1, 30.0) is True     # 30 % 600 = 30 ≤ 60
        assert _end_of_quarter(1, 60.0) is True     # boundary
        assert _end_of_quarter(1, 0.0) is True      # buzzer

    def test_outside_window_clear(self):
        assert _end_of_quarter(1, 300.0) is False   # 300 % 600 = 300 > 60

    def test_custom_window(self):
        assert _end_of_quarter(1, 80.0, window_sec=90.0) is True   # 80 ≤ 90
        assert _end_of_quarter(1, 80.0, window_sec=60.0) is False  # 80 > 60

    def test_mens_half_format(self):
        # Men's period 1, regulation=2400 → quarter_len=1200
        assert _end_of_quarter(1, 50.0, regulation_secs=2400.0) is True   # 50 % 1200 = 50 ≤ 60
        assert _end_of_quarter(1, 100.0, regulation_secs=2400.0) is False  # 100 > 60


# ---------------------------------------------------------------------------
# Early blowout dead zone
# ---------------------------------------------------------------------------

class TestEarlyBlowout:
    def test_q1_blowout_default(self):
        assert _early_blowout(1, 10, 2000.0) is True
        assert _early_blowout(1, 9, 2000.0) is False

    def test_q2_blowout_default(self):
        assert _early_blowout(2, 18, 800.0) is True
        assert _early_blowout(2, 17, 800.0) is False

    def test_custom_thresholds(self):
        assert _early_blowout(1, 12, 2000.0, blowout_q1=15) is False
        assert _early_blowout(1, 15, 2000.0, blowout_q1=15) is True

    def test_negative_lead_uses_abs(self):
        assert _early_blowout(1, -10, 2000.0) is True

    def test_q3_q4_never_triggers(self):
        assert _early_blowout(3, 30, 600.0) is False
        assert _early_blowout(4, 30, 300.0) is False

    def test_mens_half_mapping(self):
        # Men's format: period=1, regulation=2400
        # t_rem > 1200 maps to Q1 equivalent
        assert _early_blowout(1, 10, 1500.0, regulation_secs=2400.0) is True
        # t_rem <= 1200 maps to Q2 equivalent
        assert _early_blowout(1, 18, 1000.0, regulation_secs=2400.0) is True
        assert _early_blowout(1, 10, 1000.0, regulation_secs=2400.0) is False


# ---------------------------------------------------------------------------
# Foul storm dead zone
# ---------------------------------------------------------------------------

class TestFoulStorm:
    def test_three_fouls_in_window(self):
        events = _make_events([
            (1000.0, "Smith personal foul"),
            (960.0, "Jones foul on the play"),
            (920.0, "Technical foul called"),
        ])
        assert _detect_foul_storm(events, 950.0) is True

    def test_two_fouls_not_enough(self):
        events = _make_events([
            (1000.0, "Smith personal foul"),
            (960.0, "Jones foul"),
        ])
        assert _detect_foul_storm(events, 980.0) is False

    def test_custom_foul_count(self):
        events = _make_events([
            (1000.0, "foul"), (980.0, "foul"), (960.0, "foul"), (940.0, "foul"),
        ])
        assert _detect_foul_storm(events, 970.0, foul_count=4) is True
        assert _detect_foul_storm(events, 970.0, foul_count=5) is False

    def test_outside_proximity_clear(self):
        events = _make_events([
            (2000.0, "foul"), (1980.0, "foul"), (1960.0, "foul"),
        ])
        # Current clock far away from the foul window
        assert _detect_foul_storm(events, 500.0) is False

    def test_non_foul_events_ignored(self):
        events = _make_events([
            (1000.0, "three-point basket"),
            (960.0, "turnover"),
            (920.0, "rebound"),
        ])
        assert _detect_foul_storm(events, 950.0) is False


# ---------------------------------------------------------------------------
# Timeout cluster dead zone
# ---------------------------------------------------------------------------

class TestTimeoutCluster:
    def test_back_to_back_timeouts(self):
        events = _make_events([
            (800.0, "Full timeout called"),
            (750.0, "30 second time-out"),
        ])
        assert _detect_timeout_cluster(events, 770.0) is True

    def test_single_timeout_clear(self):
        events = _make_events([(800.0, "timeout")])
        assert _detect_timeout_cluster(events, 790.0) is False

    def test_wide_gap_clear(self):
        events = _make_events([
            (800.0, "timeout"),
            (600.0, "timeout"),
        ])
        assert _detect_timeout_cluster(events, 700.0) is False  # gap=200 > 90

    def test_custom_gap(self):
        events = _make_events([
            (800.0, "timeout"),
            (700.0, "timeout"),
        ])
        # gap=100 > 90 default, but ≤ 120 custom
        assert _detect_timeout_cluster(events, 750.0, gap_sec=120.0) is True
        assert _detect_timeout_cluster(events, 750.0, gap_sec=90.0) is False


# ---------------------------------------------------------------------------
# Has recent disruptions (for strict clean play)
# ---------------------------------------------------------------------------

class TestHasRecentDisruptions:
    def test_foul_within_window(self):
        events = _make_events([(500.0, "personal foul")])
        assert _has_recent_disruptions(events, 490.0, 20.0) is True

    def test_timeout_within_window(self):
        events = _make_events([(500.0, "timeout called")])
        assert _has_recent_disruptions(events, 510.0, 15.0) is True

    def test_clean_window(self):
        events = _make_events([(500.0, "three-point shot")])
        assert _has_recent_disruptions(events, 500.0, 20.0) is False

    def test_disruption_outside_window(self):
        events = _make_events([(500.0, "foul")])
        assert _has_recent_disruptions(events, 400.0, 20.0) is False


# ---------------------------------------------------------------------------
# Sigma lookup
# ---------------------------------------------------------------------------

class TestSigma:
    def test_q1_start(self):
        sig = sigma_at(1, 600.0)
        assert sig == 16.0

    def test_q4_end(self):
        sig = sigma_at(4, 0.0)
        assert sig == 2.0

    def test_interpolation(self):
        sig = sigma_at(4, 90.0)
        # Between (180, 7.5) and (45, 3.0): frac = (90-45)/(180-45) = 45/135 ≈ 0.333
        expected = 3.0 + 0.333 * (7.5 - 3.0)
        assert abs(sig - expected) < 0.1


# ---------------------------------------------------------------------------
# Engine integration: dead zone forces IDLE
# ---------------------------------------------------------------------------

class TestEngineDeadZone:
    def test_end_of_quarter_forces_idle(self):
        engine = _make_engine()
        # Tick at end of quarter
        result = _tick(engine, period=1, t_rem=30.0, edge=0.10, side="HOME")
        assert result.dead_zone == "end_of_quarter"
        assert result.state == State.IDLE

    def test_blowout_forces_idle(self):
        engine = _make_engine()
        result = _tick(engine, period=1, t_rem=2000.0, score_diff=12, edge=0.10)
        assert result.dead_zone == "early_blowout"
        assert result.state == State.IDLE

    def test_custom_blowout_threshold(self):
        dz = DeadZoneConfig(blowout_q1=15)
        engine = _make_engine(dead_zone_config=dz)
        # 12-point lead doesn't trigger with threshold=15
        result = _tick(engine, period=1, t_rem=2000.0, score_diff=12, edge=0.10)
        assert result.dead_zone == ""

    def test_foul_storm_forces_idle(self):
        engine = _make_engine()
        events = _make_events([
            (1200.0, "personal foul"),
            (1190.0, "foul called"),
            (1180.0, "technical foul"),
        ])
        result = _tick(engine, period=1, t_rem=1190.0, edge=0.10, events=events)
        assert result.dead_zone == "foul_storm"

    def test_clean_play_wait_after_dead_zone(self):
        engine = _make_engine()
        # First tick: in dead zone
        _tick(engine, period=1, t_rem=30.0, edge=0.10)
        # Second tick: dead zone cleared but clean play wait active
        result = _tick(engine, period=1, t_rem=300.0, edge=0.10)
        assert "clean_play_wait" in result.dead_zone

    def test_clean_play_completes(self):
        engine = _make_engine(dead_zone_config=DeadZoneConfig(clean_play_sec=10.0, clean_play_strict=False))
        # Enter dead zone
        _tick(engine, period=1, t_rem=30.0, edge=0.10)
        # Tick until clean play completes (10s / 5s interval = 2 ticks)
        _tick(engine, period=1, t_rem=300.0, edge=0.10)
        result = _tick(engine, period=1, t_rem=295.0, edge=0.10)
        assert result.dead_zone == ""


# ---------------------------------------------------------------------------
# Engine integration: state machine transitions
# ---------------------------------------------------------------------------

class TestEngineStateMachine:
    def test_idle_to_watching(self):
        engine = _make_engine(epsilon_base=0.04)
        # t_rem=1200 on men's half → secs_in_q=0 → end_of_quarter triggers; use mid-half instead
        # At t_rem=900: frac=900/2400=0.375; eps=0.04*(1+0.375)=0.055; eps*0.5=0.0275
        result = _tick(engine, t_rem=900.0, edge=0.05)
        assert result.state == State.WATCHING

    def test_watching_collapses_on_low_edge(self):
        engine = _make_engine(epsilon_base=0.04, ema_span=2)  # fast EMA so edge drops quickly
        _tick(engine, t_rem=900.0, edge=0.05)  # → WATCHING
        _tick(engine, t_rem=895.0, edge=0.0)
        _tick(engine, t_rem=890.0, edge=0.0)
        result = _tick(engine, t_rem=885.0, edge=0.0)  # ema should have collapsed
        assert result.state == State.IDLE

    def test_full_cycle_to_execute(self):
        dz = DeadZoneConfig(clean_play_sec=0.0)  # disable clean play wait
        engine = _make_engine(epsilon_base=0.02, persistence_sec=10.0, interval_sec=5.0,
                              dead_zone_config=dz)
        # At t_rem=900: frac=0.375; eps=0.02*(1+0.375)=0.0275
        _tick(engine, t_rem=900.0, edge=0.08)  # → WATCHING, ema initialised
        _tick(engine, t_rem=895.0, edge=0.08)  # persistence building (streak=2)
        result = _tick(engine, t_rem=890.0, edge=0.08)  # persistence=3 ≥ k=2 → ARMED → EXECUTE → COOLDOWN
        assert result.action == "execute" or result.state == State.COOLDOWN


# ---------------------------------------------------------------------------
# Run with pytest
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
