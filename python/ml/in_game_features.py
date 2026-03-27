"""
Shared feature engineering for basketball in-game training and live inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def clock_to_seconds(clock_str: str, period: int) -> float | None:
    """Convert NCAA clock string (MM:SS) + period to total seconds remaining."""
    if not clock_str or ":" not in clock_str:
        return None
    try:
        mins, secs = map(int, clock_str.split(":"))
    except (ValueError, TypeError):
        return None
    period_seconds_left = mins * 60 + secs
    if period == 1:
        return 1200 + period_seconds_left
    if period == 2:
        return period_seconds_left
    return period_seconds_left


def replay_game(pbp: dict) -> list[dict]:
    """Replay play-by-play into canonical state events."""
    periods = pbp.get("periods", [])
    events: list[dict] = []
    for per in periods:
        period_num = int(per.get("periodNumber", 1) or 1)
        for play in per.get("playbyplayStats", []):
            home_score = play.get("homeScore")
            away_score = play.get("visitorScore")
            if home_score is None or away_score is None:
                continue
            secs = clock_to_seconds(play.get("clock", ""), period_num)
            if secs is None:
                continue
            events.append(
                {
                    "period": period_num,
                    "clock": play.get("clock", ""),
                    "time_remaining_sec": float(secs),
                    "home_score": int(home_score),
                    "away_score": int(away_score),
                    "event_text": play.get("eventDescription", ""),
                }
            )
    return events


def _to_seed(value: object, default: float = 8.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def compute_snapshot_from_events(events: list[dict], meta: dict, target_time: float | None = None) -> dict:
    """
    Compute one game-state snapshot for the requested game clock.
    If target_time is None, uses latest available state (min time remaining).
    """
    if not events:
        raise ValueError("No replay events available")

    sorted_events = sorted(events, key=lambda e: -e["time_remaining_sec"])
    if target_time is None:
        target_time = min(e["time_remaining_sec"] for e in sorted_events)

    cur_home = 0
    cur_away = 0
    cur_period = 1
    lead_changes = 0
    prev_leader = 0
    largest_lead = 0
    history: list[tuple[float, int]] = []

    for ev in sorted_events:
        if ev["time_remaining_sec"] < target_time:
            break
        cur_home = ev["home_score"]
        cur_away = ev["away_score"]
        cur_period = ev["period"]

        diff = cur_home - cur_away
        history.append((ev["time_remaining_sec"], diff))
        leader = 1 if diff > 0 else (-1 if diff < 0 else 0)
        if leader != 0 and prev_leader != 0 and leader != prev_leader:
            lead_changes += 1
        if leader != 0:
            prev_leader = leader
        largest_lead = max(largest_lead, abs(diff))

    if not history:
        first = sorted_events[0]
        history.append((first["time_remaining_sec"], first["home_score"] - first["away_score"]))
        cur_home = first["home_score"]
        cur_away = first["away_score"]
        cur_period = first["period"]
        largest_lead = abs(cur_home - cur_away)

    diff = cur_home - cur_away
    run_60 = 0
    run_120 = 0
    for prev_t, prev_diff in reversed(history[:-1]):
        elapsed = prev_t - target_time
        if elapsed <= 60:
            run_60 = diff - prev_diff
        if elapsed <= 120:
            run_120 = diff - prev_diff
            break

    momentum = diff
    if len(history) > 1:
        alpha = 0.3
        momentum = alpha * diff + (1 - alpha) * history[-2][1]

    return {
        "game_id": meta.get("gameID", ""),
        "game_date": meta.get("startDate", ""),
        "home_team": meta.get("home_short", ""),
        "away_team": meta.get("away_short", ""),
        "home_seed": meta.get("home_seed", ""),
        "away_seed": meta.get("away_seed", ""),
        "time_remaining_sec": float(target_time),
        "period": int(cur_period),
        "score_diff": int(diff),
        "home_score": int(cur_home),
        "away_score": int(cur_away),
        "scoring_run_60s": int(run_60),
        "scoring_run_120s": int(run_120),
        "lead_changes_so_far": int(lead_changes),
        "largest_lead": int(largest_lead),
        "momentum": float(momentum),
        "seed_diff": _to_seed(meta.get("home_seed")) - _to_seed(meta.get("away_seed")),
        "label": 1 if meta.get("home_winner", False) else 0,
    }


def parse_poly_series(poly: dict) -> tuple[list[tuple[float, float]], list[dict]]:
    """Normalize price/trade arrays from fetched Polymarket payload."""
    prices = poly.get("prices", []) if poly else []
    trades = poly.get("trades", []) if poly else []

    price_list: list[tuple[float, float]] = []
    for p in prices:
        price_list.append((float(p.get("t", 0) or 0), float(p.get("p", 0) or 0)))
    price_list.sort(key=lambda x: x[0])

    trade_list: list[dict] = []
    for t in trades:
        ts = t.get("timestamp") or t.get("matchedAt") or t.get("createdAt") or 0
        trade_list.append(
            {
                "ts": float(ts),
                "price": float(t.get("price", 0) or 0),
                "size": float(t.get("size", 0) or 0),
                "side": str(t.get("side", "")).upper(),
            }
        )
    trade_list.sort(key=lambda x: x["ts"])
    return price_list, trade_list


def infer_game_start_epoch(max_time: float, price_list: list[tuple[float, float]], trade_list: list[dict]) -> float:
    """Infer game start epoch from available market timestamps."""
    trade_ts = [t["ts"] for t in trade_list if t["ts"] > 0]
    if trade_ts:
        return min(trade_ts)
    if price_list:
        return price_list[-1][0] - float(max_time)
    return 0.0


def poly_features_at_time(
    t_remain: float,
    max_time: float,
    game_start_epoch: float,
    price_list: list[tuple[float, float]],
    trade_list: list[dict],
) -> dict:
    """Compute per-timestamp Polymarket features aligned to game clock."""
    if not price_list and not trade_list:
        return {
            "poly_price": 0.0,
            "poly_price_drift_5m": 0.0,
            "poly_volume_1m": 0.0,
            "poly_buy_fraction_5m": 0.5,
            "poly_trade_count_5m": 0.0,
            "has_poly": 0,
        }

    wall_time = game_start_epoch + (float(max_time) - float(t_remain))

    best_price = 0.5
    price_5m_ago = 0.5
    if price_list:
        best_price = price_list[0][1]
        price_5m_ago = best_price
        for ts, price in price_list:
            if ts <= wall_time:
                best_price = price
            else:
                break
        for ts, price in price_list:
            if ts <= wall_time - 300:
                price_5m_ago = price
            else:
                break

    vol_1m = 0.0
    buys_5m = 0
    total_5m = 0
    count_5m = 0
    for tr in trade_list:
        if tr["ts"] > wall_time:
            break
        if tr["ts"] >= wall_time - 60:
            vol_1m += tr["size"]
        if tr["ts"] >= wall_time - 300:
            count_5m += 1
            total_5m += 1
            if tr["side"] == "BUY":
                buys_5m += 1

    return {
        "poly_price": float(best_price),
        "poly_price_drift_5m": float(best_price - price_5m_ago),
        "poly_volume_1m": float(vol_1m),
        "poly_buy_fraction_5m": float(buys_5m / total_5m) if total_5m else 0.5,
        "poly_trade_count_5m": float(count_5m),
        "has_poly": 1,
    }


def resample_to_minutes(events: list[dict], meta: dict) -> pd.DataFrame:
    """Create minute-by-minute game-state rows from replay events."""
    if not events:
        return pd.DataFrame()

    max_time = max(e["time_remaining_sec"] for e in events)
    minutes = list(range(int(max_time), -1, -60))
    if 0 not in minutes:
        minutes.append(0)

    rows = [compute_snapshot_from_events(events, meta, target_time=float(t)) for t in minutes]
    return pd.DataFrame(rows)


def merge_polymarket(df: pd.DataFrame, poly: dict, game_start_epoch: float = 0.0) -> pd.DataFrame:
    """Attach Polymarket features for each row's game clock."""
    out = df.copy()
    defaults = {
        "poly_price": 0.0,
        "poly_price_drift_5m": 0.0,
        "poly_volume_1m": 0.0,
        "poly_buy_fraction_5m": 0.5,
        "poly_trade_count_5m": 0.0,
        "has_poly": 0,
    }
    for key, value in defaults.items():
        out[key] = value

    if out.empty or not poly.get("found", False):
        return out

    price_list, trade_list = parse_poly_series(poly)
    if not price_list and not trade_list:
        return out

    max_time = float(out["time_remaining_sec"].max())
    if not game_start_epoch:
        game_start_epoch = infer_game_start_epoch(max_time, price_list, trade_list)

    for idx in out.index:
        feats = poly_features_at_time(
            t_remain=float(out.at[idx, "time_remaining_sec"]),
            max_time=max_time,
            game_start_epoch=float(game_start_epoch),
            price_list=price_list,
            trade_list=trade_list,
        )
        for k, v in feats.items():
            out.at[idx, k] = v
    return out


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns required by training/serving."""
    out = df.copy()
    eps = 1e-6
    p = out["poly_price"].clip(eps, 1.0 - eps)
    out["logit_poly_price"] = np.log(p / (1.0 - p))
    out["price_x_score_diff"] = out["poly_price"] * out["score_diff"]
    out["time_x_score_diff"] = out["time_remaining_sec"] * out["score_diff"]
    out["log_time_remaining"] = np.log1p(out["time_remaining_sec"])
    out["abs_score_diff"] = out["score_diff"].abs()
    if "seed_diff" not in out.columns:
        out["seed_diff"] = (
            pd.to_numeric(out.get("home_seed"), errors="coerce").fillna(8)
            - pd.to_numeric(out.get("away_seed"), errors="coerce").fillna(8)
        )
    return out
