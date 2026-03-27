"""
Predict one live/in-progress NCAA game by chaining:
scoreboard/game lookup -> NCAA play-by-play -> optional Polymarket -> ML server.

Examples:
  python python/ml/predict_live_game.py --game-id 6534602
  python python/ml/predict_live_game.py --date 2026-03-20 --away UCF --home UCLA
  python python/ml/predict_live_game.py --from-json data/games/6534602.json
"""

from __future__ import annotations

import argparse
import json
import ssl
import time
import urllib.request
from datetime import datetime

from fetch_games import (
    _match_slug,
    fetch_play_by_play,
    find_game_on_scoreboard,
    scrape_poly_slugs,
)
from in_game_features import (
    compute_snapshot_from_events,
    replay_game,
)

NCAA_BASE = "https://ncaa-api.henrygd.me"
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE


def _norm(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _get_json(url: str) -> dict | list | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "penn-ml-live/1.0"})
        with urllib.request.urlopen(req, timeout=10, context=SSL_CTX) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _pick_moneyline_market(event: dict) -> dict | None:
    markets = event.get("markets", [])
    if not markets:
        return None
    for m in markets:
        q = (m.get("question") or "").lower()
        if "spread" in q or "o/u" in q or "over" in q or "under" in q or "prop" in q:
            continue
        return m
    return markets[0]


def _pick_home_index(outcomes: list[str], meta: dict) -> int:
    home = _norm(meta.get("home_short", ""))
    away = _norm(meta.get("away_short", ""))
    scores = []
    for i, out in enumerate(outcomes):
        o = _norm(out)
        score = 0
        if home and (home in o or o in home):
            score += 2
        if away and (away in o or o in away):
            score -= 2
        scores.append((score, i))
    best_score, best_idx = max(scores) if scores else (0, 0)
    if best_score != 0:
        return best_idx
    if len(outcomes) == 2 and away and any(away in _norm(outcomes[i]) for i in range(2)):
        away_idx = 0 if away in _norm(outcomes[0]) else 1
        return 1 - away_idx
    return 0


def _fetch_live_polymarket_features(meta: dict, slug: str) -> dict:
    defaults = {
        "poly_price": 0.0,
        "poly_price_drift_5m": 0.0,
        "poly_volume_1m": 0.0,
        "poly_buy_fraction_5m": 0.5,
        "poly_trade_count_5m": 0,
        "has_poly": 0,
    }
    event_data = _get_json(f"https://gamma-api.polymarket.com/events?slug={slug}&limit=1")
    if not event_data:
        return defaults
    event = event_data[0]
    market = _pick_moneyline_market(event)
    if not market:
        return defaults

    try:
        outcomes = json.loads(market.get("outcomes", "[]"))
        tokens = json.loads(market.get("clobTokenIds", "[]"))
        outcome_prices = [float(x) for x in json.loads(market.get("outcomePrices", "[]"))]
    except Exception:
        return defaults
    if not outcomes or not tokens:
        return defaults

    home_idx = _pick_home_index(outcomes, meta)
    home_idx = max(0, min(home_idx, len(tokens) - 1))
    home_token = tokens[home_idx]

    # Prefer latest displayed outcome price, fallback to token history latest.
    poly_price = None
    if home_idx < len(outcome_prices):
        poly_price = float(outcome_prices[home_idx])

    history = _get_json(f"https://clob.polymarket.com/prices-history?market={home_token}&interval=max&fidelity=500")
    hist = history.get("history", []) if isinstance(history, dict) else []
    if poly_price is None and hist:
        poly_price = float(hist[-1].get("p", 0.0))
    if poly_price is None:
        return defaults

    now_ts = time.time()
    drift = 0.0
    if hist:
        older = None
        for pt in hist:
            ts = float(pt.get("t", 0) or 0)
            if ts <= now_ts - 300:
                older = float(pt.get("p", 0) or 0)
            else:
                break
        if older is not None:
            drift = poly_price - older

    trades = _get_json(
        f"https://data-api.polymarket.com/trades?market={market.get('conditionId','')}&limit=500"
    )
    trades = trades if isinstance(trades, list) else []
    vol_1m = 0.0
    buys_5m = 0
    total_5m = 0
    count_5m = 0
    for t in trades:
        asset = str(t.get("asset", ""))
        if asset and asset != home_token:
            continue
        ts = float(t.get("timestamp") or t.get("matchedAt") or t.get("createdAt") or 0)
        if ts <= 0:
            continue
        if ts >= now_ts - 60:
            vol_1m += float(t.get("size", 0) or 0)
        if ts >= now_ts - 300:
            count_5m += 1
            total_5m += 1
            if str(t.get("side", "")).upper() == "BUY":
                buys_5m += 1

    return {
        "poly_price": float(poly_price),
        "poly_price_drift_5m": float(drift),
        "poly_volume_1m": float(vol_1m),
        "poly_buy_fraction_5m": float(buys_5m / total_5m) if total_5m else 0.5,
        "poly_trade_count_5m": int(count_5m),
        "has_poly": 1,
    }


def _meta_from_pbp(game_id: str, pbp: dict, date_str: str | None = None) -> dict:
    teams = pbp.get("teams", [])
    home = next((t for t in teams if t.get("isHome")), {}) if teams else {}
    away = next((t for t in teams if not t.get("isHome")), {}) if teams else {}
    if not date_str:
        date_str = datetime.now().strftime("%m/%d/%Y")
    else:
        date_str = datetime.strptime(date_str, "%Y-%m-%d").strftime("%m/%d/%Y")
    return {
        "gameID": str(game_id),
        "title": pbp.get("description", ""),
        "startDate": date_str,
        "home_seo": home.get("seoname", ""),
        "home_short": home.get("nameShort", home.get("name6Char", "")),
        "home_seed": "",
        "away_seo": away.get("seoname", ""),
        "away_short": away.get("nameShort", away.get("name6Char", "")),
        "away_seed": "",
    }


def _post_predict(server: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        server.rstrip("/") + "/predict",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def _resolve_from_args(args) -> tuple[dict, dict, dict | None]:
    if args.from_json:
        with open(args.from_json) as f:
            record = json.load(f)
        return record.get("meta", {}), record.get("ncaa_pbp", {}), record.get("polymarket")

    if args.game_id:
        pbp = fetch_play_by_play(str(args.game_id))
        if not pbp:
            raise RuntimeError(f"Could not fetch play-by-play for game id {args.game_id}")
        meta = _meta_from_pbp(str(args.game_id), pbp, date_str=args.date)
        return meta, pbp, None

    if not (args.date and args.away and args.home):
        raise RuntimeError("Provide either --game-id, or (--date --away --home), or --from-json")

    game = find_game_on_scoreboard(
        date=args.date,
        away_hint=args.away,
        home_hint=args.home,
        require_bracket=args.require_bracket,
    )
    if not game:
        raise RuntimeError("No matching game found on scoreboard for provided date/teams")
    pbp = fetch_play_by_play(str(game["gameID"]))
    if not pbp:
        raise RuntimeError(f"Could not fetch play-by-play for game id {game['gameID']}")
    return game, pbp, None


def main():
    parser = argparse.ArgumentParser(description="Predict one live NCAA game via ML server")
    parser.add_argument("--game-id", help="NCAA game id")
    parser.add_argument("--date", help="Date YYYY-MM-DD (needed for scoreboard lookup)")
    parser.add_argument("--away", help="Away team hint (for scoreboard lookup)")
    parser.add_argument("--home", help="Home team hint (for scoreboard lookup)")
    parser.add_argument("--from-json", help="Use pre-fetched game JSON file")
    parser.add_argument("--poly-slug", help="Optional Polymarket event slug override")
    parser.add_argument("--no-poly", action="store_true", help="Skip Polymarket fetch")
    parser.add_argument("--require-bracket", action="store_true", default=False)
    parser.add_argument("--server", default="http://127.0.0.1:8766", help="Prediction server URL")
    args = parser.parse_args()

    meta, pbp, poly_prefetched = _resolve_from_args(args)

    events = replay_game(pbp)
    if not events:
        raise RuntimeError("No usable play-by-play events with score/clock data")
    snapshot = compute_snapshot_from_events(events, meta)

    poly_feats = {
        "poly_price": 0.0,
        "poly_price_drift_5m": 0.0,
        "poly_volume_1m": 0.0,
        "poly_buy_fraction_5m": 0.5,
        "poly_trade_count_5m": 0,
        "has_poly": 0,
    }
    if poly_prefetched and isinstance(poly_prefetched, dict) and poly_prefetched.get("found"):
        # Compatibility path for offline json runs; live fetch below can override.
        try:
            prices = poly_prefetched.get("prices", [])
            if prices:
                poly_feats["poly_price"] = float(prices[-1].get("p", 0) or 0)
                poly_feats["has_poly"] = 1
        except Exception:
            pass
    slug = args.poly_slug
    if not args.no_poly:
        if not slug:
            slugs = scrape_poly_slugs()
            slug = _match_slug(
                slugs,
                meta.get("away_seo", ""),
                meta.get("home_seo", ""),
                meta.get("away_short", ""),
                meta.get("home_short", ""),
                datetime.strptime(
                    meta.get("startDate", datetime.now().strftime("%m/%d/%Y")), "%m/%d/%Y"
                ).strftime("%Y-%m-%d"),
            )
        if slug:
            poly_feats = _fetch_live_polymarket_features(meta, slug)

    payload = {
        "time_remaining_sec": float(snapshot["time_remaining_sec"]),
        "period": int(snapshot["period"]),
        "score_diff": int(snapshot["score_diff"]),
        "scoring_run_60s": int(snapshot["scoring_run_60s"]),
        "scoring_run_120s": int(snapshot["scoring_run_120s"]),
        "lead_changes_so_far": int(snapshot["lead_changes_so_far"]),
        "largest_lead": int(snapshot["largest_lead"]),
        "momentum": float(snapshot["momentum"]),
        "seed_diff": float(snapshot["seed_diff"]),
        "poly_price": float(poly_feats["poly_price"]),
        "poly_price_drift_5m": float(poly_feats["poly_price_drift_5m"]),
        "poly_volume_1m": float(poly_feats["poly_volume_1m"]),
        "poly_buy_fraction_5m": float(poly_feats["poly_buy_fraction_5m"]),
        "poly_trade_count_5m": int(poly_feats["poly_trade_count_5m"]),
        "has_poly": int(poly_feats["has_poly"]),
    }

    result = _post_predict(args.server, payload)

    print("\n=== Live Prediction ===")
    print(f"Game: {meta.get('away_short', '?')} @ {meta.get('home_short', '?')} ({meta.get('gameID', '')})")
    print(f"Clock: {snapshot['time_remaining_sec']:.0f}s remaining | Period {snapshot['period']}")
    print(f"Score: {snapshot['away_score']} - {snapshot['home_score']} (away-home)")
    print(f"Polymarket: {'yes' if payload['has_poly'] else 'no'}")
    print("\nModel response:")
    print(json.dumps(result, indent=2))
    print("\nFeature payload sent:")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
