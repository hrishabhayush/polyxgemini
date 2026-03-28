"""
Predict one live/in-progress NCAA game by chaining:
scoreboard/game lookup -> NCAA play-by-play -> optional Polymarket -> ML server.

Examples:
  python python/ml/predict_live_game.py --game-id 6534602
  python python/ml/predict_live_game.py --date 2026-03-20 --away UCF --home UCLA
  python python/ml/predict_live_game.py --from-json data/games/6534602.json
  python python/ml/predict_live_game.py --game-id 6534602 --live --interval-sec 5
  python python/ml/predict_live_game.py --game-id 6534602 --live --trade-engine
  python python/ml/predict_live_game.py --game-id 6534713 --demo data/games/6534602.json
  python python/ml/predict_live_game.py --game-id 6534713 --live --demo data/games/6534602.json --trade-engine --risk-engine
"""

from __future__ import annotations

import argparse
import json
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime

from fetch_games import (
    _match_slug,
    find_game_on_scoreboard,
    scrape_poly_slugs,
)
from in_game_features import (
    compute_snapshot_from_events,
    infer_game_start_epoch,
    parse_poly_series,
    poly_features_at_time,
    replay_game,
)
from trading_engine import DeadZoneConfig, TradingEngine
from paper_portfolio import PaperPortfolio, synthetic_paper_orderbook
from risk_engine import CurveConfig, RiskEngine
import prom_metrics as pm

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

NCAA_BASE = "https://ncaa-api.henrygd.me"


def fetch_play_by_play_live(game_id: str) -> dict | None:
    """Fresh PBP JSON (cache-busted); use for live loops so scores track ncaa.com."""
    # Large payloads; avoid flaky 10s timeouts and silent failures.
    return _get_json(
        f"{NCAA_BASE}/game/{game_id}/play-by-play",
        timeout=45.0,
        log_errors=True,
        error_label=f"NCAA PBP game_id={game_id}",
    )


DEFAULT_POLY_FEATS = {
    "poly_price": 0.0,
    "poly_price_drift_5m": 0.0,
    "poly_volume_1m": 0.0,
    "poly_buy_fraction_5m": 0.5,
    "poly_trade_count_5m": 0,
    "has_poly": 0,
}


def _demo_poly_features(
    time_remaining_sec: float,
    max_time: float,
    game_start_epoch: float,
    price_list: list[tuple[float, float]],
    trade_list: list[dict],
) -> dict:
    """Serve historical Polymarket features aligned to current game clock."""
    return poly_features_at_time(
        t_remain=time_remaining_sec,
        max_time=max_time,
        game_start_epoch=game_start_epoch,
        price_list=price_list,
        trade_list=trade_list,
    )


def _norm(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _cache_bust_url(url: str) -> str:
    """Append a unique query param so CDNs/proxies do not serve stale JSON."""
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}_={int(time.time() * 1000)}"


def _get_json(
    url: str,
    *,
    timeout: float = 10.0,
    log_errors: bool = False,
    error_label: str | None = None,
) -> dict | list | None:
    """
    GET JSON (cache-busted). On failure returns None.
    NCAA PBP uses fetch_play_by_play_live(..., log_errors=True) for stderr diagnostics.
    """
    label = error_label or url
    try:
        url_busted = _cache_bust_url(url)
        req = urllib.request.Request(
            url_busted,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/131.0.0.0 Safari/537.36 penn-ml-live/1.0"
                ),
                "Accept": "application/json,*/*;q=0.8",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout, context=SSL_CTX) as r:
            raw = r.read()
        text = raw.decode("utf-8", errors="replace")
        return json.loads(text)
    except urllib.error.HTTPError as e:
        if log_errors:
            try:
                body = e.read().decode("utf-8", errors="replace")[:400]
            except Exception:
                body = ""
            print(
                f"WARN: HTTP {e.code} fetching {label}: {e.reason!r} {body}",
                file=sys.stderr,
            )
        return None
    except urllib.error.URLError as e:
        if log_errors:
            print(f"WARN: URL error fetching {label}: {e.reason!r}", file=sys.stderr)
        return None
    except json.JSONDecodeError as e:
        if log_errors:
            print(f"WARN: invalid JSON from {label}: {e}", file=sys.stderr)
        return None
    except Exception as e:
        if log_errors:
            print(f"WARN: fetch failed for {label}: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def _clob_book_price(token_id: str) -> tuple[float | None, float | None]:
    """
    Live CLOB top-of-book + last trade (matches UI better than Gamma outcomePrices).
    Returns (mid_price_or_none, last_trade_or_none).
    """
    q = urllib.parse.urlencode({"token_id": token_id})
    book = _get_json(f"https://clob.polymarket.com/book?{q}")
    if not isinstance(book, dict):
        return None, None

    best_bid: float | None = None
    for b in book.get("bids") or []:
        try:
            p = float(b.get("price", 0) or 0)
        except (TypeError, ValueError):
            continue
        if p > 0:
            best_bid = p if best_bid is None else max(best_bid, p)

    best_ask: float | None = None
    for a in book.get("asks") or []:
        try:
            p = float(a.get("price", 0) or 0)
        except (TypeError, ValueError):
            continue
        if p > 0:
            best_ask = p if best_ask is None else min(best_ask, p)

    last_trade: float | None = None
    try:
        if book.get("last_trade_price") not in (None, ""):
            last_trade = float(book["last_trade_price"])
    except (TypeError, ValueError):
        pass

    mid: float | None = None
    if best_bid is not None and best_ask is not None:
        if best_bid <= best_ask:
            mid = (best_bid + best_ask) / 2.0
        elif last_trade is not None:
            mid = last_trade
    if mid is None and best_bid is not None and last_trade is not None:
        mid = (best_bid + last_trade) / 2.0
    if mid is None and best_ask is not None and last_trade is not None:
        mid = (best_ask + last_trade) / 2.0
    if mid is None and last_trade is not None:
        mid = last_trade

    return mid, last_trade


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
    defaults = {**DEFAULT_POLY_FEATS}
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

    # 1) CLOB order book each call (no caching) — matches UI bid/ask / mid; Gamma outcomePrices are often stale or 50/50.
    poly_price = None
    book_mid, book_last = _clob_book_price(home_token)
    if book_mid is not None:
        poly_price = book_mid
    elif book_last is not None:
        poly_price = book_last

    # 2) Gamma outcomePrices (fallback)
    if poly_price is None and home_idx < len(outcome_prices):
        poly_price = float(outcome_prices[home_idx])

    history = _get_json(
        f"https://clob.polymarket.com/prices-history?market={urllib.parse.quote(str(home_token), safe='')}&interval=max&fidelity=500"
    )
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


def _merge_scoreboard_meta(meta: dict, game: dict | None) -> dict:
    """Attach seeds from scoreboard lookup when available."""
    if not game:
        return meta
    out = {**meta}
    if game.get("home_seed") is not None and game.get("home_seed") != "":
        out["home_seed"] = game.get("home_seed", "")
    if game.get("away_seed") is not None and game.get("away_seed") != "":
        out["away_seed"] = game.get("away_seed", "")
    return out


def _pbp_indicates_final(pbp: dict) -> bool:
    st = (pbp.get("status") or "").strip().lower()
    if st in ("final", "complete", "completed"):
        return True
    return False


def _fetch_pbp_with_retries(game_id: str, retries: int = 5) -> dict | None:
    for attempt in range(retries):
        pbp = fetch_play_by_play_live(str(game_id))
        if pbp:
            return pbp
        if attempt < retries - 1:
            time.sleep(0.5 * (attempt + 1))
    return None


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


def _build_payload_from_snapshot(snapshot: dict, poly_feats: dict) -> dict:
    """Build ML payload from a pre-computed snapshot + poly features."""
    return {
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


def build_predict_payload(meta: dict, pbp: dict, poly_feats: dict) -> tuple[dict, dict]:
    """Shared feature assembly for one-shot and live loops."""
    events = replay_game(pbp)
    if not events:
        raise ValueError("No usable play-by-play events with score/clock data")
    snapshot = compute_snapshot_from_events(events, meta)
    payload = _build_payload_from_snapshot(snapshot, poly_feats)
    return snapshot, payload


def _poly_from_prefetched(poly_prefetched: dict | None) -> dict:
    feats = {**DEFAULT_POLY_FEATS}
    if poly_prefetched and isinstance(poly_prefetched, dict) and poly_prefetched.get("found"):
        try:
            prices = poly_prefetched.get("prices", [])
            if prices:
                feats["poly_price"] = float(prices[-1].get("p", 0) or 0)
                feats["has_poly"] = 1
        except Exception:
            pass
    return feats


def _resolve_polymarket_slug(
    meta: dict,
    poly_slug_arg: str | None,
    slugs_list: list[str] | None,
) -> str | None:
    if poly_slug_arg:
        return poly_slug_arg
    slugs = slugs_list if slugs_list is not None else scrape_poly_slugs()
    return _match_slug(
        slugs,
        meta.get("away_seo", ""),
        meta.get("home_seo", ""),
        meta.get("away_short", ""),
        meta.get("home_short", ""),
        datetime.strptime(
            meta.get("startDate", datetime.now().strftime("%m/%d/%Y")), "%m/%d/%Y"
        ).strftime("%Y-%m-%d"),
    )


def _resolve_from_args(args) -> tuple[dict, dict, dict | None, dict | None]:
    """
    Returns (meta, pbp, poly_prefetched, scoreboard_game_or_none).
    scoreboard_game is set when resolving via date/teams (for seeds).
    """
    if args.from_json:
        with open(args.from_json) as f:
            record = json.load(f)
        return record.get("meta", {}), record.get("ncaa_pbp", {}), record.get("polymarket"), None

    if args.game_id:
        pbp = _fetch_pbp_with_retries(str(args.game_id))
        if not pbp:
            raise RuntimeError(f"Could not fetch play-by-play for game id {args.game_id}")
        meta = _meta_from_pbp(str(args.game_id), pbp, date_str=args.date)
        return meta, pbp, None, None

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
    pbp = _fetch_pbp_with_retries(str(game["gameID"]))
    if not pbp:
        raise RuntimeError(f"Could not fetch play-by-play for game id {game['gameID']}")
    meta = _merge_scoreboard_meta(_meta_from_pbp(str(game["gameID"]), pbp, date_str=args.date), game)
    return meta, pbp, None, game


def _fingerprint(snapshot: dict, payload: dict) -> tuple:
    return (
        round(float(snapshot["time_remaining_sec"]), 1),
        int(snapshot["away_score"]),
        int(snapshot["home_score"]),
        round(float(payload["poly_price"]), 4),
    )


def run_live_loop(args, *, stop_on_final: bool = True) -> None:
    meta0, pbp0, poly_pf, game_sb = _resolve_from_args(args)
    game_id = meta0.get("gameID")
    if not game_id:
        raise RuntimeError("Could not determine game id for live loop")

    meta0 = _merge_scoreboard_meta(meta0, game_sb)

    # Demo mode: load historical Polymarket data + pre-compute all PBP events for replay
    demo_price_list: list[tuple[float, float]] | None = None
    demo_trade_list: list[dict] | None = None
    demo_start_epoch: float = 0.0
    demo_max_time: float = 2400.0
    demo_all_events: list | None = None
    demo_mode = bool(getattr(args, "demo", None))
    if demo_mode:
        with open(args.demo) as f:
            demo_record = json.load(f)
        demo_poly = demo_record.get("polymarket", {})
        demo_price_list, demo_trade_list = parse_poly_series(demo_poly)
        demo_start_epoch = infer_game_start_epoch(demo_max_time, demo_price_list, demo_trade_list)
        demo_title = demo_record.get("meta", {}).get("title", args.demo)
        # Pre-load all PBP events so we can step through them
        pbp_init = _fetch_pbp_with_retries(str(game_id))
        if not pbp_init:
            raise RuntimeError("Could not fetch play-by-play for demo replay")
        demo_all_events = replay_game(pbp_init)
        if not demo_all_events:
            raise RuntimeError("No usable PBP events for demo replay")
        print(
            f"[DEMO] Replaying market data from {demo_title} "
            f"({len(demo_all_events)} PBP events)",
            flush=True,
        )

    slugs_cache: list[str] | None = None
    if not demo_mode and not args.poly_slug and not args.no_poly:
        slugs_cache = scrape_poly_slugs()

    slug = None
    if not demo_mode and not args.no_poly:
        slug = _resolve_polymarket_slug(meta0, args.poly_slug, slugs_cache)

    use_engine = getattr(args, "trade_engine", False)
    engine: TradingEngine | None = None
    if use_engine:
        dz_cfg = DeadZoneConfig(
            end_of_quarter_sec=getattr(args, "dz_eoq_sec", 60.0),
            blowout_q1=getattr(args, "dz_blowout_q1", 10),
            blowout_q2=getattr(args, "dz_blowout_q2", 18),
            foul_count=getattr(args, "dz_foul_count", 3),
            foul_window_sec=getattr(args, "dz_foul_window_sec", 120.0),
            foul_proximity_sec=getattr(args, "dz_foul_proximity_sec", 90.0),
            timeout_gap_sec=getattr(args, "dz_timeout_gap_sec", 90.0),
            timeout_proximity_sec=getattr(args, "dz_timeout_proximity_sec", 90.0),
            clean_play_sec=getattr(args, "dz_clean_play_sec", 120.0),
            clean_play_strict=getattr(args, "dz_clean_play_strict", True),
        )
        engine = TradingEngine(
            interval_sec=args.interval_sec,
            epsilon_base=getattr(args, "epsilon_base", 0.04),
            epsilon_time_k=1.0,
            persistence_sec=getattr(args, "persistence_sec", 45.0),
            cooldown_min=getattr(args, "cooldown_min", 3.0),
            ema_span=getattr(args, "ema_span", 25),
            trade_log_path=getattr(args, "trade_log", "data/trade_log.jsonl"),
            min_seconds_between_executes=getattr(args, "trade_interval_sec", None),
            dead_zone_config=dz_cfg,
        )

    use_risk = getattr(args, "risk_engine", False)
    risk: RiskEngine | None = None
    portfolio: PaperPortfolio | None = None
    if use_risk:
        risk_cfg = CurveConfig(
            alpha=getattr(args, "risk_curve_alpha", 3.0),
            beta=getattr(args, "risk_curve_beta", 2.0),
            max_contracts=getattr(args, "risk_max_contracts", 10.0),
            news_boost_max=getattr(args, "risk_news_boost", 2.0),
        )
        risk = RiskEngine(risk_cfg)
        portfolio = PaperPortfolio(contracts_per_trade=1.0)
    prev_ema_edge: float | None = None

    engine_label = " | trade_engine=ON" if use_engine else ""
    if use_engine and getattr(args, "trade_interval_sec", None) is not None:
        engine_label += f" | min_trade_interval={args.trade_interval_sec:g}s"
    if use_risk:
        engine_label += f" | risk_engine=ON (max={risk_cfg.max_contracts} α={risk_cfg.alpha} β={risk_cfg.beta})"
    poly_source = "demo" if demo_mode else (slug or "none")
    print(
        f"Live loop: game {game_id} | poll={args.interval_sec}s | "
        f"stop_on_final={stop_on_final} | poly_source={poly_source}{engine_label}",
        flush=True,
    )

    # Start Prometheus /metrics endpoint (default 9200 — Go bot uses 9090)
    metrics_port = getattr(args, "metrics_port", 9200)
    if metrics_port:
        try:
            pm.start_metrics_server(metrics_port)
            print(f"Prometheus metrics on :{metrics_port}/metrics", flush=True)
        except OSError as e:
            if e.errno == 48:  # EADDRINUSE
                print(
                    f"[prom] port {metrics_port} in use — pick another with "
                    f"--metrics-port (e.g. 9201) or stop the other process.",
                    file=sys.stderr,
                    flush=True,
                )
            raise
    else:
        print("[prom] metrics disabled (--metrics-port 0)", flush=True)

    iteration = 0
    demo_cursor = 0  # index into demo_all_events
    last_fp: tuple | None = None

    try:
        while True:
            iteration += 1
            if args.max_iterations is not None and iteration > args.max_iterations:
                print("Stopping: max-iterations reached.", flush=True)
                break

            if demo_mode:
                # Step through pre-loaded events instead of fetching live PBP
                demo_cursor += 1
                if demo_cursor > len(demo_all_events):
                    print("[DEMO] All PBP events replayed; stopping.", flush=True)
                    break
                current_events = demo_all_events[:demo_cursor]
                snapshot = compute_snapshot_from_events(current_events, meta0)
                poly_feats = _demo_poly_features(
                    float(snapshot["time_remaining_sec"]),
                    demo_max_time,
                    demo_start_epoch,
                    demo_price_list,
                    demo_trade_list,
                )
                payload = _build_payload_from_snapshot(snapshot, poly_feats)
            else:
                pbp = _fetch_pbp_with_retries(str(game_id))
                if not pbp:
                    print(f"[{datetime.now().isoformat(timespec='seconds')}] WARN: PBP fetch failed; retrying...", flush=True)
                    time.sleep(args.interval_sec)
                    continue

                meta0 = _meta_from_pbp(str(game_id), pbp, date_str=args.date)
                meta0 = _merge_scoreboard_meta(meta0, game_sb)

                if stop_on_final and _pbp_indicates_final(pbp):
                    print("Game status is final; stopping live loop.", flush=True)
                    break

                poly_feats = {**DEFAULT_POLY_FEATS}
                if not args.no_poly and slug:
                    poly_feats = _fetch_live_polymarket_features(meta0, slug)

                try:
                    snapshot, payload = build_predict_payload(meta0, pbp, poly_feats)
                except ValueError as e:
                    print(f"[{datetime.now().isoformat(timespec='seconds')}] WARN: {e}", flush=True)
                    time.sleep(args.interval_sec)
                    continue

            fp = _fingerprint(snapshot, payload)
            if args.skip_duplicate_lines and fp == last_fp:
                time.sleep(args.interval_sec)
                continue

            try:
                result = _post_predict(args.server, payload)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                print(f"[{datetime.now().isoformat(timespec='seconds')}] WARN: /predict failed: {e}", flush=True)
                time.sleep(args.interval_sec)
                continue

            last_fp = fp

            ph = float(result.get("prob_home_win", 0.0))
            pa = float(result.get("prob_away_win", 0.0))
            edge = float(result.get("edge_vs_market", 0.0))
            side = result.get("model_side", "?")
            ts = datetime.now().isoformat(timespec="seconds")

            engine_suffix = ""
            risk_suffix = ""
            tr = None
            rs = None
            t_norm = 0.0
            if engine is not None:
                current_events = demo_all_events[:demo_cursor] if demo_mode else replay_game(pbp)
                tr = engine.tick(snapshot, payload, result, current_events)
                engine_suffix = (
                    f" | {tr.state.value} ema={tr.ema_edge:+.4f} "
                    f"nL={tr.normalised_lead:+.2f} eps={tr.epsilon_eff:.3f} "
                    f"persist={tr.persistence_ticks}/{engine._k_ticks}"
                )
                if tr.dead_zone:
                    engine_suffix += f" DZ:{tr.dead_zone}"

                # Risk engine integration
                if risk is not None and portfolio is not None:
                    regulation_secs = 2400.0
                    t_rem_game = float(snapshot["time_remaining_sec"])
                    t_norm = 1.0 - (t_rem_game / regulation_secs) if regulation_secs > 0 else 1.0
                    t_norm = max(0.0, min(1.0, t_norm))
                    now_mono = time.monotonic()

                    # News detection: large EMA edge shift (>3x epsilon)
                    if prev_ema_edge is not None:
                        ema_shift = abs(tr.ema_edge - prev_ema_edge)
                        if ema_shift > 3.0 * tr.epsilon_eff and tr.epsilon_eff > 0:
                            mag = min(1.0, ema_shift / (6.0 * tr.epsilon_eff))
                            risk.on_news(mag, now_mono)
                    prev_ema_edge = tr.ema_edge

                    tick_side = engine.last_side.upper() if engine.last_side else side.upper()
                    books = synthetic_paper_orderbook(payload["poly_price"])

                    if tr.action == "execute":
                        directive = risk.evaluate(tr.action, tick_side, t_norm, now_mono)
                        if directive.action == "buy" and directive.delta_qty > 0:
                            portfolio.on_execute(
                                tick_side, books,
                                ema_edge=tr.ema_edge, ts_iso=ts,
                                qty=directive.delta_qty,
                            )
                            risk.on_fill(directive.delta_qty, tick_side)
                        elif directive.action == "reduce" and directive.delta_qty < 0:
                            portfolio.reduce_position(abs(directive.delta_qty), books)
                            risk.on_fill(directive.delta_qty, tick_side)
                    else:
                        # Curve-driven rebalance (trimming) every tick
                        directive = risk.check_rebalance(t_norm, now_mono)
                        if directive.action == "reduce" and directive.delta_qty < 0:
                            portfolio.reduce_position(abs(directive.delta_qty), books)
                            risk.on_fill(directive.delta_qty, risk._actual_side or tick_side)

                    portfolio.mark_to_market(books)
                    rs = risk.status(t_norm, now_mono)
                    risk_suffix = (
                        f" | RISK tgt={rs['target_qty']:.1f} act={rs['actual_qty']:.1f} "
                        f"boost={rs['news_boost']:.2f} curve={rs['base_curve']:.3f}"
                    )
                    if rs["in_consolidation"]:
                        risk_suffix += " [CONSOL]"
                    risk_suffix += f" | {portfolio.summary_line(books)}"

                if tr.action == "execute":
                    engine_suffix += " >>> TRADE SIGNAL <<<"

            # ---- Prometheus metrics ----
            pm.trading_time_remaining.set(float(snapshot["time_remaining_sec"]))
            pm.trading_score_diff.set(int(snapshot["score_diff"]))
            pm.trading_poly_price.set(float(payload["poly_price"]))
            pm.confidence_score.labels(market=str(game_id), category="ncaa").set(ph)
            if engine is not None:
                pm.trading_ema_edge.set(tr.ema_edge)
                pm.trading_state.set(pm.STATE_MAP.get(tr.state.value, -1))
                if tr.action == "execute":
                    pm.hedge_actions_total.labels(action="execute").inc()
            if rs is not None and portfolio is not None:
                pm.hedge_t_norm.set(t_norm)
                pm.hedge_curve_value.set(rs["base_curve"])
                pm.hedge_target_qty.set(rs["target_qty"])
                pm.hedge_actual_qty.set(rs["actual_qty"])
                pm.hedge_news_active.set(1.0 if rs["news_boost"] > 1.01 else 0.0)
                pm.hedge_realised_pnl.set(portfolio.realised_pnl)
                pm.hedge_unrealised_pnl.set(portfolio.unrealised_pnl)
                pm.hedge_combined_pnl.set(portfolio.realised_pnl + portfolio.unrealised_pnl)

            print(
                f"{ts} | t_rem={snapshot['time_remaining_sec']:.0f}s P{snapshot['period']} | "
                f"score {snapshot['away_score']}-{snapshot['home_score']} (away-home) | "
                f"poly={payload['poly_price']:.3f} has_poly={payload['has_poly']} | "
                f"p_home={ph:.4f} p_away={pa:.4f} edge={edge:+.4f} side={side}"
                f"{engine_suffix}{risk_suffix}",
                flush=True,
            )

            time.sleep(args.interval_sec)

    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl+C).", flush=True)


def run_one_shot(args) -> None:
    meta, pbp, poly_prefetched, game_sb = _resolve_from_args(args)
    meta = _merge_scoreboard_meta(meta, game_sb)

    poly_feats = {**DEFAULT_POLY_FEATS}
    demo_mode = bool(getattr(args, "demo", None))
    if demo_mode:
        with open(args.demo) as f:
            demo_record = json.load(f)
        demo_poly = demo_record.get("polymarket", {})
        demo_price_list, demo_trade_list = parse_poly_series(demo_poly)
        demo_max_time = 2400.0
        demo_start_epoch = infer_game_start_epoch(demo_max_time, demo_price_list, demo_trade_list)
        demo_title = demo_record.get("meta", {}).get("title", args.demo)
        print(f"[DEMO] Replaying market data from {demo_title}", flush=True)
        events = replay_game(pbp)
        if events:
            snap = compute_snapshot_from_events(events, meta)
            poly_feats = _demo_poly_features(
                float(snap["time_remaining_sec"]),
                demo_max_time,
                demo_start_epoch,
                demo_price_list,
                demo_trade_list,
            )
    elif poly_prefetched and isinstance(poly_prefetched, dict) and poly_prefetched.get("found"):
        poly_feats = _poly_from_prefetched(poly_prefetched)

    slugs_cache: list[str] | None = None
    if not demo_mode and not args.no_poly and not (poly_prefetched and poly_prefetched.get("found")):
        if not args.poly_slug:
            slugs_cache = scrape_poly_slugs()
        slug = _resolve_polymarket_slug(meta, args.poly_slug, slugs_cache)
        if slug:
            poly_feats = _fetch_live_polymarket_features(meta, slug)

    snapshot, payload = build_predict_payload(meta, pbp, poly_feats)

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


def main():
    parser = argparse.ArgumentParser(description="Predict one live NCAA game via ML server")
    parser.add_argument("--game-id", help="NCAA game id")
    parser.add_argument("--date", help="Date YYYY-MM-DD (needed for scoreboard lookup)")
    parser.add_argument("--away", help="Away team hint (for scoreboard lookup)")
    parser.add_argument("--home", help="Home team hint (for scoreboard lookup)")
    parser.add_argument("--from-json", help="Use pre-fetched game JSON file")
    parser.add_argument("--poly-slug", help="Optional Polymarket event slug override")
    parser.add_argument("--no-poly", action="store_true", help="Skip Polymarket fetch")
    parser.add_argument(
        "--demo",
        metavar="PATH",
        help="Replay historical Polymarket data from a game JSON (implies --no-poly for live API)",
    )
    parser.add_argument("--require-bracket", action="store_true", default=False)
    parser.add_argument("--server", default="http://127.0.0.1:8766", help="Prediction server URL")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Continuously refresh NCAA + Polymarket and call /predict each interval",
    )
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=1.0,
        help="Seconds between NCAA/Polymarket fetch and /predict (default: 1)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Optional maximum number of live iterations (then exit)",
    )
    parser.add_argument(
        "--no-stop-on-final",
        action="store_true",
        help="Keep looping after game is final (default: stop when PBP status is final)",
    )
    parser.add_argument(
        "--skip-duplicate-lines",
        action="store_true",
        help="Skip a tick when clock/scores/poly_price unchanged (reduces noisy repeats)",
    )
    parser.add_argument(
        "--trade-engine",
        action="store_true",
        help="Enable trading decision engine (regime gate, EMA, persistence, state machine)",
    )
    parser.add_argument(
        "--trade-interval-sec",
        type=float,
        default=None,
        help="Optional minimum seconds between EXECUTE signals (wall clock; still requires full guardrails)",
    )
    parser.add_argument(
        "--epsilon-base",
        type=float,
        default=0.04,
        help="Base edge threshold in probability space (default: 0.04)",
    )
    parser.add_argument(
        "--persistence-sec",
        type=float,
        default=45.0,
        help="Seconds of same-side persistence before ARMED (default: 45)",
    )
    parser.add_argument(
        "--cooldown-min",
        type=float,
        default=3.0,
        help="Minutes of cooldown after a trade signal (default: 3)",
    )
    parser.add_argument(
        "--ema-span",
        type=int,
        default=25,
        help="EMA span in ticks for edge smoothing (default: 25)",
    )
    parser.add_argument(
        "--trade-log",
        type=str,
        default="data/trade_log.jsonl",
        help="Path for JSON-lines trade log (default: data/trade_log.jsonl)",
    )
    parser.add_argument(
        "--risk-engine",
        action="store_true",
        help="Enable risk engine for continuous position sizing (requires --trade-engine)",
    )
    parser.add_argument(
        "--risk-max-contracts",
        type=float,
        default=10.0,
        help="Peak position at curve mode (default: 10)",
    )
    parser.add_argument(
        "--risk-news-boost",
        type=float,
        default=2.0,
        help="Max news boost multiplier (default: 2.0)",
    )
    parser.add_argument(
        "--risk-curve-alpha",
        type=float,
        default=3.0,
        help="Beta distribution alpha — ramp-up speed (default: 3.0)",
    )
    parser.add_argument(
        "--risk-curve-beta",
        type=float,
        default=2.0,
        help="Beta distribution beta — ramp-down speed (default: 2.0)",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=9200,
        help="Prometheus /metrics port for this process (default: 9200; use 9090 only if Go bot is off; 0=disable)",
    )
    args = parser.parse_args()

    if args.trade_interval_sec is not None and args.trade_interval_sec <= 0:
        print("--trade-interval-sec must be positive.", file=sys.stderr)
        sys.exit(2)

    stop_on_final = not args.no_stop_on_final

    if args.live:
        if args.from_json:
            print(
                "Live mode requires live NCAA data: use --game-id or --date/--away/--home (not --from-json).",
                file=sys.stderr,
            )
            sys.exit(2)
        if args.interval_sec <= 0:
            print("--interval-sec must be positive.", file=sys.stderr)
            sys.exit(2)
        run_live_loop(args, stop_on_final=stop_on_final)
        return

    run_one_shot(args)


if __name__ == "__main__":
    main()
