"""
Fetch NCAA March Madness play-by-play and matching Polymarket market data.

Discovers tournament games from the NCAA scoreboard API for a date range,
downloads play-by-play for each game, then looks up the corresponding
Polymarket per-game market (moneyline) and fetches price history + trades.

Usage:
    python fetch_games.py
    python fetch_games.py --start 2026-03-18 --end 2026-03-27 --out data/games
"""

import argparse
import json
import os
import ssl
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

# NCAA API returns errors with default python SSL on some systems
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

NCAA_BASE = "https://ncaa-api.henrygd.me"
GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
DATA_BASE = "https://data-api.polymarket.com"

_HEADERS = {"User-Agent": "penn-ml/1.0"}

REPO_ROOT = Path(__file__).resolve().parents[2]


def _get(url: str, timeout: int = 10) -> dict | list | None:
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"  WARN: GET {url[:120]}... -> {e}")
        return None


def discover_tournament_games(start: str, end: str) -> list[dict]:
    """Return list of {gameID, title, startDate, home_seo, away_seo, ...}."""
    games = []
    d = datetime.strptime(start, "%Y-%m-%d")
    end_d = datetime.strptime(end, "%Y-%m-%d")
    while d <= end_d:
        date_path = d.strftime("%Y/%m/%d")
        url = f"{NCAA_BASE}/scoreboard/basketball-men/d1/{date_path}/all-conf"
        data = _get(url)
        if data and "games" in data:
            for entry in data["games"]:
                g = entry.get("game", {})
                if not (g.get("bracketId") or g.get("championshipId")):
                    continue
                if g.get("gameState") != "final":
                    continue
                home = g.get("home", {})
                away = g.get("away", {})
                games.append({
                    "gameID": g["gameID"],
                    "title": g.get("title", ""),
                    "startDate": g.get("startDate", d.strftime("%m/%d/%Y")),
                    "home_seo": home.get("names", {}).get("seo", ""),
                    "home_short": home.get("names", {}).get("short", ""),
                    "home_score": int(home.get("score", 0) or 0),
                    "home_seed": home.get("seed", ""),
                    "away_seo": away.get("names", {}).get("seo", ""),
                    "away_short": away.get("names", {}).get("short", ""),
                    "away_score": int(away.get("score", 0) or 0),
                    "away_seed": away.get("seed", ""),
                    "home_winner": home.get("winner", False),
                })
            time.sleep(0.25)
        d += timedelta(days=1)
    return games


def find_game_on_scoreboard(
    date: str,
    away_hint: str,
    home_hint: str,
    require_bracket: bool = True,
) -> dict | None:
    """
    Find one game on a given date using team name hints.
    date format: YYYY-MM-DD
    """
    day = datetime.strptime(date, "%Y-%m-%d")
    date_path = day.strftime("%Y/%m/%d")
    url = f"{NCAA_BASE}/scoreboard/basketball-men/d1/{date_path}/all-conf"
    data = _get(url)
    if not data or "games" not in data:
        return None

    away_norm = away_hint.lower().replace(".", "").strip()
    home_norm = home_hint.lower().replace(".", "").strip()

    best = None
    best_score = -1
    for entry in data["games"]:
        g = entry.get("game", {})
        if require_bracket and not (g.get("bracketId") or g.get("championshipId")):
            continue
        home = g.get("home", {})
        away = g.get("away", {})
        home_short = (home.get("names", {}).get("short", "") or "").lower().replace(".", "")
        away_short = (away.get("names", {}).get("short", "") or "").lower().replace(".", "")
        title = (g.get("title", "") or "").lower().replace(".", "")

        score = 0
        if away_norm and (away_norm in away_short or away_short in away_norm):
            score += 2
        if home_norm and (home_norm in home_short or home_short in home_norm):
            score += 2
        if away_norm and away_norm in title:
            score += 1
        if home_norm and home_norm in title:
            score += 1
        if score > best_score:
            best_score = score
            best = g

    if not best or best_score <= 0:
        return None

    home = best.get("home", {})
    away = best.get("away", {})
    return {
        "gameID": best["gameID"],
        "title": best.get("title", ""),
        "startDate": best.get("startDate", day.strftime("%m/%d/%Y")),
        "gameState": best.get("gameState", ""),
        "home_seo": home.get("names", {}).get("seo", ""),
        "home_short": home.get("names", {}).get("short", ""),
        "home_seed": home.get("seed", ""),
        "away_seo": away.get("names", {}).get("seo", ""),
        "away_short": away.get("names", {}).get("short", ""),
        "away_seed": away.get("seed", ""),
    }


def fetch_play_by_play(game_id: str) -> dict | None:
    url = f"{NCAA_BASE}/game/{game_id}/play-by-play"
    return _get(url)


def _parse_game_date(start_date: str) -> str:
    try:
        dt = datetime.strptime(start_date, "%m/%d/%Y")
    except ValueError:
        dt = datetime.strptime(start_date, "%Y-%m-%d")
    return dt.strftime("%Y-%m-%d")


def scrape_poly_slugs() -> list[str]:
    """Scrape Polymarket bracket page for all CBB event slugs."""
    import re
    url = "https://polymarket.com/sports/cbb/bracket"
    try:
        req = urllib.request.Request(url, headers={**_HEADERS, "Accept": "text/html"})
        with urllib.request.urlopen(req, timeout=15, context=_SSL_CTX) as r:
            html = r.read().decode("utf-8", errors="replace")
        return sorted(set(re.findall(r"cbb-[a-z0-9\-]+-\d{4}-\d{2}-\d{2}", html)))
    except Exception as e:
        print(f"WARN: could not scrape Polymarket bracket page: {e}")
        return []


def _match_slug(
    slugs: list[str],
    away_seo: str,
    home_seo: str,
    away_short: str,
    home_short: str,
    date_str: str,
) -> str | None:
    """Find the best matching slug for a game from the pre-scraped list."""
    dated = [s for s in slugs if s.endswith(date_str)]
    if not dated:
        return None

    names = {n.lower().replace(".", "").replace("'", "").replace(" ", "")
             for n in [away_seo, home_seo, away_short, home_short]}

    def score(slug: str) -> int:
        parts = slug.replace(f"-{date_str}", "").replace("cbb-", "").split("-")
        body = slug.replace(f"-{date_str}", "").replace("cbb-", "")
        hits = 0
        for name in names:
            for part in parts:
                if part in name or name in part or name.startswith(part) or part.startswith(name[:3]):
                    hits += 1
                    break
            if body in name or name in body:
                hits += 1
        return hits

    best = max(dated, key=score)
    if score(best) >= 2:
        return best

    for s in dated:
        events = _get(f"{GAMMA_BASE}/events?slug={s}&limit=1")
        if events:
            title = (events[0].get("title") or "").lower()
            if any(n.lower() in title for n in [away_short, home_short]):
                return s
    return None


def fetch_polymarket_data(
    away_seo: str, home_seo: str, away_short: str, home_short: str,
    start_date: str, slug_override: str | None = None,
) -> dict:
    """Fetch Polymarket moneyline price history + trades for a game."""
    result = {"found": False, "prices": [], "trades": [], "slug": "", "conditionId": ""}

    date_str = _parse_game_date(start_date)
    slug = slug_override or f"cbb-{away_seo}-{home_seo}-{date_str}"
    result["slug"] = slug

    events = _get(f"{GAMMA_BASE}/events?slug={slug}&limit=1") if slug else None

    if not events:
        return result

    ev = events[0]
    markets = ev.get("markets", [])
    moneyline = None
    for m in markets:
        q = (m.get("question") or "").lower()
        if "spread" not in q and "o/u" not in q and "over" not in q and "under" not in q and "prop" not in q:
            moneyline = m
            break
    if not moneyline and markets:
        moneyline = markets[0]
    if not moneyline:
        return result

    result["found"] = True
    result["conditionId"] = moneyline.get("conditionId", "")

    tokens = json.loads(moneyline.get("clobTokenIds", "[]"))
    outcomes = json.loads(moneyline.get("outcomes", "[]"))
    result["outcomes"] = outcomes
    result["tokens"] = tokens

    if tokens:
        url_prices = f"{CLOB_BASE}/prices-history?market={tokens[0]}&interval=max&fidelity=500"
        prices_data = _get(url_prices)
        if prices_data:
            result["prices"] = prices_data.get("history", [])

    cid = moneyline.get("conditionId", "")
    if cid:
        all_trades = []
        for page_limit in [500]:
            url_trades = f"{DATA_BASE}/trades?market={cid}&limit={page_limit}"
            trades_data = _get(url_trades)
            if trades_data and isinstance(trades_data, list):
                all_trades.extend(trades_data)
        result["trades"] = all_trades

    return result


def main():
    parser = argparse.ArgumentParser(description="Fetch NCAA + Polymarket game data")
    parser.add_argument("--start", default="2026-03-13", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-03-27", help="End date (YYYY-MM-DD)")
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "games"),
                        help="Output directory for game JSON files")
    parser.add_argument("--delay", type=float, default=0.25,
                        help="Seconds between API calls (respect rate limits)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Discovering tournament games from {args.start} to {args.end}...")
    games = discover_tournament_games(args.start, args.end)
    print(f"Found {len(games)} tournament games\n")

    print("Scraping Polymarket for CBB event slugs...")
    poly_slugs = scrape_poly_slugs()
    print(f"Found {len(poly_slugs)} Polymarket CBB slugs\n")

    fetched = 0
    poly_matched = 0
    for i, game in enumerate(games):
        gid = game["gameID"]
        out_path = os.path.join(args.out, f"{gid}.json")

        if os.path.exists(out_path):
            print(f"[{i+1}/{len(games)}] {game['title']} — already exists, skipping")
            fetched += 1
            with open(out_path) as f:
                existing = json.load(f)
            if existing.get("polymarket", {}).get("found"):
                poly_matched += 1
            continue

        print(f"[{i+1}/{len(games)}] {game['title']} (ID: {gid})")

        pbp = fetch_play_by_play(gid)
        time.sleep(args.delay)

        matched_slug = _match_slug(
            poly_slugs,
            game["away_seo"], game["home_seo"],
            game["away_short"], game["home_short"],
            _parse_game_date(game["startDate"]),
        )

        poly = fetch_polymarket_data(
            game["away_seo"], game["home_seo"],
            game["away_short"], game["home_short"],
            game["startDate"],
            slug_override=matched_slug,
        )
        if poly["found"]:
            poly_matched += 1
            print(f"  Polymarket: {poly['slug']} — {len(poly['prices'])} prices, {len(poly['trades'])} trades")
        else:
            print(f"  Polymarket: no market found for {poly['slug']}")
        time.sleep(args.delay)

        record = {
            "meta": game,
            "ncaa_pbp": pbp,
            "polymarket": poly,
        }

        with open(out_path, "w") as f:
            json.dump(record, f)
        fetched += 1
        print(f"  Saved to {out_path}")

    print(f"\nDone: {fetched} games fetched, {poly_matched} matched to Polymarket")


if __name__ == "__main__":
    main()
