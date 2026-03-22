"""
End-to-end sentiment test: fetches real articles from NewsAPI, GDELT, and Reddit,
scores them all with the local FinBERT server, and pretty-prints the results.

Requirements:
  - FinBERT server running:  uvicorn server:app --host 127.0.0.1 --port 8765
  - NewsAPI key in env:      export NEWSAPI_KEY=<your_key>
  - GDELT and Reddit require no API key

Run:
    python test_finbert.py
    python test_finbert.py --query "fed rate cut" --max 8
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import httpx

# ── Config ──────────────────────────────────────────────────────────────────

FINBERT_URL  = os.getenv("FINBERT_URL", "http://localhost:8765")
NEWSAPI_KEY  = os.getenv("NEWSAPI_KEY", "")
NEWSAPI_BASE = "https://newsapi.org/v2"
GDELT_BASE   = "https://api.gdeltproject.org/api/v2/doc/doc"
REDDIT_BASE  = "https://www.reddit.com"
REDDIT_SUBS  = ["politics", "worldnews"]
USER_AGENT   = "sentimentbot/1.0"

# ── API fetchers ─────────────────────────────────────────────────────────────

def fetch_newsapi(query: str, n: int) -> list[dict]:
    if not NEWSAPI_KEY:
        print("  [SKIP] NEWSAPI_KEY not set — skipping NewsAPI")
        return []
    params = {"q": query, "sortBy": "publishedAt", "language": "en", "pageSize": n}
    resp = httpx.get(f"{NEWSAPI_BASE}/everything", params=params,
                     headers={"X-Api-Key": NEWSAPI_KEY}, timeout=10)
    resp.raise_for_status()
    articles = resp.json().get("articles", [])
    return [
        {
            "title":       a.get("title", ""),
            "description": a.get("description", "") or "",
            "source":      a.get("source", {}).get("name", ""),
            "url":         a.get("url", ""),
            "publishedAt": a.get("publishedAt", ""),
        }
        for a in articles
    ]


def fetch_gdelt(query: str, n: int) -> list[dict]:
    params = {"query": query, "mode": "artlist", "format": "json",
              "maxrecords": n, "sort": "DateDesc"}
    resp = httpx.get(GDELT_BASE, params=params, timeout=10)
    resp.raise_for_status()
    articles = resp.json().get("articles") or []
    return [
        {
            "title":  a.get("title", ""),
            "domain": a.get("domain", ""),
            "url":    a.get("url", ""),
            "seendate": a.get("seendate", ""),
        }
        for a in articles
    ]


def fetch_reddit(query: str, n: int) -> list[dict]:
    posts = []
    seen = set()
    per_sub = max(1, n // len(REDDIT_SUBS) + 1)
    headers = {"User-Agent": USER_AGENT}

    for sub in REDDIT_SUBS:
        url = f"{REDDIT_BASE}/r/{sub}/search.json"
        params = {"q": query, "sort": "new", "restrict_sr": "true", "limit": per_sub}
        try:
            resp = httpx.get(url, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            children = resp.json().get("data", {}).get("children", [])
            for child in children:
                d = child.get("data", {})
                post_url = d.get("url", "")
                if post_url in seen:
                    continue
                seen.add(post_url)
                posts.append({
                    "title":       d.get("title", ""),
                    "selftext":    (d.get("selftext") or "")[:200],
                    "subreddit":   d.get("subreddit", ""),
                    "score":       d.get("score", 0),
                    "num_comments": d.get("num_comments", 0),
                    "url":         post_url,
                })
                if len(posts) >= n:
                    return posts
        except Exception as e:
            print(f"  [WARN] Reddit r/{sub} error: {e}")
        time.sleep(0.15)  # respect ~10 req/min unauthenticated rate limit

    return posts


# ── FinBERT scoring ──────────────────────────────────────────────────────────

def score_texts(texts: list[str]) -> list[dict]:
    resp = httpx.post(f"{FINBERT_URL}/score", json={"texts": texts}, timeout=60)
    resp.raise_for_status()
    return resp.json()["results"]


# ── Pretty printing ──────────────────────────────────────────────────────────

LABEL_MARKER = {"positive": "+", "negative": "-", "neutral": "~"}
LABEL_PAD    = {"positive": "POSITIVE", "negative": "NEGATIVE", "neutral": "NEUTRAL "}

def print_source_table(source: str, items: list[dict], texts: list[str], scores: list[dict]):
    print(f"\n{'═'*70}")
    print(f"  SOURCE: {source}")
    print(f"{'═'*70}")
    if not items:
        print("  (no results)")
        return
    print(f"  {'LABEL':<11} {'CONF':>6}   TITLE / TEXT")
    print(f"  {'─'*64}")
    for item, text, s in zip(items, texts, scores):
        marker = LABEL_MARKER.get(s["label"], "~")
        label  = LABEL_PAD.get(s["label"], s["label"].upper())
        conf   = f"{s['score']*100:.1f}%"
        title  = (item.get("title") or text)[:55]
        if len(item.get("title","")) > 55:
            title += "…"
        print(f"  {marker} {label:<10} {conf:>6}   {title}")
    print()


def print_aggregate(query: str, all_scores: list[dict]):
    counts  = {"positive": 0, "negative": 0, "neutral": 0}
    conf_sum = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    for s in all_scores:
        lbl = s["label"]
        counts[lbl]   += 1
        conf_sum[lbl] += s["score"]

    total = len(all_scores) or 1
    avg_conf = sum(s["score"] for s in all_scores) / total
    dom = max(counts, key=counts.get)

    signal = {
        "query":          query,
        "total_articles": len(all_scores),
        "positive":       counts["positive"],
        "negative":       counts["negative"],
        "neutral":        counts["neutral"],
        "avg_confidence": round(avg_conf, 3),
        "dominant":       dom,
        "timestamp":      datetime.now(timezone.utc).isoformat(),
    }
    print(f"\n{'═'*70}")
    print(f"  AGGREGATED SIGNAL")
    print(f"{'═'*70}")
    print(json.dumps(signal, indent=2))
    print()
    return signal


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FinBERT sentiment test")
    parser.add_argument("--query", default="trump visit china",
                        help="Market query to test (default: 'trump visit china')")
    parser.add_argument("--max", type=int, default=5,
                        help="Max articles per source (default: 5)")
    args = parser.parse_args()

    query = args.query
    n     = args.max

    # 1. Health check
    try:
        health = httpx.get(f"{FINBERT_URL}/health", timeout=5).json()
        if not health.get("model_loaded"):
            print(f"[ERROR] FinBERT model not loaded. Check server logs.")
            sys.exit(1)
        print(f"[OK] FinBERT server healthy (model_loaded=true)")
    except Exception as e:
        print(f"[ERROR] FinBERT server unreachable at {FINBERT_URL}: {e}")
        print("        Start with: uvicorn server:app --host 127.0.0.1 --port 8765")
        sys.exit(1)

    print(f"\n[INFO] Query: \"{query}\"  |  Max per source: {n}")
    print(f"[INFO] Fetching articles from NewsAPI, GDELT, Reddit...\n")

    # 2. Fetch articles
    newsapi_items  = fetch_newsapi(query, n)
    gdelt_items    = fetch_gdelt(query, n)
    reddit_items   = fetch_reddit(query, n)

    print(f"  NewsAPI: {len(newsapi_items)} articles")
    print(f"  GDELT:   {len(gdelt_items)} articles")
    print(f"  Reddit:  {len(reddit_items)} posts")

    # 3. Build text strings for FinBERT
    def newsapi_text(a):
        return (a["title"] + ". " + a["description"]).strip(". ")

    def gdelt_text(a):
        return a["title"]

    def reddit_text(p):
        base = p["title"]
        if p.get("selftext"):
            base += ". " + p["selftext"][:200]
        return base

    newsapi_texts = [newsapi_text(a) for a in newsapi_items]
    gdelt_texts   = [gdelt_text(a)   for a in gdelt_items]
    reddit_texts  = [reddit_text(p)  for p in reddit_items]

    # 4. Score all sources
    all_scores = []
    newsapi_scores, gdelt_scores, reddit_scores = [], [], []

    if newsapi_texts:
        print("\n[INFO] Scoring NewsAPI with FinBERT...")
        newsapi_scores = score_texts(newsapi_texts)
        all_scores.extend(newsapi_scores)

    if gdelt_texts:
        print("[INFO] Scoring GDELT with FinBERT...")
        gdelt_scores = score_texts(gdelt_texts)
        all_scores.extend(gdelt_scores)

    if reddit_texts:
        print("[INFO] Scoring Reddit with FinBERT...")
        reddit_scores = score_texts(reddit_texts)
        all_scores.extend(reddit_scores)

    # 5. Pretty print per-source tables
    print_source_table("NewsAPI", newsapi_items, newsapi_texts, newsapi_scores)
    print_source_table("GDELT",   gdelt_items,   gdelt_texts,   gdelt_scores)
    print_source_table("Reddit",  reddit_items,  reddit_texts,  reddit_scores)

    # 6. Aggregate
    if all_scores:
        print_aggregate(query, all_scores)
    else:
        print("\n[WARN] No articles scored — check API keys and server status.")


if __name__ == "__main__":
    main()
