"""Score the latest active snapshot against the running ML server."""

import glob
import json
import math
import datetime as dt
import urllib.request


def jump_enc(s):
    s = (s or "").strip().lower()
    return 2 if s == "sustained" else 1 if s == "reversed" else 0


def market_age_days(created_at, snapshot_at, end_date):
    try:
        c = dt.datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        # Prefer end_date; fall back to snapshot time (matches training)
        ref_str = end_date if end_date else snapshot_at
        if ref_str:
            e = dt.datetime.fromisoformat(ref_str.replace("Z", "+00:00"))
            return max(0.0, (e - c).total_seconds() / 86400.0)
    except Exception:
        pass
    return 0.0


files = sorted(glob.glob("data/snapshots/active_*.jsonl"))
if not files:
    raise SystemExit("No active_*.jsonl files found in data/snapshots/")
path = files[-1]
print(f"Scoring: {path}")

def is_binary_question(question: str) -> bool:
    """True for Yes/No questions; False for multi-outcome (e.g. 'X vs Y' sports matchups)."""
    q = question.strip()
    # Sports matchup pattern: "Team A vs. Team B" or "Team A vs Team B"
    import re
    if re.search(r'\bvs\.?\s+\S', q, re.IGNORECASE):
        return False
    return True


rows = []
errors = 0
skipped_nonbinary = 0
with open(path) as f:
    for line in f:
        if not line.strip():
            continue
        m = json.loads(line)
        if not is_binary_question(m.get("question", "")):
            skipped_nonbinary += 1
            continue
        cp = float(m.get("current_price", 0.0) or 0.0)
        tv = float(m.get("total_volume", 0.0) or 0.0)
        tc = int(m.get("trade_count", 0) or 0)

        payload = {
            "current_price": cp,
            "hurst_exp": float(m.get("hurst_exp", 0.0) or 0.0),
            "vol_ratio": float(m.get("vol_ratio", 0.0) or 0.0),
            "jump_result_enc": jump_enc(m.get("jump_result")),
            "trade_count": tc,
            "total_volume": tv,
            "log_volume": math.log1p(tv),
            "kyles_lambda": float(m.get("kyles_lambda", 0.0) or 0.0),
            "vpin": float(m.get("vpin", 0.0) or 0.0),
            "buy_fraction": float(m.get("buy_fraction", 0.0) or 0.0),
            "wallet_hhi": float(m.get("wallet_hhi", 0.0) or 0.0),
            "market_age_days": market_age_days(
                m.get("created_at", ""),
                m.get("snapshot_at", ""),
                m.get("end_date", ""),
            ),
            "avg_trade_size": tv / (tc + 1),
            "log_avg_trade_size": math.log1p(tv / (tc + 1)),
            "log_trade_count": math.log1p(tc),
            "has_hurst": 1 if float(m.get("hurst_exp", 0.0) or 0.0) != 0.0 else 0,
            "price_distance_from_50": abs(cp - 0.5),
        }

        try:
            req = urllib.request.Request(
                "http://127.0.0.1:8766/predict",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                pred = json.loads(r.read().decode())
        except Exception as e:
            errors += 1
            continue

        edge = max(pred["edge_yes"], pred["edge_no"])
        side = "YES" if pred["edge_yes"] >= pred["edge_no"] else "NO"
        rows.append((edge, side, m.get("slug", ""), cp, pred, m))

if skipped_nonbinary:
    print(f"Skipped {skipped_nonbinary} non-binary markets (e.g. sports matchups)")
if errors:
    print(f"Warning: {errors} markets failed (server down?)")

# Quality filters
rows = [x for x in rows if x[5].get("trade_count", 0) >= 50 and x[5].get("total_volume", 0) >= 5000]
rows.sort(key=lambda x: x[0], reverse=True)

print(f"\nTop 15 by edge ({len(rows)} markets passed filters):")
for edge, side, slug, cp, pred, _ in rows[:15]:
    print(f"  {edge:+.4f}  buy={side:3}  price={cp:.4f}  p_yes={pred['prob_yes']:.4f}  slug={slug}")
