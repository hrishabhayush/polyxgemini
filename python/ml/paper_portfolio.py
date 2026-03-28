"""
Simulated Polymarket YES positions (paper only).

Holds at most one side: HOME outcome YES or AWAY outcome YES. Switching sides closes
at bid then opens at ask; repeated same-side EXECUTE adds contracts (VWAP entry).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# TODO: Optional real CLOB top-of-book bid/ask for paper fills when the order book
#   has liquidity (today we use synthetic bid/ask around mid so EXECUTE always fills).


def synthetic_paper_orderbook(
    home_mid: float,
    *,
    away_mid: float | None = None,
    half_spread: float = 0.005,
) -> dict[str, Any]:
    """
    Build bid/ask around mids so paper trades never no-op for missing CLOB levels.
    Binary market: away_mid defaults to 1 - home_mid.
    """
    hm = float(home_mid)
    hm = max(1e-6, min(1.0 - 1e-6, hm))
    am = float(away_mid) if away_mid is not None else (1.0 - hm)
    am = max(1e-6, min(1.0 - 1e-6, am))
    hs = max(1e-6, float(half_spread))

    def _clamp01(x: float) -> float:
        return max(1e-6, min(1.0 - 1e-6, x))

    return {
        "paper_synthetic_book": True,
        "home_mid": hm,
        "away_mid": am,
        "home_bid": _clamp01(hm - hs),
        "home_ask": _clamp01(hm + hs),
        "away_bid": _clamp01(am - hs),
        "away_ask": _clamp01(am + hs),
    }


@dataclass
class PaperPortfolio:
    contracts_per_trade: float = 1.0
    position: str | None = None  # "HOME" | "AWAY"
    qty: float = 0.0
    entry_price: float = 0.0
    realised_pnl: float = 0.0
    unrealised_pnl: float = 0.0
    _last_incremental_realised: float = field(default=0.0, repr=False)

    def mark_to_market(self, books: dict[str, Any]) -> None:
        if not self.position or self.qty <= 0:
            self.unrealised_pnl = 0.0
            return
        mid = (
            books.get("home_mid")
            if self.position == "HOME"
            else books.get("away_mid")
        )
        if mid is None:
            self.unrealised_pnl = 0.0
            return
        self.unrealised_pnl = self.qty * (float(mid) - self.entry_price)

    def on_execute(
        self,
        target_side: str,
        books: dict[str, Any],
        *,
        ema_edge: float,
        ts_iso: str,
    ) -> None:
        """Close/reverse at bid then open at ask; same-side EXECUTE adds at ask (VWAP)."""
        self._last_incremental_realised = 0.0
        qty = self.contracts_per_trade
        target_side = "HOME" if target_side.upper() == "HOME" else "AWAY"

        if self.position == target_side and self.qty > 0:
            ask = books.get("home_ask") if target_side == "HOME" else books.get("away_ask")
            if ask is None:
                return
            ask_f = float(ask)
            new_q = self.qty + qty
            self.entry_price = (self.qty * self.entry_price + qty * ask_f) / new_q
            self.qty = new_q
            return

        inc = 0.0
        if self.position in ("HOME", "AWAY") and self.qty > 0:
            bid = (
                books.get("home_bid")
                if self.position == "HOME"
                else books.get("away_bid")
            )
            if bid is None:
                return
            bid_f = float(bid)
            leg = self.qty * (bid_f - self.entry_price)
            inc += leg
            self.realised_pnl += leg
            self._last_incremental_realised += leg
            self.position = None
            self.qty = 0.0
            self.entry_price = 0.0

        if target_side == "HOME":
            ask = books.get("home_ask")
        else:
            ask = books.get("away_ask")
        if ask is None:
            return
        ask_f = float(ask)
        self.position = target_side
        self.qty = qty
        self.entry_price = ask_f

    def log_fields(self, books: dict[str, Any] | None) -> dict[str, Any]:
        """Flatten for JSONL merge (USD PnL on $1 face per YES contract)."""
        self.mark_to_market(books or {})
        qh = self.qty if self.position == "HOME" else 0.0
        qa = self.qty if self.position == "AWAY" else 0.0
        side = "FLAT"
        if self.position == "HOME":
            side = "HOME_YES"
        elif self.position == "AWAY":
            side = "AWAY_YES"
        out: dict[str, Any] = {
            "paper_sim": True,
            "paper_position": side,
            "paper_qty_home_yes": round(qh, 6),
            "paper_qty_away_yes": round(qa, 6),
            "paper_contracts_open": round(self.qty, 6) if self.qty else 0.0,
            "paper_entry_px": round(self.entry_price, 6) if self.qty else None,
            "paper_realised_pnl_usd": round(self.realised_pnl, 6),
            "paper_unrealised_pnl_usd": round(self.unrealised_pnl, 6),
            "paper_last_leg_realised_usd": round(self._last_incremental_realised, 6),
        }
        if books and books.get("paper_synthetic_book"):
            out["paper_synthetic_book"] = True
        if books:
            out["paper_mid_home_yes"] = (
                round(float(books["home_mid"]), 6) if books.get("home_mid") is not None else None
            )
            out["paper_mid_away_yes"] = (
                round(float(books["away_mid"]), 6) if books.get("away_mid") is not None else None
            )
        return out

    def summary_line(self, books: dict[str, Any] | None) -> str:
        d = self.log_fields(books)
        return (
            f"[PAPER] {d['paper_position']} | "
            f"realised={d['paper_realised_pnl_usd']:+.4f} "
            f"unrealised={d['paper_unrealised_pnl_usd']:+.4f} "
            f"(home_yes={d['paper_qty_home_yes']} away_yes={d['paper_qty_away_yes']})"
        )
