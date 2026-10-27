"""Dividend features and an event study of price behaviour around NGX dividends.

On NGX a buyer must hold the shares by the qualification date to receive the
dividend; the price is usually marked down after that date. The event study
measures, from history, how each stock has traded between the announcement
and the qualification date and after it, and uses that to suggest an entry
and exit for dividends that are announced but not yet qualified.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ENTRY_OFFSETS = range(0, 11)   # trading days after announcement to try entering
POST_QUAL_DAYS = 5             # trading days after qualification to measure the mark-down


def add_dividend_features(df: pd.DataFrame, dividends: pd.DataFrame | None) -> pd.DataFrame:
    """Trailing 12-month dividend yield and days until a known qualification date.

    Only dividends announced on or before each row's date are used, so the
    features carry no look-ahead.
    """
    df = df.copy()
    df["dividend_yield"] = 0.0
    df["days_to_qualification"] = np.nan
    if dividends is None or dividends.empty:
        return df

    out = []
    for sym, g in df.groupby("symbol", sort=False):
        g = g.copy()
        d = dividends[dividends["symbol"] == sym]
        if not d.empty:
            dates = g["date"].values
            dps_12m = np.zeros(len(g))
            days_to_q = np.full(len(g), np.nan)
            for _, ev in d.iterrows():
                ann = np.datetime64(ev["announcement_date"])
                dps = ev["dividend_per_share"] if pd.notna(ev["dividend_per_share"]) else 0.0
                within = (dates >= ann) & (dates < ann + np.timedelta64(365, "D"))
                dps_12m[within] += dps
                if pd.notna(ev["qualification_date"]):
                    q = np.datetime64(ev["qualification_date"])
                    pending = (dates >= ann) & (dates <= q)
                    days_to_q[pending] = ((q - dates[pending]) / np.timedelta64(1, "D")).astype(float)
            g["dividend_yield"] = dps_12m / g["close"].values
            g["days_to_qualification"] = days_to_q
        out.append(g)
    return pd.concat(out).sort_values(["symbol", "date"]).reset_index(drop=True)


def _close_on_or_after(g: pd.DataFrame, date) -> int | None:
    idx = g.index[g["date"] >= date]
    return None if len(idx) == 0 else g.index.get_loc(idx[0])


def _close_on_or_before(g: pd.DataFrame, date) -> int | None:
    idx = g.index[g["date"] <= date]
    return None if len(idx) == 0 else g.index.get_loc(idx[-1])


def event_study(prices: pd.DataFrame, dividends: pd.DataFrame) -> pd.DataFrame:
    """One row per past dividend with returns over the dividend window."""
    rows = []
    for _, ev in dividends.dropna(subset=["qualification_date"]).iterrows():
        g = prices[prices["symbol"] == ev["symbol"]].sort_values("date").reset_index(drop=True)
        if g.empty:
            continue
        a = _close_on_or_after(g, ev["announcement_date"])
        q = _close_on_or_before(g, ev["qualification_date"])
        if a is None or q is None or q <= a or q + POST_QUAL_DAYS >= len(g):
            continue
        close = g["close"].values
        dps = ev["dividend_per_share"] if pd.notna(ev["dividend_per_share"]) else 0.0
        row = {
            "symbol": ev["symbol"],
            "announcement_date": ev["announcement_date"],
            "qualification_date": ev["qualification_date"],
            "dividend_per_share": dps,
            "yield_at_announcement": dps / close[a],
            "trading_days_to_qualification": q - a,
            "pre_run_5d": close[a] / close[max(a - 5, 0)] - 1,
            "runup_to_qualification": close[q] / close[a] - 1,
            "post_qualification_move": close[q + POST_QUAL_DAYS] / close[q] - 1,
            # Bought at announcement, held through qualification, sold 5 days later, dividend received.
            "hold_through_return": (close[q + POST_QUAL_DAYS] + dps) / close[a] - 1,
        }
        for k in ENTRY_OFFSETS:
            if a + k < q:
                row[f"entry_d{k}_to_qual"] = close[q] / close[a + k] - 1
        rows.append(row)
    return pd.DataFrame(rows)


def summarise_events(events: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol averages, plus an 'ALL' row for the whole market."""
    if events.empty:
        return pd.DataFrame()
    metrics = ["yield_at_announcement", "runup_to_qualification", "post_qualification_move",
               "hold_through_return", "trading_days_to_qualification"]
    entry_cols = [c for c in events.columns if c.startswith("entry_d")]

    def agg(e: pd.DataFrame) -> pd.Series:
        s = e[metrics].mean()
        s["events"] = len(e)
        s["runup_win_rate"] = (e["runup_to_qualification"] > 0).mean()
        means = e[entry_cols].mean()
        best = means.idxmax() if means.notna().any() else None
        s["best_entry_day"] = int(best.split("_d")[1].split("_")[0]) if best else np.nan
        s["best_entry_return"] = means.max() if best else np.nan
        return s

    per_symbol = pd.DataFrame({sym: agg(e) for sym, e in events.groupby("symbol")}).T
    overall = agg(events).to_frame("ALL").T
    return pd.concat([per_symbol, overall])


def dividend_opportunities(prices: pd.DataFrame, dividends: pd.DataFrame,
                           summary: pd.DataFrame, as_of=None) -> pd.DataFrame:
    """Advice for dividends announced but not yet qualified as of the latest date."""
    as_of = pd.Timestamp(as_of) if as_of is not None else prices["date"].max()
    open_divs = dividends[(dividends["announcement_date"] <= as_of) &
                          (dividends["qualification_date"].isna() | (dividends["qualification_date"] >= as_of))]
    rows = []
    for _, ev in open_divs.iterrows():
        g = prices[(prices["symbol"] == ev["symbol"]) & (prices["date"] <= as_of)]
        if g.empty:
            continue
        price = g["close"].iloc[-1]
        dps = ev["dividend_per_share"] if pd.notna(ev["dividend_per_share"]) else 0.0
        hist = summary.loc[ev["symbol"]] if ev["symbol"] in summary.index else (
            summary.loc["ALL"] if "ALL" in summary.index else None)
        basis = "own history" if ev["symbol"] in summary.index else "market average"
        runup = hist["runup_to_qualification"] if hist is not None else np.nan
        post = hist["post_qualification_move"] if hist is not None else np.nan
        dy = dps / price
        days_since = int(((g["date"] > ev["announcement_date"])).sum())
        best_day = hist["best_entry_day"] if hist is not None else np.nan

        # Hold-through value = dividend + typical post-qualification move.
        hold_value = dy + (post if pd.notna(post) else 0.0)
        if pd.notna(runup) and runup > 0.02 and runup > hold_value:
            action = "BUY NOW, SELL ON/BEFORE QUALIFICATION DATE (capture pre-qualification run-up)"
        elif hold_value > 0.01:
            action = "BUY BEFORE QUALIFICATION DATE AND HOLD FOR DIVIDEND"
        elif dy > 0.05:
            action = "HIGH YIELD BUT PRICE USUALLY DROPS AFTER QUALIFICATION - BUY ONLY FOR LONG-TERM INCOME"
        else:
            action = "SKIP - DIVIDEND DOES NOT COVER TYPICAL MARK-DOWN"
        rows.append({
            "symbol": ev["symbol"],
            "announcement_date": ev["announcement_date"].date(),
            "qualification_date": ev["qualification_date"].date() if pd.notna(ev["qualification_date"]) else None,
            "payment_date": ev["payment_date"].date() if pd.notna(ev["payment_date"]) else None,
            "dividend_per_share": dps,
            "last_price": price,
            "dividend_yield": dy,
            "trading_days_since_announcement": days_since,
            "historical_best_entry_day": best_day,
            "expected_runup_to_qualification": runup,
            "expected_post_qualification_move": post,
            "history_basis": basis,
            "action": action,
        })
    return pd.DataFrame(rows).sort_values("dividend_yield", ascending=False) if rows else pd.DataFrame()
