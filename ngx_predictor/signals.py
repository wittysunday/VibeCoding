"""Turn model scores and indicators into BUY / HOLD / SELL / AVOID with reasons."""

from __future__ import annotations

import pandas as pd

BUY_PROB = 0.55       # model probability needed for a BUY
SELL_PROB = 0.30      # below this the model no longer supports holding
TRAILING_STOP = 0.15  # sell if price is this far below its recent peak


def red_flags(row: pd.Series) -> list[str]:
    """Warning signs that a share is risky to buy or hold, whatever the model says."""
    flags = []
    if row.get("limit_down_5d", 0) >= 2:
        flags.append("Hit the 10% daily down-limit 2+ times in 5 days (panic selling)")
    if row.get("volume_ratio", 0) > 3 and row.get("ret_1d", 0) < -0.03:
        flags.append("Volume spike >3x average on a falling price (heavy distribution)")
    if row.get("no_trade_days_20", 0) >= 8:
        flags.append("Illiquid: no trades on 8+ of the last 20 days (hard to exit)")
    if row.get("turnover_20d", 1e9) < 1_000_000:
        flags.append("Average daily turnover under N1m (thinly traded)")
    if row.get("drawdown_52w", 0) < -0.40:
        flags.append("More than 40% below its 52-week high")
    if row.get("price_vs_sma200", 0) < -0.15 and row.get("sma50_slope", 0) < 0:
        flags.append("Long-term downtrend: >15% below 200-day average and falling")
    if row.get("volatility_20d", 0) > 0.80:
        flags.append("Extreme volatility (>80% annualised)")
    if row.get("close", 1) < 1.0:
        flags.append("Penny stock (price under N1)")
    if row.get("ret_20d", 0) > 0.60:
        flags.append("Up >60% in 20 days: possible speculative bubble")
    return flags


def sell_reasons(row: pd.Series) -> list[str]:
    reasons = []
    if row.get("rsi_14", 50) > 75:
        reasons.append(f"Overbought (RSI {row['rsi_14']:.0f} > 75)")
    if row.get("macd_hist", 0) < 0 and row.get("prev_macd_hist", 0) >= 0:
        reasons.append("MACD crossed below its signal line")
    if row.get("close", 0) < row.get("sma_50", 0) and row.get("prev_close", 0) >= row.get("prev_sma_50", 0):
        reasons.append("Price broke below the 50-day average")
    if row.get("close", 0) / row.get("peak_60d", row.get("close", 1)) - 1 < -TRAILING_STOP:
        reasons.append(f"Trailing stop: >{TRAILING_STOP:.0%} below 60-day peak")
    if row.get("prob", 1) < SELL_PROB:
        reasons.append(f"Model probability fell to {row['prob']:.0%}")
    if row.get("bb_pctb", 0.5) > 1.0 and row.get("rsi_14", 50) > 70:
        reasons.append("Above upper Bollinger band and overbought")
    return reasons


def _context(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol", "date"]).copy()
    g = df.groupby("symbol")
    df["prev_macd_hist"] = g["macd_hist"].shift()
    df["prev_close"] = g["close"].shift()
    df["prev_sma_50"] = g["sma_50"].shift()
    df["peak_60d"] = g["close"].transform(lambda s: s.rolling(60, min_periods=1).max())
    return df


def classify(df: pd.DataFrame) -> pd.DataFrame:
    """Add flags, sell reasons and an action to every row (needs a `prob` column)."""
    df = _context(df)
    flags = df.apply(red_flags, axis=1)
    sells = df.apply(sell_reasons, axis=1)
    df["red_flags"] = flags.apply("; ".join)
    df["sell_reasons"] = sells.apply("; ".join)
    df["n_flags"] = flags.apply(len)

    def action(i):
        p = df.at[i, "prob"]
        if df.at[i, "n_flags"] >= 2:
            return "AVOID"
        if sells.at[i] and p < BUY_PROB:
            return "SELL"
        if p >= BUY_PROB and df.at[i, "n_flags"] == 0 and not sells.at[i]:
            return "BUY"
        return "HOLD"

    df["action"] = [action(i) for i in df.index]
    return df


def latest_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    latest = df.sort_values("date").groupby("symbol").tail(1)
    cols = ["symbol", "date", "close", "prob", "action", "rsi_14", "ret_20d", "price_vs_sma50",
            "volume_ratio", "dividend_yield", "red_flags", "sell_reasons"]
    order = {"BUY": 0, "HOLD": 1, "SELL": 2, "AVOID": 3}
    latest = latest.assign(_o=latest["action"].map(order))
    return latest.sort_values(["_o", "prob"], ascending=[True, False])[cols].reset_index(drop=True)
