"""Technical indicators computed per symbol on daily NGX prices."""

from __future__ import annotations

import numpy as np
import pandas as pd

# NGX daily price movement limit (+/-10%).
DAILY_LIMIT = 0.10


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(loss.notna())


def _add_indicators(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    c, h, l, v = g["close"], g["high"], g["low"], g["volume"]
    ret = c.pct_change()

    g["ret_1d"] = ret
    g["ret_5d"] = c.pct_change(5)
    g["ret_20d"] = c.pct_change(20)
    g["ret_60d"] = c.pct_change(60)

    for n in (10, 20, 50, 200):
        g[f"sma_{n}"] = c.rolling(n, min_periods=n).mean()
    g["price_vs_sma20"] = c / g["sma_20"] - 1
    g["price_vs_sma50"] = c / g["sma_50"] - 1
    g["price_vs_sma200"] = c / g["sma_200"] - 1
    g["sma50_slope"] = g["sma_50"].pct_change(10)

    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    g["macd"] = (ema12 - ema26) / c
    g["macd_signal"] = g["macd"].ewm(span=9, adjust=False).mean()
    g["macd_hist"] = g["macd"] - g["macd_signal"]

    g["rsi_14"] = _rsi(c)

    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    g["bb_pctb"] = (c - (mid - 2 * sd)) / (4 * sd).replace(0, np.nan)
    g["bb_width"] = 4 * sd / mid

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    g["atr_pct"] = tr.rolling(14).mean() / c

    g["volatility_20d"] = ret.rolling(20).std() * np.sqrt(252)
    g["volume_ratio"] = v / v.rolling(20).mean().replace(0, np.nan)
    g["obv_slope"] = (np.sign(ret.fillna(0)) * v).cumsum().diff(20) / v.rolling(20).sum().replace(0, np.nan)
    g["no_trade_days_20"] = (v <= 0).astype(float).rolling(20).sum()
    g["turnover_20d"] = g["value"].rolling(20).mean()

    g["high_52w"] = c.rolling(250, min_periods=20).max()
    g["drawdown_52w"] = c / g["high_52w"] - 1
    g["limit_down"] = (ret <= -DAILY_LIMIT + 0.005).astype(int)
    g["limit_up"] = (ret >= DAILY_LIMIT - 0.005).astype(int)
    g["limit_down_5d"] = g["limit_down"].rolling(5).sum()
    return g


def add_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([_add_indicators(g) for _, g in prices.groupby("symbol")], ignore_index=True)


FEATURES = [
    "ret_5d", "ret_20d", "ret_60d",
    "price_vs_sma20", "price_vs_sma50", "price_vs_sma200", "sma50_slope",
    "macd", "macd_hist", "rsi_14", "bb_pctb", "bb_width", "atr_pct",
    "volatility_20d", "volume_ratio", "obv_slope", "no_trade_days_20",
    "drawdown_52w", "limit_down_5d",
    "dividend_yield", "days_to_qualification",
]
