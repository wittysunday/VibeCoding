"""Load NGX price and dividend files downloaded from the NGX Data Library.

NGX publishes daily price lists (one file per trading day, or a multi-day
export) and corporate disclosures with dividend dates. Column names vary a
little between exports, so every loader maps known aliases onto one schema.

Price schema:    date, symbol, open, high, low, close, volume, value, trades
Dividend schema: symbol, announcement_date, qualification_date,
                 payment_date, dividend_per_share
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PRICE_ALIASES = {
    "date": ["date", "trade date", "trading date", "tradedate", "price date"],
    "symbol": ["symbol", "ticker", "company", "security", "company / symbol",
               "company/symbol", "stock", "name"],
    "open": ["open", "opening price", "open price"],
    "high": ["high", "high price", "day high"],
    "low": ["low", "low price", "day low"],
    "close": ["close", "closing price", "close price", "last", "price"],
    "prev_close": ["pclose", "previous close", "prev close", "previous closing price"],
    "volume": ["volume", "vol", "deals volume", "quantity traded"],
    "value": ["value", "turnover", "value traded", "value (n)"],
    "trades": ["trades", "deals", "no of deals", "number of deals"],
}

DIVIDEND_ALIASES = {
    "symbol": ["symbol", "ticker", "company", "security", "stock", "name"],
    "announcement_date": ["announcement date", "declaration date", "date declared",
                          "announced", "board meeting date", "date"],
    "qualification_date": ["qualification date", "ex-date", "ex date", "ex-dividend date",
                           "closure date", "register closure", "record date"],
    "payment_date": ["payment date", "pay date", "date paid"],
    "dividend_per_share": ["dividend per share", "dps", "dividend", "amount",
                           "final dividend", "interim dividend", "dividend (n)"],
}


def _normalise(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower().replace("_", " "))


def _rename(df: pd.DataFrame, aliases: dict[str, list[str]]) -> pd.DataFrame:
    lookup = {_normalise(c): c for c in df.columns}
    mapping = {}
    for target, names in aliases.items():
        for alias in names:
            if alias in lookup and lookup[alias] not in mapping:
                mapping[lookup[alias]] = target
                break
    return df.rename(columns=mapping)


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path, thousands=",")


def _date_from_filename(path: Path):
    """NGX daily price lists usually carry the date in the file name."""
    m = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", path.stem)
    if m:
        return pd.Timestamp(f"{m[1]}-{m[2]}-{m[3]}")
    m = re.search(r"(\d{2})[-_](\d{2})[-_](\d{4})", path.stem)
    if m:
        return pd.Timestamp(f"{m[3]}-{m[2]}-{m[1]}")
    return pd.NaT


def _to_number(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(r"[,\sN₦]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def load_prices(source: str | Path) -> pd.DataFrame:
    """Load one file or every CSV/XLSX file in a folder into a tidy price table."""
    source = Path(source)
    files = sorted(source.glob("*.csv")) + sorted(source.glob("*.xls*")) if source.is_dir() else [source]
    if not files:
        raise FileNotFoundError(f"No CSV/XLSX price files found in {source}")

    frames = []
    for f in files:
        df = _rename(_read_any(f), PRICE_ALIASES)
        if "date" not in df.columns:
            df["date"] = _date_from_filename(f)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    missing = {"date", "symbol", "close"} - set(df.columns)
    if missing:
        raise ValueError(f"Price data is missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    for col in ["open", "high", "low", "close", "prev_close", "volume", "value", "trades"]:
        if col in df.columns:
            df[col] = _to_number(df[col])

    # Fill gaps NGX lists leave for untraded securities.
    for col in ["open", "high", "low"]:
        if col not in df.columns:
            df[col] = df["close"]
        df[col] = df[col].fillna(df["close"])
    for col in ["volume", "value", "trades"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    df = df.dropna(subset=["date", "close"])
    df = df[df["close"] > 0]
    df = df.drop_duplicates(subset=["symbol", "date"], keep="last")
    cols = ["date", "symbol", "open", "high", "low", "close", "volume", "value", "trades"]
    return df[cols].sort_values(["symbol", "date"]).reset_index(drop=True)


def load_dividends(source: str | Path) -> pd.DataFrame:
    source = Path(source)
    files = sorted(source.glob("*.csv")) + sorted(source.glob("*.xls*")) if source.is_dir() else [source]
    df = pd.concat([_rename(_read_any(f), DIVIDEND_ALIASES) for f in files], ignore_index=True)

    missing = {"symbol", "announcement_date"} - set(df.columns)
    if missing:
        raise ValueError(f"Dividend data is missing required columns: {sorted(missing)}")

    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    for col in ["announcement_date", "qualification_date", "payment_date"]:
        if col not in df.columns:
            df[col] = pd.NaT
        df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
    if "dividend_per_share" not in df.columns:
        df["dividend_per_share"] = float("nan")
    df["dividend_per_share"] = _to_number(df["dividend_per_share"])

    cols = ["symbol", "announcement_date", "qualification_date", "payment_date", "dividend_per_share"]
    return df.dropna(subset=["announcement_date"])[cols].sort_values(["symbol", "announcement_date"]).reset_index(drop=True)
