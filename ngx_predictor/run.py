"""Command-line entry point.

    python -m ngx_predictor.run --prices data/prices --dividends data/dividends.csv --out output
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import dividends as dv
from .backtest import backtest
from .data_loader import load_dividends, load_prices
from .indicators import FEATURES, add_indicators
from .model import ModelConfig, add_target, feature_importance, train_and_validate
from .report import write_report
from .signals import classify, latest_recommendations


def main(argv=None):
    p = argparse.ArgumentParser(description="NGX buy/sell/red-flag/dividend signal model")
    p.add_argument("--prices", required=True, help="Price file or folder of NGX daily price lists (CSV/XLSX)")
    p.add_argument("--dividends", help="Dividend/corporate-action file or folder (CSV/XLSX)")
    p.add_argument("--out", default="output", help="Output folder")
    p.add_argument("--horizon", type=int, default=20, help="Holding period in trading days")
    p.add_argument("--target", type=float, default=0.05, help="Forward return counted as a good buy")
    p.add_argument("--top-k", type=int, default=5, help="Stocks per period in the backtest")
    args = p.parse_args(argv)

    cfg = ModelConfig(horizon=args.horizon, target_return=args.target)
    prices = load_prices(args.prices)
    divs = load_dividends(args.dividends) if args.dividends else None
    print(f"Loaded {len(prices):,} price rows for {prices['symbol'].nunique()} symbols, "
          f"{prices['date'].min().date()} to {prices['date'].max().date()}")

    df = add_indicators(prices)
    df = dv.add_dividend_features(df, divs)
    df = add_target(df, cfg)

    model, test, metrics = train_and_validate(df, cfg)
    print("Validation:", {k: (round(float(v), 3) if isinstance(v, float) else v) for k, v in metrics.items()})

    scored = df[df["history_len"] >= cfg.min_history].copy()
    scored["prob"] = model.predict_proba(scored[FEATURES])[:, 1]
    scored = classify(scored)

    # Backtest uses the walk-forward model's out-of-sample probabilities, not the refit model.
    test_cls = classify(test)
    bt, bt_stats = backtest(test_cls, cfg.horizon, args.top_k)

    recs = latest_recommendations(scored)
    as_of = prices["date"].max()
    flagged = recs[recs["red_flags"] != ""][["symbol", "close", "action", "red_flags"]]

    events = summary = opps = None
    if divs is not None and not divs.empty:
        events = dv.event_study(prices, divs)
        summary = dv.summarise_events(events)
        opps = dv.dividend_opportunities(prices, divs, summary, as_of)

    report = write_report(Path(args.out), recs, metrics, feature_importance(model, scored),
                          bt, bt_stats, summary, opps, events, flagged, as_of)
    print(f"\nTop BUY signals as of {as_of.date()}:")
    buys = recs[recs["action"] == "BUY"].head(10)[["symbol", "close", "prob"]]
    print(buys.to_string(index=False) if len(buys) else "  none")
    print(f"\nReport written to {report}")


if __name__ == "__main__":
    main()
