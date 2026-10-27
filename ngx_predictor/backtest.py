"""Out-of-sample backtest of the BUY signals on the validation period."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Approximate NGX dealing costs per side (brokerage, SEC/NGX fees, stamp duty).
COST_PER_SIDE = 0.015


def backtest(test: pd.DataFrame, horizon: int, top_k: int = 5,
             cost_per_side: float = COST_PER_SIDE) -> tuple[pd.DataFrame, dict]:
    """Every `horizon` days buy the top_k BUY-rated stocks, hold `horizon` days.

    Compared against an equal-weight portfolio of every scored stock.
    """
    dates = np.sort(test["date"].unique())[::horizon]
    rows = []
    for d in dates:
        day = test[(test["date"] == d) & test["fwd_return"].notna()]
        if day.empty:
            continue
        picks = day[day["action"] == "BUY"].nlargest(top_k, "prob")
        strat = picks["fwd_return"].mean() - 2 * cost_per_side if len(picks) else 0.0
        rows.append({"date": d, "n_picks": len(picks), "strategy": strat,
                     "market": day["fwd_return"].mean(), "picks": ", ".join(picks["symbol"])})
    res = pd.DataFrame(rows)
    if res.empty:
        return res, {}
    res["strategy_equity"] = (1 + res["strategy"]).cumprod()
    res["market_equity"] = (1 + res["market"]).cumprod()
    traded = res[res["n_picks"] > 0]
    dd = res["strategy_equity"] / res["strategy_equity"].cummax() - 1
    stats = {
        "periods": len(res),
        "periods_invested": len(traded),
        "strategy_total_return": res["strategy_equity"].iloc[-1] - 1,
        "market_total_return": res["market_equity"].iloc[-1] - 1,
        "win_rate": (traded["strategy"] > 0).mean() if len(traded) else np.nan,
        "max_drawdown": dd.min(),
    }
    return res, stats
