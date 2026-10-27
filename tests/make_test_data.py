"""Generate SYNTHETIC NGX-format files to smoke-test the pipeline.

These numbers are random and say nothing about real NGX stocks.
Usage: python tests/make_test_data.py <out_dir>
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(out: Path, n_days: int = 700, n_stocks: int = 12, seed: int = 7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    symbols = [f"TEST{i:02d}" for i in range(n_stocks)]
    (out / "prices").mkdir(parents=True, exist_ok=True)

    paths = {}
    for s in symbols:
        drift, vol = rng.normal(0.0004, 0.0006), rng.uniform(0.012, 0.03)
        r = np.clip(rng.normal(drift, vol, n_days), -0.1, 0.1)
        paths[s] = 10 * np.exp(np.cumsum(r)) * rng.uniform(0.5, 5)

    divs = []
    for s in symbols[: n_stocks // 2]:
        for year_start in (60, 310, 560):
            ann, qual = year_start, year_start + 15
            dps = round(paths[s][ann] * rng.uniform(0.03, 0.09), 2)
            paths[s][ann:qual] *= np.linspace(1, 1.04, qual - ann)  # pre-qualification run-up
            paths[s][qual + 1:] *= 1 - dps / paths[s][qual]          # mark-down after qualification
            divs.append({"Company": s, "Declaration Date": dates[ann].strftime("%d/%m/%Y"),
                         "Qualification Date": dates[qual].strftime("%d/%m/%Y"),
                         "Payment Date": dates[qual + 20].strftime("%d/%m/%Y"), "Dividend (N)": dps})

    for i, d in enumerate(dates):
        rows = []
        for s in symbols:
            c = paths[s][i]
            pc = paths[s][i - 1] if i else c
            vol = int(rng.lognormal(12, 1))
            rows.append({"Company / Symbol": s, "PClose": round(pc, 2), "Open": round(pc, 2),
                         "High": round(max(c, pc) * 1.01, 2), "Low": round(min(c, pc) * 0.99, 2),
                         "Close": round(c, 2), "Change": round(c - pc, 2), "Trades": rng.integers(5, 300),
                         "Volume": f"{vol:,}", "Value": f"{vol * c:,.2f}"})
        pd.DataFrame(rows).to_csv(out / "prices" / f"PriceList_{d:%Y-%m-%d}.csv", index=False)
    pd.DataFrame(divs).to_csv(out / "dividends.csv", index=False)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
