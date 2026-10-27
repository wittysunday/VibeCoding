# NGX Share Signals

A model built on historical Nigerian Exchange (NGX) prices that:

1. **Ranks shares to buy.** A gradient-boosted classifier estimates the probability that each share gains more than 5% over the next 20 trading days. It learns from 21 technical and dividend indicators.
2. **Says when to sell.** Sell rules fire on overbought RSI, a bearish MACD cross, a break below the 50-day average, a 15% trailing stop, or a drop in the model's probability.
3. **Raises red flags.** It warns on repeated 10% limit-down days, a volume spike on a falling price, illiquidity, deep drawdowns, a long-term downtrend, extreme volatility, penny stocks, and bubble-like run-ups.
4. **Times dividends.** An event study measures how each share has moved between the dividend announcement and the qualification date, and after it. For every dividend announced but not yet qualified, it then advises whether to buy and hold for the dividend, trade the pre-qualification run-up, or skip.

## 1. Get the data

Download from <https://ngxgroup.com/exchange/data/data-library/>:

- **Daily price lists** (CSV or XLSX), as many days as you can get; two or more years is recommended. Put them all in one folder, e.g. `data/prices/`. If a file has no date column, the date is read from its name (e.g. `PriceList_2025-03-14.csv`).
- **Corporate actions / dividends.** Use one file with the symbol, the announcement (declaration) date, the qualification (closure) date, the payment date and the dividend per share. Save it as `data/dividends.csv`.

Common NGX column names (`Company / Symbol`, `PClose`, `Close`, `Volume`, `Value`, `Deals`, `Qualification Date`, `Dividend (N)` and so on) are recognised automatically. See the aliases in `data_loader.py` to add others.

## 2. Run

```bash
pip install -r requirements.txt
python -m ngx_predictor.run --prices data/prices --dividends data/dividends.csv --out output
```

Options: `--horizon 20` (holding days), `--target 0.05` (return counted as a good buy), `--top-k 5` (picks per period in the backtest).

## 3. Outputs (`output/`)

| File | Contents |
|---|---|
| `report.html` | Everything below in one page |
| `recommendations.csv` | Latest BUY / HOLD / SELL / AVOID per share, with probability, indicators and reasons |
| `red_flags.csv` | Shares with warning signs |
| `dividend_opportunities.csv` | Open dividends: yield, expected run-up and mark-down, suggested action |
| `dividend_event_summary.csv` | Per-share history around past dividends, including the best entry day after announcement |
| `backtest.csv` | Out-of-sample backtest of the BUY picks against an equal-weight market, after about 1.5% dealing costs per side |

## How to read the model's reliability

The model is validated walk-forward: it is trained on older dates and tested on the most recent 25% it never saw. Check **AUC** (0.5 is a coin flip; above about 0.6 is useful) and the **top-decile hit rate** against the base rate before trusting the picks. A backtest with only a handful of trades is luck, not evidence.

## Testing

`tests/make_test_data.py` writes **synthetic** files in NGX format, for checking that the pipeline runs. Its numbers are random and mean nothing about real shares.

*Educational tool, not financial advice.*
