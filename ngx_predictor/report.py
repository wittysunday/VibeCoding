"""Write CSV outputs and a single self-contained HTML report."""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CSS = """
body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:24px;color:#1b1f24;background:#fff;max-width:1200px}
h1{margin-bottom:4px}h2{margin-top:32px;border-bottom:1px solid #ddd;padding-bottom:4px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}
th,td{border:1px solid #ddd;padding:5px 7px;text-align:left;vertical-align:top}th{background:#f3f4f6}
.BUY{background:#e6f4ea;font-weight:600}.SELL{background:#fdecea;font-weight:600}
.AVOID{background:#fbe3e3;color:#9b1c1c;font-weight:700}.HOLD{background:#f6f6f6}
.flag{color:#b42318}.note{color:#555;font-size:13px}.wrap{overflow-x:auto}
"""

PCT_COLS = {"prob", "ret_20d", "price_vs_sma50", "dividend_yield", "expected_runup_to_qualification",
            "expected_post_qualification_move", "yield_at_announcement", "runup_to_qualification",
            "post_qualification_move", "hold_through_return", "runup_win_rate", "best_entry_return"}


def _fmt(v, col):
    if isinstance(v, float):
        if pd.isna(v):
            return ""
        return f"{v:.1%}" if col in PCT_COLS else f"{v:,.2f}"
    return html.escape(str(v.date()) if isinstance(v, pd.Timestamp) else str(v))


def _table(df: pd.DataFrame, action_col: str | None = None) -> str:
    if df is None or df.empty:
        return "<p class='note'>No rows.</p>"
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = []
    for _, r in df.iterrows():
        cls = f" class='{r[action_col]}'" if action_col and action_col in r else ""
        cells = []
        for c in df.columns:
            cell_cls = " class='flag'" if c == "red_flags" and r[c] else ""
            cells.append(f"<td{cell_cls}>{_fmt(r[c], c)}</td>")
        body.append(f"<tr{cls}>{''.join(cells)}</tr>")
    return f"<div class='wrap'><table><tr>{head}</tr>{''.join(body)}</table></div>"


def _equity_chart(bt: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(bt["date"], bt["strategy_equity"], label="Model BUY picks (after costs)", color="#1f6feb")
    ax.plot(bt["date"], bt["market_equity"], label="Equal-weight all stocks", color="#8b949e")
    ax.set_ylabel("Growth of N1"), ax.grid(alpha=0.3), ax.legend()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return f"<img alt='Backtest equity curve' style='max-width:100%' src='data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}'>"


def write_report(out_dir: Path, recs, metrics, importance, bt, bt_stats, div_summary, div_opps,
                 events, flagged, as_of) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    recs.to_csv(out_dir / "recommendations.csv", index=False)
    flagged.to_csv(out_dir / "red_flags.csv", index=False)
    if not bt.empty:
        bt.to_csv(out_dir / "backtest.csv", index=False)
    if div_summary is not None and not div_summary.empty:
        div_summary.to_csv(out_dir / "dividend_event_summary.csv")
        events.to_csv(out_dir / "dividend_events.csv", index=False)
    if div_opps is not None and not div_opps.empty:
        div_opps.to_csv(out_dir / "dividend_opportunities.csv", index=False)

    m = "".join(f"<tr><th>{k}</th><td>{_fmt(v, 'prob' if 'rate' in k or 'return' in k else k)}</td></tr>"
                for k, v in metrics.items())
    b = "".join(f"<tr><th>{k}</th><td>{_fmt(v, 'prob' if 'return' in k or 'rate' in k or 'drawdown' in k else k)}</td></tr>"
                for k, v in bt_stats.items())
    imp = importance.head(10).rename("importance").reset_index().rename(columns={"index": "indicator"})

    div_cols = ["events", "yield_at_announcement", "runup_to_qualification", "runup_win_rate",
                "post_qualification_move", "hold_through_return", "best_entry_day", "best_entry_return"]
    div_tbl = (div_summary[div_cols].reset_index().rename(columns={"index": "symbol"})
               if div_summary is not None and not div_summary.empty else None)

    page = f"""<!doctype html><html><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>NGX Share Signals</title>
<style>{CSS}</style></head><body>
<h1>NGX Share Signals</h1><p class='note'>Data up to {as_of.date()}. Educational model output, not financial advice.
Check every pick against company filings and your own risk tolerance.</p>

<h2>1. What to buy, hold or sell now</h2>
<p class='note'>BUY = model probability &ge; 55% of a 20-day gain above the target and no red flags.
SELL = a sell rule fired and the model no longer supports holding. AVOID = two or more red flags.</p>
{_table(recs, "action")}

<h2>2. Red flags</h2>
{_table(flagged)}

<h2>3. Dividend timing</h2>
<h3>Dividends announced and not yet qualified</h3>
{_table(div_opps)}
<h3>How prices have behaved around past dividends</h3>
<p class='note'>runup_to_qualification = price change from announcement to qualification date.
post_qualification_move = change in the 5 trading days after qualification (the mark-down).
hold_through_return = buy at announcement, collect the dividend, sell 5 days after qualification.
best_entry_day = trading days after announcement that historically gave the best entry.</p>
{_table(div_tbl)}

<h2>4. How reliable is the model?</h2>
<p class='note'>Measured on the most recent dates, which the model never saw during training.
AUC 0.5 = coin flip; above 0.6 is useful for stock selection.</p>
<table>{m}</table>
<h3>Most influential indicators</h3>{_table(imp)}
<h3>Backtest of BUY signals (out of sample)</h3>
{"<p class='flag'>Fewer than 10 invested periods: too few trades to trust this backtest.</p>" if bt_stats.get("periods_invested", 0) < 10 else ""}
<table>{b}</table>{_equity_chart(bt) if not bt.empty else ''}
</body></html>"""
    path = out_dir / "report.html"
    path.write_text(page, encoding="utf-8")
    return path
