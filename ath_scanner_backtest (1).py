"""
ATH Scanner - 12-month backtest visual
======================================

Replays the All-Time-High scanner (same signal + same filters) on every trading
day over the last N months, then builds an interactive HTML report with:

  1. Daily stacked bar  - number of tickers that passed the scanner each day
  2. Weekly stacked bar - sum of daily hits per week (Mon-Fri)

Each bar is split into segments, one per ticker, labeled with the ticker symbol.
Segment color = forward return after the hit (default 10 trading days):
red = negative, gray = flat, green = positive, darker green = bigger gain.
Hits too recent to have a forward result yet are shown hatched.

Outputs (written next to this script):
  ath_backtest_report.html  - open in any browser (zoom in to reveal labels)
  ath_backtest_hits.csv     - every hit with indicators + forward returns

Requires: pip install yfinance pandas requests plotly lxml
"""

import json
import os
import warnings
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

# ============================================================
# SCANNER SETTINGS (keep in sync with scanner_All-Time-High.py)
# ============================================================

DATA_PERIOD = "max"
LOOKBACK_DAYS = 10
RVOL_LOOKBACK = 20

FILTERS_ACTIVE = True
FILTER_MAX_DAYS_SINCE_ATH = 3
FILTER_MIN_RVOL = 1.0
FILTER_MIN_TODAY_RETURN = None
FILTER_MIN_RSI = 50
FILTER_MAX_RSI = 80
FILTER_REQUIRE_MACD_BULL = True
FILTER_REQUIRE_STOCH_BULL = True

SECTOR_ETF_MAP = {
    "Energy": "XLE",
    "Materials": "XLB",
    "Industrials": "XLI",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Health Care": "XLV",
    "Financials": "XLF",
    "Information Technology": "XLK",
    "Communication Services": "XLC",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
}

# ============================================================
# BACKTEST / CHART SETTINGS
# ============================================================

BACKTEST_MONTHS = 12
FORWARD_DAYS = [5, 10, 20]   # forward returns added to the CSV + summary
COLOR_FWD_DAYS = 10          # which forward return colors the bars
COLOR_CAP_PCT = 10           # +/- this % (or beyond) gets the darkest green / red

# Output location. Defaults: next to the script (.py) or the notebook's folder (Jupyter).
# The GitHub Pages workflow overrides these with environment variables.
if os.environ.get("BACKTEST_OUTPUT_DIR"):
    OUT_DIR = Path(os.environ["BACKTEST_OUTPUT_DIR"]).resolve()
else:
    try:
        OUT_DIR = Path(__file__).resolve().parent
    except NameError:
        OUT_DIR = Path.cwd()
OUT_DIR.mkdir(parents=True, exist_ok=True)
HTML_OUT = OUT_DIR / os.environ.get("BACKTEST_HTML_NAME", "ath_backtest_report.html")
CSV_OUT = OUT_DIR / "ath_backtest_hits.csv"

# How this scanner is listed when embedded in the S&P 500 dashboard (site/scanners/...)
SCANNER_TITLE = "All-Time-High Scanner - 12-Month Backtest"
SCANNER_ORDER = 10           # lower numbers appear first among scanner sections

# "inline" = report works fully offline (~5 MB file); "cdn" = small file that loads
# Plotly from the web (used for GitHub Pages)
PLOTLY_JS = os.environ.get("BACKTEST_PLOTLY_JS", "inline")

# Diverging scale: dark red -> red -> neutral gray (0%) -> green -> dark green
GRADIENT = [(-1.0, "#8e1b1b"), (-0.5, "#e0584e"), (0.0, "#e4e2dc"),
            (0.5, "#4fae68"), (1.0, "#0b5a24")]
PENDING_COLOR = "#f1f0ec"    # hit too recent for a forward result (hatched)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"


# ============================================================
# DATA LOADERS + INDICATORS (identical to the live scanner)
# ============================================================

def get_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    r.raise_for_status()
    df = pd.read_html(StringIO(r.text))[0]
    df.columns = [c.strip() for c in df.columns]

    ticker_col = name_col = sector_col = None
    for c in df.columns:
        if "Symbol" in c or "Ticker" in c:
            ticker_col = c
        if "Security" in c or ("Name" in c and "Sub" not in c) or "Company" in c:
            name_col = c
        if "GICS Sector" in c:
            sector_col = c

    sp500 = df[[ticker_col, name_col, sector_col]].copy()
    sp500.columns = ["Ticker", "Company", "Sector"]
    sp500["Ticker"] = sp500["Ticker"].str.replace(".", "-", regex=False)
    sp500["Sector_ETF"] = sp500["Sector"].map(SECTOR_ETF_MAP).fillna("N/A")
    return sp500


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    return 100 - (100 / (1 + avg_gain / avg_loss))


def macd(series, fast=12, slow=26, signal_period=9):
    macd_line = series.ewm(span=fast, adjust=False).mean() - series.ewm(span=slow, adjust=False).mean()
    return macd_line, macd_line.ewm(span=signal_period, adjust=False).mean()


def stochastic(high, low, close, k_period=14, smooth_k=3, d_period=3):
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = (100 * (close - lowest_low) / (highest_high - lowest_low)).rolling(smooth_k).mean()
    return k, k.rolling(d_period).mean()


# ============================================================
# BACKTEST: replay the scanner on every day in the window
# ============================================================

def scan_history(full_df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per trading day with the scanner's fields as of that day.

    Every indicator is causal (uses only data up to that day), so computing it
    once on the full history and reading day D gives exactly what the live
    scanner would have printed if it had run on day D.
    """
    close, high, low, vol = full_df["Close"], full_df["High"], full_df["Low"], full_df["Volume"]

    recent_high = close.rolling(LOOKBACK_DAYS).max()
    prior_ath = close.shift(LOOKBACK_DAYS).expanding().max()   # max of close.iloc[:-LOOKBACK_DAYS]
    # position of the (first) max inside the 10-day window -> trading days since ATH
    argmax_pos = close.rolling(LOOKBACK_DAYS).apply(np.argmax, raw=True)
    days_since = (LOOKBACK_DAYS - 1) - argmax_pos

    avg_vol = vol.shift(1).rolling(RVOL_LOOKBACK, min_periods=1).mean()
    rvol = vol / avg_vol.where(avg_vol > 0)

    macd_line, signal_line = macd(close)
    k, d = stochastic(high, low, close)

    out = pd.DataFrame({
        "Close": close,
        "Is_New_ATH": recent_high > prior_ath,
        f"{LOOKBACK_DAYS}D_High_Close": recent_high.round(2),
        "Days_Since_ATH": days_since,
        "Today_Return%": (close.pct_change() * 100).round(2),
        "Volume": vol,
        "RVOL": rvol.round(2),
        "RSI": rsi(close).round(1),
        "MACD_Bull": macd_line > signal_line,
        "Stoch_Bull": k > d,
    })
    for n in FORWARD_DAYS:
        out[f"Fwd_{n}D%"] = ((close.shift(-n) / close - 1) * 100).round(2)
    return out


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["Is_New_ATH"]]
    if not FILTERS_ACTIVE:
        return out
    if FILTER_MAX_DAYS_SINCE_ATH is not None:
        out = out[out["Days_Since_ATH"] <= FILTER_MAX_DAYS_SINCE_ATH]
    if FILTER_MIN_RVOL is not None:
        out = out[out["RVOL"] >= FILTER_MIN_RVOL]
    if FILTER_MIN_TODAY_RETURN is not None:
        out = out[out["Today_Return%"] >= FILTER_MIN_TODAY_RETURN]
    if FILTER_MIN_RSI is not None:
        out = out[out["RSI"] >= FILTER_MIN_RSI]
    if FILTER_MAX_RSI is not None:
        out = out[out["RSI"] <= FILTER_MAX_RSI]
    if FILTER_REQUIRE_MACD_BULL is not None:
        out = out[out["MACD_Bull"] == FILTER_REQUIRE_MACD_BULL]
    if FILTER_REQUIRE_STOCH_BULL is not None:
        out = out[out["Stoch_Bull"] == FILTER_REQUIRE_STOCH_BULL]
    return out


def run_backtest(data, sp500: pd.DataFrame, start: pd.Timestamp):
    hits, all_days = [], set()

    for _, row in sp500.iterrows():
        t = row["Ticker"]
        try:
            full_df = data[t].dropna()
        except KeyError:
            continue
        if len(full_df) <= LOOKBACK_DAYS:
            continue

        hist = scan_history(full_df)
        hist = hist[hist.index >= start]
        all_days.update(hist.index)

        passed = apply_filters(hist)
        if passed.empty:
            continue
        passed = passed.copy()
        passed.insert(0, "Ticker", t)
        passed.insert(1, "Company", row["Company"])
        passed.insert(2, "Sector_ETF", row["Sector_ETF"])
        passed.insert(3, "Sector", row["Sector"])
        hits.append(passed)

    trading_days = pd.DatetimeIndex(sorted(all_days))
    if not hits:
        return pd.DataFrame(), trading_days

    hits = pd.concat(hits)
    hits.index.name = "Date"
    hits = hits.reset_index().drop(columns=["Is_New_ATH"])
    hits["Days_Since_ATH"] = hits["Days_Since_ATH"].astype(int)
    return hits.sort_values(["Date", "Ticker"]).reset_index(drop=True), trading_days


# ============================================================
# CHARTS
# ============================================================

def _hex_to_rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def fwd_color(v):
    """Forward return % -> hex color on the red/gray/green gradient (None/NaN -> pending)."""
    if v is None or pd.isna(v):
        return PENDING_COLOR
    x = max(-1.0, min(1.0, float(v) / COLOR_CAP_PCT))
    for (p0, c0), (p1, c1) in zip(GRADIENT, GRADIENT[1:]):
        if x <= p1:
            f = (x - p0) / (p1 - p0)
            r0, r1 = _hex_to_rgb(c0), _hex_to_rgb(c1)
            return "#%02x%02x%02x" % tuple(round(a + (b - a) * f) for a, b in zip(r0, r1))
    return GRADIENT[-1][1]


def label_color(v):
    """Dark text on the light middle of the scale, white text on saturated ends."""
    if v is None or pd.isna(v) or abs(v) / COLOR_CAP_PCT < 0.4:
        return INK
    return "#ffffff"


PLOTLY_SCALE = [[(p + 1) / 2, c] for p, c in GRADIENT]


def stacked_bar(frame: pd.DataFrame, x_col: str, y_col: str, x_all, title: str,
                hover_fmt: str, totals: pd.Series, unique: pd.Series, bar_width_ms=None,
                annotate_totals=False):
    """frame needs a 'Color_Val' column: the forward return % used for the segment color."""
    fig = go.Figure()
    # tickers with the best average forward return sit at the bottom of the stacks
    order = frame.groupby("Ticker")["Color_Val"].mean().sort_values(ascending=False, na_position="last")

    for t in order.index:
        sub = frame[frame["Ticker"] == t]
        vals = sub["Color_Val"].tolist()
        fig.add_bar(
            x=sub[x_col], y=sub[y_col], name=t, showlegend=False,
            marker=dict(
                color=[fwd_color(v) for v in vals],
                line=dict(color=SURFACE, width=0.5),
                pattern=dict(shape=["/" if pd.isna(v) else "" for v in vals],
                             fgcolor="#a9a8a2", size=5, solidity=0.25),
            ),
            text=sub["Ticker"], textposition="inside", insidetextanchor="middle",
            textfont=dict(color=[label_color(v) for v in vals], size=10), constraintext="inside",
            meta={"t": t, "s": str(sub["Sector_ETF"].iloc[0])},
            customdata=sub[["Ticker", "Company", "Sector_ETF"] + hover_fmt[1]].values,
            hovertemplate=hover_fmt[0] + "<extra></extra>",
            width=bar_width_ms,
        )

    # invisible trace so hovering the empty day still shows 0
    fig.add_scatter(
        meta="__total__",
        x=x_all, y=totals.reindex(x_all, fill_value=0),
        mode="markers", marker=dict(opacity=0, size=1), showlegend=False,
        customdata=np.stack([unique.reindex(x_all, fill_value=0)], axis=-1),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Total hits: %{y}<br>Unique tickers: %{customdata[0]}<extra></extra>",
    )

    # color scale key (a hidden point that only exists to draw the colorbar)
    fig.add_scatter(
        meta="__scale__", x=[x_all[0]], y=[0], mode="markers", hoverinfo="skip", showlegend=False,
        marker=dict(size=0.1, opacity=0, color=[0], colorscale=PLOTLY_SCALE,
                    cmin=-COLOR_CAP_PCT, cmax=COLOR_CAP_PCT, showscale=True,
                    colorbar=dict(orientation="h", x=1, xanchor="right", y=1.03, yanchor="bottom",
                                  len=0.34, thickness=10, outlinewidth=0,
                                  tickvals=[-COLOR_CAP_PCT, -COLOR_CAP_PCT / 2, 0,
                                            COLOR_CAP_PCT / 2, COLOR_CAP_PCT],
                                  ticktext=[f"≤ -{COLOR_CAP_PCT}%", f"-{COLOR_CAP_PCT / 2:g}%", "0%",
                                            f"+{COLOR_CAP_PCT / 2:g}%", f"≥ +{COLOR_CAP_PCT}%"],
                                  title=dict(text=f"{COLOR_FWD_DAYS}D forward return",
                                             side="top", font=dict(size=11)),
                                  tickfont=dict(size=10))),
    )

    if annotate_totals:
        for x, v in totals.items():
            if v > 0:
                fig.add_annotation(x=x, y=v, text=str(int(v)), showarrow=False, yshift=9,
                                   font=dict(size=10, color=INK_2))

    fig.update_layout(
        barmode="stack", bargap=0.18,
        plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", color=INK_2, size=12),
        height=520, margin=dict(l=50, r=20, t=70, b=40), showlegend=False,
        hoverlabel=dict(bgcolor="#ffffff", font=dict(color=INK)),
        xaxis=dict(showgrid=False, linecolor=GRID, rangeslider=dict(visible=True, thickness=0.06)),
        yaxis=dict(gridcolor=GRID, zeroline=False, rangemode="tozero", title=None,
                   tickformat=",d"),
    )
    return fig


def build_report(hits: pd.DataFrame, trading_days: pd.DatetimeIndex, start, end):
    fwd_col = f"Fwd_{COLOR_FWD_DAYS}D%"

    # ---------------- daily ----------------
    daily = hits.copy()
    daily["Hit"] = 1
    daily["Color_Val"] = daily[fwd_col]
    daily["Fwd_Label"] = daily[fwd_col].map(lambda v: "not yet available" if pd.isna(v) else f"{v:+.2f}%")
    daily_totals = daily.groupby("Date").size()
    daily_unique = daily.groupby("Date")["Ticker"].nunique()

    daily_hover = (
        "<b>%{customdata[0]}</b> - %{customdata[1]}<br>%{x|%b %d, %Y} · %{customdata[2]}<br>"
        "Days since ATH: %{customdata[3]} · RVOL: %{customdata[4]}x · RSI: %{customdata[5]}<br>"
        f"<b>{COLOR_FWD_DAYS}D forward return: %{{customdata[6]}}</b>",
        ["Days_Since_ATH", "RVOL", "RSI", "Fwd_Label"],
    )
    fig_daily = stacked_bar(
        daily, "Date", "Hit", trading_days,
        "Daily hits - unique tickers passing the scanner each trading day",
        daily_hover, daily_totals, daily_unique,
        bar_width_ms=0.8 * 86_400_000,
    )
    # hide weekends + market holidays so each trading day gets full bar width
    all_bdays = pd.bdate_range(trading_days.min(), trading_days.max())
    holidays = [d.strftime("%Y-%m-%d") for d in all_bdays.difference(trading_days)]
    fig_daily.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(values=holidays)])

    # ---------------- weekly (sum of daily hits) ----------------
    hits_w = hits.copy()
    hits_w["Week"] = hits_w["Date"].dt.to_period("W-FRI").dt.start_time + pd.Timedelta(days=2)
    weekly = (hits_w.groupby(["Week", "Ticker", "Company", "Sector_ETF"])
              .agg(Days_Hit=("Date", "size"),
                   Color_Val=(fwd_col, "mean"),
                   Dates=("Date", lambda s: ", ".join(d.strftime("%a %m/%d") for d in s)))
              .reset_index())
    weekly["Week_Label"] = (weekly["Week"] - pd.Timedelta(days=2)).dt.strftime("%b %d, %Y")
    weekly["Fwd_Label"] = weekly["Color_Val"].map(lambda v: "not yet available" if pd.isna(v) else f"{v:+.2f}%")
    weekly_totals = weekly.groupby("Week")["Days_Hit"].sum()
    weekly_unique = weekly.groupby("Week")["Ticker"].nunique()
    all_weeks = pd.DatetimeIndex(sorted(set(pd.Series(trading_days).dt.to_period("W-FRI").dt.start_time + pd.Timedelta(days=2))))

    weekly_hover = (
        "<b>%{customdata[0]}</b> - %{customdata[1]}<br>Week of %{customdata[4]} · %{customdata[2]}<br>"
        "Days hit this week: %{y}<br>%{customdata[3]}<br>"
        f"<b>Avg {COLOR_FWD_DAYS}D forward return: %{{customdata[5]}}</b>",
        ["Dates", "Week_Label", "Fwd_Label"],
    )
    fig_weekly = stacked_bar(
        weekly, "Week", "Days_Hit", all_weeks,
        "Weekly hits - sum of daily hits per week (segment = days that ticker hit)",
        weekly_hover, weekly_totals, weekly_unique,
        bar_width_ms=0.8 * 5 * 86_400_000, annotate_totals=True,
    )
    fig_weekly.update_xaxes(tickformat="%b %d")

    # ---------------- summary ----------------
    n_days = len(trading_days)
    days_with_hit = daily_totals.index.nunique()
    top = (hits.groupby(["Ticker", "Company", "Sector_ETF"])
           .agg(Days=("Date", "size"), Fwd=(fwd_col, "mean"))
           .sort_values("Days", ascending=False).head(15).reset_index())

    ticker_info = {}
    for t, g in hits.groupby("Ticker"):
        ticker_info[t] = {
            "company": str(g["Company"].iloc[0]),
            "sector": str(g["Sector_ETF"].iloc[0]),
            "days": int(len(g)),
            "first": g["Date"].min().strftime("%b %d, %Y"),
            "last": g["Date"].max().strftime("%b %d, %Y"),
            "fwd": {str(n): (None if g[f"Fwd_{n}D%"].dropna().empty
                             else round(float(g[f"Fwd_{n}D%"].mean()), 2)) for n in FORWARD_DAYS},
        }
    sector_info = {}
    for etf, g in hits.groupby("Sector_ETF"):
        sector_info[etf] = {
            "name": str(g["Sector"].iloc[0]) if "Sector" in g else etf,
            "hits": int(len(g)),
            "tickers": int(g["Ticker"].nunique()),
            "fwd": {str(n): (None if g[f"Fwd_{n}D%"].dropna().empty
                             else round(float(g[f"Fwd_{n}D%"].mean()), 2)) for n in FORWARD_DAYS},
        }
    sector_options = "".join(
        f"<option value='{etf}'>{v['name']} ({etf}) - {v['hits']} hits</option>"
        for etf, v in sorted(sector_info.items(), key=lambda kv: kv[1]["name"]))
    # weekly rows, so the page can recompute the weekly totals shown above bars when filtered
    weekly_rows = [[w.strftime("%Y-%m-%d"), t, sec, int(n)] for w, t, sec, n in
                   weekly[["Week", "Ticker", "Sector_ETF", "Days_Hit"]].itertuples(index=False)]

    fwd_rows = ""
    for n in FORWARD_DAYS:
        s = hits[f"Fwd_{n}D%"].dropna()
        if len(s):
            fwd_rows += (f"<tr><td>{n} trading days</td><td>{len(s):,}</td>"
                         f"<td>{s.mean():+.2f}%</td><td>{s.median():+.2f}%</td>"
                         f"<td>{(s > 0).mean() * 100:.0f}%</td></tr>")

    top_rows = "".join(
        f"<tr><td><b>{r.Ticker}</b></td><td>{r.Company}</td><td>{r.Sector_ETF}</td>"
        f"<td class='num'>{r.Days}</td>"
        f"<td class='num'><span class='sw' style='background:{fwd_color(r.Fwd)}'></span>"
        f"{'n/a' if pd.isna(r.Fwd) else f'{r.Fwd:+.2f}%'}</td></tr>" for r in top.itertuples()
    )

    filters = [
        f"new closing high in last {LOOKBACK_DAYS} days",
        f"days since ATH ≤ {FILTER_MAX_DAYS_SINCE_ATH}" if FILTER_MAX_DAYS_SINCE_ATH is not None else None,
        f"RVOL ≥ {FILTER_MIN_RVOL}" if FILTER_MIN_RVOL is not None else None,
        f"today return ≥ {FILTER_MIN_TODAY_RETURN}%" if FILTER_MIN_TODAY_RETURN is not None else None,
        f"RSI {FILTER_MIN_RSI}-{FILTER_MAX_RSI}" if FILTER_MIN_RSI is not None else None,
        "MACD bullish" if FILTER_REQUIRE_MACD_BULL else None,
        "Stochastic bullish" if FILTER_REQUIRE_STOCH_BULL else None,
    ]
    filters = " · ".join(f for f in filters if f) if FILTERS_ACTIVE else "filters off"

    tiles = [
        (f"{len(hits):,}", "total hits"),
        (f"{hits['Ticker'].nunique():,}", "unique tickers"),
        (f"{len(hits) / max(n_days, 1):.1f}", "avg hits / trading day"),
        (f"{days_with_hit}/{n_days}", "days with ≥1 hit"),
        (f"{int(daily_totals.max()) if len(daily_totals) else 0}", "busiest day"),
    ]
    tiles_html = "".join(f"<div class='tile'><div class='v'>{v}</div><div class='l'>{l}</div></div>"
                         for v, l in tiles)

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ATH Scanner Backtest</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body {{ background:{SURFACE}; color:{INK}; font-family:Inter,'Segoe UI',Arial,sans-serif;
         margin:0; padding:24px 16px; }}
  .wrap {{ max-width:1280px; margin:0 auto; }}
  h1 {{ font-size:22px; margin:0 0 4px; }}
  .sub {{ color:{INK_2}; font-size:13px; margin-bottom:18px; line-height:1.5; }}
  .tiles {{ display:flex; flex-wrap:wrap; gap:12px; margin-bottom:18px; }}
  .tile {{ border:1px solid {GRID}; border-radius:8px; padding:10px 16px; min-width:130px; }}
  .tile .v {{ font-size:22px; font-weight:600; }}
  .tile .l {{ font-size:12px; color:{INK_2}; }}
  .card {{ border:1px solid {GRID}; border-radius:10px; padding:8px; margin-bottom:18px; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
  @media (max-width:800px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
  table {{ border-collapse:collapse; width:100%; font-size:13px; }}
  th, td {{ text-align:left; padding:6px 8px; border-bottom:1px solid {GRID}; }}
  th {{ color:{INK_2}; font-weight:500; }}
  td.num {{ text-align:right; }}
  h2 {{ font-size:15px; margin:6px 8px 10px; }}
  .note {{ color:{INK_2}; font-size:12px; margin:8px; line-height:1.5; }}
  .filter {{ display:flex; flex-wrap:wrap; align-items:center; gap:10px; margin-bottom:10px; }}
  .filter label {{ font-size:13px; color:{INK_2}; }}
  .filter input {{ font:inherit; font-size:14px; padding:7px 10px; border:1px solid #c9c8c2;
                   border-radius:6px; width:260px; background:#fff; color:{INK}; }}
  .filter button {{ font:inherit; font-size:13px; padding:7px 12px; border:1px solid #c9c8c2;
                    border-radius:6px; background:#fff; color:{INK}; cursor:pointer; }}
  .filter button:hover {{ background:#f0efec; }}
  .filter select {{ font:inherit; font-size:14px; padding:7px 10px; border:1px solid #c9c8c2;
                    border-radius:6px; background:#fff; color:{INK}; max-width:100%; }}
  .filter .sep {{ width:1px; height:26px; background:{GRID}; margin:0 4px; }}
  a {{ color:#2a78d6; }}
  h2.ct {{ font-size:16px; font-weight:600; margin:10px 10px 0; color:{INK}; }}
  .sw {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px;
         vertical-align:-1px; border:1px solid rgba(0,0,0,.08); }}
  #tickerSummary {{ font-size:13px; color:{INK_2}; margin-bottom:14px; min-height:18px; line-height:1.6; }}
  #tickerSummary b {{ color:{INK}; }}
</style></head><body><div class="wrap">
<h1>ATH Scanner - {BACKTEST_MONTHS}-month backtest</h1>
<div class="sub">{start:%b %d, %Y} → {end:%b %d, %Y} · S&amp;P 500 · {filters}<br>
Each bar segment is one ticker (labeled). Color = {COLOR_FWD_DAYS}-day forward return after the hit:
red = loss, gray = flat, green = gain, darker green = bigger gain (±{COLOR_CAP_PCT}% or more = darkest).
Hatched = hit too recent to have a {COLOR_FWD_DAYS}-day result yet. Drag to zoom (or use the slider)
to reveal labels on the daily chart.</div>
<div class="tiles">{tiles_html}</div>
<div class="filter">
  <label for="sectorSelect">Sector</label>
  <select id="sectorSelect"><option value="">All sectors</option>{sector_options}</select>
  <span class="sep"></span>
  <label for="tickerInput">Tickers</label>
  <input id="tickerInput" list="tickerList" placeholder="e.g. NVDA  or  NVDA, AVGO, JPM" autocomplete="off">
  <datalist id="tickerList"></datalist>
  <button id="applyBtn">Apply</button><button id="clearBtn">Reset</button>
</div>
<div id="tickerSummary"></div>
<div class="card"><h2 class="ct">Daily hits - unique tickers passing the scanner each trading day</h2>{fig_daily.to_html(full_html=False, include_plotlyjs=(True if PLOTLY_JS == 'inline' else 'cdn'), div_id='dailyChart', config={'displaylogo': False})}</div>
<div class="card"><h2 class="ct">Weekly hits - sum of daily hits per week (segment = days that ticker hit)</h2>{fig_weekly.to_html(full_html=False, include_plotlyjs=False, div_id='weeklyChart', config={'displaylogo': False})}</div>
<div class="grid2">
  <div class="card"><h2>Most frequent tickers</h2>
    <table><tr><th>Ticker</th><th>Company</th><th>Sector</th><th class="num">Days hit</th><th class="num">Avg {COLOR_FWD_DAYS}D fwd</th></tr>{top_rows}</table></div>
  <div class="card"><h2>Forward returns after a hit (close-to-close)</h2>
    <table><tr><th>Horizon</th><th>Hits</th><th>Mean</th><th>Median</th><th>% positive</th></tr>{fwd_rows}</table>
    <div class="note">Recent hits without enough future data are excluded from each horizon.
    Uses today's S&amp;P 500 members, so names that left the index in the past year are missing
    (survivorship bias) - treat returns as indicative, not a trading result.
    <br><a href="{CSV_OUT.name}" download>Download every hit (CSV)</a></div></div>
</div>
</div>
<script>
const INFO = {json.dumps(ticker_info)};
const SECTORS = {json.dumps(sector_info)};
const WEEKLY = {json.dumps(weekly_rows)};
const FWD = {json.dumps([str(n) for n in FORWARD_DAYS])};
const charts = ['dailyChart', 'weeklyChart'];
const input = document.getElementById('tickerInput');
const sectorSel = document.getElementById('sectorSelect');
const summary = document.getElementById('tickerSummary');
let weeklyAnn = null;

function parse(v) {{
  return v.toUpperCase().split(/[\s,;]+/).map(x => x.replace('.', '-')).filter(Boolean);
}}
function pct(v) {{ return v === null ? 'n/a' : (v > 0 ? '+' : '') + v.toFixed(2) + '%'; }}
function fwdText(f) {{ return FWD.map(n => `${{n}}D ${{pct(f[n])}}`).join(' · '); }}

// ticker suggestions follow the chosen sector
function fillTickerList() {{
  const sec = sectorSel.value;
  document.getElementById('tickerList').innerHTML = Object.keys(INFO).sort()
    .filter(t => !sec || INFO[t].sector === sec)
    .map(t => `<option value="${{t}}">${{INFO[t].company}}</option>`).join('');
}}

function applyFilter() {{
  const sec = sectorSel.value;
  const tickers = new Set(parse(input.value));
  const byTicker = tickers.size > 0;
  const filtering = byTicker || !!sec;
  const keep = m => (!sec || m.s === sec) && (!byTicker || tickers.has(m.t));

  charts.forEach(id => {{
    const g = document.getElementById(id);
    const vis = g.data.map(tr => tr.meta === '__scale__' ? true
                               : tr.meta === '__total__' ? !filtering : keep(tr.meta));
    Plotly.restyle(g, {{visible: vis}});

    const upd = {{'yaxis.autorange': true, 'yaxis.dtick': null}};
    if (id === 'weeklyChart') {{
      if (!filtering) upd.annotations = weeklyAnn;
      else {{
        const tot = {{}};
        WEEKLY.forEach(([w, t, s, n]) => {{ if (keep({{t: t, s: s}})) tot[w] = (tot[w] || 0) + n; }});
        upd.annotations = Object.entries(tot).map(([w, v]) => ({{
          x: w, y: v, text: String(v), showarrow: false, yshift: 9,
          font: {{size: 10, color: '{INK_2}'}} }}));
      }}
    }}
    Plotly.relayout(g, upd).then(() => {{
      // whole-number ticks when counts are small (avoids repeated 1, 1, 2, 2 labels)
      if (filtering && g._fullLayout.yaxis.range[1] <= 12) Plotly.relayout(g, {{'yaxis.dtick': 1}});
    }});
  }});

  const lines = [];
  if (sec && SECTORS[sec]) {{
    const s = SECTORS[sec];
    lines.push(`<b>${{s.name}} (${{sec}})</b> · <b>${{s.hits}}</b> hits · ${{s.tickers}} unique tickers · ` +
               `avg forward return: ${{fwdText(s.fwd)}}`);
  }}
  [...tickers].forEach(t => {{
    const i = INFO[t];
    if (!i) lines.push(`<b>${{t}}</b>: no hits in this backtest window (or not an S&amp;P 500 ticker).`);
    else if (sec && i.sector !== sec) lines.push(`<b>${{t}}</b> is in ${{i.sector}}, not ${{sec}} - clear the sector to see it.`);
    else lines.push(`<b>${{t}}</b> - ${{i.company}} (${{i.sector}}) · <b>${{i.days}}</b> days hit · ` +
                    `first ${{i.first}} · last ${{i.last}} · avg forward return: ${{fwdText(i.fwd)}}`);
  }});
  summary.innerHTML = lines.join('<br>');
}}

// when embedded in the dashboard (iframe), tell the parent page how tall this report is
function postHeight() {{
  if (window.parent !== window)
    window.parent.postMessage({{type: 'scanner-height', h: document.documentElement.scrollHeight}}, '*');
}}
window.addEventListener('load', postHeight);
if (window.ResizeObserver) new ResizeObserver(postHeight).observe(document.body);

window.addEventListener('load', () => {{
  charts.forEach(id => {{
    const g = document.getElementById(id);
  }});
  weeklyAnn = JSON.parse(JSON.stringify(document.getElementById('weeklyChart').layout.annotations || []));
  fillTickerList();
  sectorSel.addEventListener('change', () => {{ fillTickerList(); applyFilter(); }});
  document.getElementById('applyBtn').onclick = applyFilter;
  document.getElementById('clearBtn').onclick = () => {{
    input.value = ''; sectorSel.value = ''; fillTickerList(); applyFilter();
  }};
  input.addEventListener('keydown', e => {{ if (e.key === 'Enter') applyFilter(); }});
  input.addEventListener('change', applyFilter);
}});
</script>
</body></html>"""
    HTML_OUT.write_text(html, encoding="utf-8")

    # manifest the dashboard reads to list this scanner (only in site/Pages mode)
    if os.environ.get("BACKTEST_OUTPUT_DIR"):
        (OUT_DIR / "scanner.json").write_text(json.dumps({
            "title": SCANNER_TITLE,
            "order": SCANNER_ORDER,
            "page": HTML_OUT.name,
            "subtitle": f"{start:%b %d, %Y} to {end:%b %d, %Y} · {len(hits):,} hits · "
                        f"{hits['Ticker'].nunique()} unique tickers",
        }, indent=2), encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def main():
    sp500 = get_sp500()
    tickers = sp500["Ticker"].tolist()
    print(f"Loaded {len(sp500)} S&P 500 tickers. Downloading full history (period='max')...")

    data = yf.download(tickers, period=DATA_PERIOD, interval="1d", group_by="ticker",
                       auto_adjust=False, threads=True)

    end = data.index.max()
    start = end - pd.DateOffset(months=BACKTEST_MONTHS)
    print(f"Replaying scanner from {start:%Y-%m-%d} to {end:%Y-%m-%d}...")

    hits, trading_days = run_backtest(data, sp500, start)
    if hits.empty:
        print("No hits in the backtest window.")
        return

    hits.to_csv(CSV_OUT, index=False)
    build_report(hits, trading_days, start, end)
    print(f"{len(hits):,} hits across {hits['Ticker'].nunique()} tickers.")
    print(f"Report: {HTML_OUT}\nCSV:    {CSV_OUT}")


if __name__ == "__main__":
    main()
