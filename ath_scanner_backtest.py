"""
ATH Scanner - 12-month backtest visual
======================================

Replays the All-Time-High scanner on every trading day over the last N months and
builds an interactive HTML report. Every new-ATH candidate is stored; the scanner's
filters (RSI, RVOL, days since ATH, today's return, MACD, Stochastic) are applied
live in the page, starting from your live scanner's settings, so you can test
other values. Also filters by sector, ticker, and positive/negative forward return.

  1. Daily stacked bar  - number of tickers passing the filters each day
  2. Weekly stacked bar - sum of daily hits per week (Mon-Fri)
  3. Forward returns by sector + most frequent tickers (update with the filters)

Each bar is split into segments, one per ticker, labeled with the ticker symbol.
Segment color = forward return after the hit (default 10 trading days):
red = negative, gray = flat, green = positive, darker green = bigger gain.
Hits too recent to have a forward result yet are shown hatched.

Outputs (written next to this script):
  ath_backtest_report.html  - open in any browser (zoom in to reveal labels)
  ath_backtest_hits.csv     - every ATH candidate with indicators, forward returns,
                              and a Passes_Scanner_Filters flag

Requires: pip install yfinance pandas requests plotly lxml
"""

import json
import os
import warnings
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs, get_plotlyjs_version
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
    """The live scanner's filters (used for the page defaults and the CSV flag)."""
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
    """Every (day, ticker) with a new all-time closing high in the lookback window.

    The indicator filters are NOT applied here - the report applies them live so
    they can be changed in the page. 'Passes_Scanner_Filters' marks the rows that
    pass the live scanner's current settings.
    """
    cands, all_days = [], set()

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

        ath = hist[hist["Is_New_ATH"]]
        if ath.empty:
            continue
        ath = ath.copy()
        ath["Passes_Scanner_Filters"] = ath.index.isin(apply_filters(hist).index)
        ath.insert(0, "Ticker", t)
        ath.insert(1, "Company", row["Company"])
        ath.insert(2, "Sector_ETF", row["Sector_ETF"])
        ath.insert(3, "Sector", row["Sector"])
        cands.append(ath)

    trading_days = pd.DatetimeIndex(sorted(all_days))
    if not cands:
        return pd.DataFrame(), trading_days

    cands = pd.concat(cands)
    cands.index.name = "Date"
    cands = cands.reset_index().drop(columns=["Is_New_ATH"])
    cands["Days_Since_ATH"] = cands["Days_Since_ATH"].astype(int)
    return cands.sort_values(["Date", "Ticker"]).reset_index(drop=True), trading_days


# ============================================================
# REPORT (charts + tables are drawn in the browser from the data)
# ============================================================

def _col(series, nd=2):
    """Series -> JSON-friendly list (NaN -> null)."""
    return [None if pd.isna(v) else round(float(v), nd) for v in series]


def build_report(cands: pd.DataFrame, trading_days: pd.DatetimeIndex, start, end):
    dates = sorted(cands["Date"].dt.strftime("%Y-%m-%d").unique())
    date_ix = {d: i for i, d in enumerate(dates)}
    tickers = sorted(cands["Ticker"].unique())
    tick_ix = {t: i for i, t in enumerate(tickers)}
    first = cands.drop_duplicates("Ticker").set_index("Ticker")

    all_bdays = pd.bdate_range(trading_days.min(), trading_days.max())
    holidays = [d.strftime("%Y-%m-%d") for d in all_bdays.difference(trading_days)]

    cols = {
        "d": [date_ix[d] for d in cands["Date"].dt.strftime("%Y-%m-%d")],
        "t": [tick_ix[t] for t in cands["Ticker"]],
        "rsi": _col(cands["RSI"], 1),
        "rvol": _col(cands["RVOL"], 2),
        "dsa": cands["Days_Since_ATH"].astype(int).tolist(),
        "ret": _col(cands["Today_Return%"], 2),
        "macd": cands["MACD_Bull"].astype(int).tolist(),
        "stoch": cands["Stoch_Bull"].astype(int).tolist(),
    }
    for n in FORWARD_DAYS:
        cols[f"f{n}"] = _col(cands[f"Fwd_{n}D%"], 2)

    def tri(v):  # True -> "bull", False -> "bear", None -> "any"
        return "any" if v is None else ("bull" if v else "bear")

    on = FILTERS_ACTIVE
    defaults = {
        "rsiMin": FILTER_MIN_RSI if on else None,
        "rsiMax": FILTER_MAX_RSI if on else None,
        "rvolMin": FILTER_MIN_RVOL if on else None,
        "maxDays": FILTER_MAX_DAYS_SINCE_ATH if on else None,
        "retMin": FILTER_MIN_TODAY_RETURN if on else None,
        "macd": tri(FILTER_REQUIRE_MACD_BULL) if on else "any",
        "stoch": tri(FILTER_REQUIRE_STOCH_BULL) if on else "any",
    }

    data = {
        "dates": dates,
        "trading": [d.strftime("%Y-%m-%d") for d in trading_days],
        "holidays": holidays,
        "tickers": tickers,
        "company": {t: str(first.loc[t, "Company"]) for t in tickers},
        "sector": {t: str(first.loc[t, "Sector_ETF"]) for t in tickers},
        "sectorName": {str(r.Sector_ETF): str(r.Sector)
                       for r in cands.drop_duplicates("Sector_ETF").itertuples()},
        "cols": cols,
        "fwdDays": FORWARD_DAYS,
        "colorDays": COLOR_FWD_DAYS,
        "cap": COLOR_CAP_PCT,
        "gradient": GRADIENT,
        "pending": PENDING_COLOR,
        "defaults": defaults,
        "lookback": LOOKBACK_DAYS,
    }

    default_desc = [
        f"new closing high in last {LOOKBACK_DAYS} days",
        f"days since ATH ≤ {defaults['maxDays']}" if defaults["maxDays"] is not None else None,
        f"RVOL ≥ {defaults['rvolMin']}" if defaults["rvolMin"] is not None else None,
        f"today return ≥ {defaults['retMin']}%" if defaults["retMin"] is not None else None,
        f"RSI {defaults['rsiMin']}-{defaults['rsiMax']}" if defaults["rsiMin"] is not None else None,
        {"bull": "MACD bullish", "bear": "MACD bearish"}.get(defaults["macd"]),
        {"bull": "Stochastic bullish", "bear": "Stochastic bearish"}.get(defaults["stoch"]),
    ]
    default_desc = " · ".join(x for x in default_desc if x)

    if PLOTLY_JS == "inline":
        plotly_tag = f"<script>{get_plotlyjs()}</script>"
    else:
        plotly_tag = f'<script src="https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"></script>'

    day_opts = "".join(f"<option value='{i}'>{i}</option>" for i in range(LOOKBACK_DAYS))

    html = (PAGE_TEMPLATE
            .replace("__PERIOD__", f"{start:%b %d, %Y} → {end:%b %d, %Y}")
            .replace("__DEFAULTS_DESC__", default_desc)
            .replace("__DAY_OPTS__", day_opts)
            .replace("__CSV__", CSV_OUT.name)
            .replace("__N__", str(COLOR_FWD_DAYS))
            .replace("__CAP__", f"{COLOR_CAP_PCT:g}")
            .replace("__MONTHS__", str(BACKTEST_MONTHS))
            .replace("__SURFACE__", SURFACE).replace("__INK2__", INK_2)
            .replace("__INK__", INK).replace("__GRID__", GRID))
    html = html.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    html = html.replace("__PLOTLY__", plotly_tag)
    HTML_OUT.write_text(html, encoding="utf-8")

    # manifest the dashboard reads to list this scanner (only in site/Pages mode)
    if os.environ.get("BACKTEST_OUTPUT_DIR"):
        hits = cands[cands["Passes_Scanner_Filters"]]
        (OUT_DIR / "scanner.json").write_text(json.dumps({
            "title": SCANNER_TITLE,
            "order": SCANNER_ORDER,
            "page": HTML_OUT.name,
            "subtitle": f"{start:%b %d, %Y} to {end:%b %d, %Y} · {len(hits):,} hits with scanner "
                        f"settings · {hits['Ticker'].nunique()} unique tickers",
        }, indent=2), encoding="utf-8")


PAGE_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>ATH Scanner Backtest</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
__PLOTLY__
<style>
  body { background:__SURFACE__; color:__INK__; font-family:Inter,'Segoe UI',Arial,sans-serif;
         margin:0; padding:24px 16px; }
  .wrap { max-width:1280px; margin:0 auto; }
  h1 { font-size:22px; margin:0 0 4px; }
  .sub { color:__INK2__; font-size:13px; margin-bottom:18px; line-height:1.55; }
  .tiles { display:flex; flex-wrap:wrap; gap:12px; margin-bottom:18px; }
  .tile { border:1px solid __GRID__; border-radius:8px; padding:10px 16px; min-width:130px; }
  .tile .v { font-size:22px; font-weight:600; }
  .tile .l { font-size:12px; color:__INK2__; }
  .card { border:1px solid __GRID__; border-radius:10px; padding:8px; margin-bottom:18px; }
  h2.ct { font-size:16px; font-weight:600; margin:10px 10px 0; }
  h2 { font-size:15px; margin:6px 8px 10px; }
  .panel { border:1px solid __GRID__; border-radius:10px; padding:12px 14px; margin-bottom:12px; background:#fff; }
  .row { display:flex; flex-wrap:wrap; align-items:center; gap:10px 16px; }
  .row + .row { margin-top:10px; padding-top:10px; border-top:1px solid __GRID__; }
  .ctl { display:flex; align-items:center; gap:6px; font-size:13px; color:__INK2__; }
  .ctl input, .ctl select { font:inherit; font-size:14px; padding:6px 8px; border:1px solid #c9c8c2;
         border-radius:6px; background:#fff; color:__INK__; }
  .ctl input.n { width:64px; }
  #tickerInput { width:230px; }
  button { font:inherit; font-size:13px; padding:7px 12px; border:1px solid #c9c8c2;
           border-radius:6px; background:#fff; color:__INK__; cursor:pointer; }
  button:hover { background:#f0efec; }
  .hint { font-size:12px; color:__INK2__; }
  #summary { font-size:13px; color:__INK2__; margin:0 0 14px; line-height:1.6; min-height:4px; }
  #summary b { color:__INK__; }
  .warn { background:#fff6e0; border:1px solid #f0d58a; border-radius:6px; padding:6px 10px;
          color:#6b4e00; margin-bottom:6px; }
  table { border-collapse:collapse; width:100%; font-size:13px; }
  th, td { text-align:left; padding:6px 8px; border-bottom:1px solid __GRID__; white-space:nowrap; }
  th { color:__INK2__; font-weight:500; }
  .num { text-align:right; }
  tr.click { cursor:pointer; }
  tr.click:hover td { background:#f5f4f0; }
  tr.all td { font-weight:600; background:#f7f6f2; }
  .tscroll { overflow-x:auto; }
  .sw { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px;
        vertical-align:-1px; border:1px solid rgba(0,0,0,.08); }
  .note { color:__INK2__; font-size:12px; margin:8px; line-height:1.5; }
  a { color:#2a78d6; }
</style></head><body><div class="wrap">
<h1>ATH Scanner - __MONTHS__-month backtest</h1>
<div class="sub">__PERIOD__ · S&amp;P 500 · scanner settings: __DEFAULTS_DESC__<br>
Each bar segment is one ticker (labeled). Color = __N__-day forward return after the hit:
red = loss, gray = flat, green = gain, darker green = bigger gain (±__CAP__% or more = darkest).
Hatched = hit too recent to have a __N__-day result yet. Drag to zoom (or use the slider) to reveal labels.</div>

<div class="panel">
  <div class="row">
    <label class="ctl">Sector <select id="sector"><option value="">All sectors</option></select></label>
    <label class="ctl">Tickers <input id="tickerInput" list="tickerList" autocomplete="off"
           placeholder="e.g. NVDA  or  NVDA, AVGO, JPM"></label>
    <datalist id="tickerList"></datalist>
    <button id="applyBtn">Apply</button>
    <button id="resetBtn" title="Back to your live scanner's settings">Reset to scanner settings</button>
  </div>
  <div class="row">
    <label class="ctl">RSI <input class="n" id="rsiMin" type="number" step="1" placeholder="min">
           to <input class="n" id="rsiMax" type="number" step="1" placeholder="max"></label>
    <label class="ctl">Min RVOL <input class="n" id="rvolMin" type="number" step="0.1" placeholder="any"></label>
    <label class="ctl">Max days since ATH <select id="maxDays"><option value="">any</option>__DAY_OPTS__</select></label>
    <label class="ctl">Min today return % <input class="n" id="retMin" type="number" step="0.5" placeholder="any"></label>
    <label class="ctl">MACD <select id="macd"><option value="any">any</option><option value="bull">bullish</option><option value="bear">bearish</option></select></label>
    <label class="ctl">Stochastic <select id="stoch"><option value="any">any</option><option value="bull">bullish</option><option value="bear">bearish</option></select></label>
    <label class="ctl">__N__D forward return <select id="fwdSign">
      <option value="all">all</option><option value="pos">positive only</option><option value="neg">negative only</option></select></label>
  </div>
</div>
<div id="summary"></div>
<div class="tiles" id="tiles"></div>

<div class="card"><h2 class="ct">Daily hits - unique tickers passing the filters each trading day</h2><div id="dailyChart"></div></div>
<div class="card"><h2 class="ct">Weekly hits - sum of daily hits per week (segment = days that ticker hit)</h2><div id="weeklyChart"></div></div>

<div class="card"><h2>Forward returns after a hit (close-to-close), by sector</h2>
  <div class="tscroll"><table id="sectorTable"></table></div>
  <div class="note">Click a sector row to filter to it. Hits without enough future data yet are left out
  of each horizon. Uses today's S&amp;P 500 members, so names that left the index in the past year are
  missing (survivorship bias) - treat returns as indicative, not a trading result.
  <br><a href="__CSV__" download>Download every all-time-high candidate (CSV)</a>
  (column Passes_Scanner_Filters = passes your live scanner's settings)</div></div>

<div class="card"><h2>Most frequent tickers</h2>
  <div class="tscroll"><table id="topTable"></table></div>
  <div class="note">Click a ticker to filter to it.</div></div>
</div>

<script>
const D = __DATA__;
const C = D.cols, N = C.d.length, CK = 'f' + D.colorDays;
const SURF = '__SURFACE__', INK = '__INK__', INK2 = '__INK2__', GRIDC = '__GRID__';
const DAY = 86400000;
const $ = id => document.getElementById(id);

// ---------- helpers ----------
const hexRgb = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
function color(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return D.pending;
  const x = Math.max(-1, Math.min(1, v / D.cap)), g = D.gradient;
  for (let k = 0; k < g.length - 1; k++) {
    const [p0, c0] = g[k], [p1, c1] = g[k + 1];
    if (x <= p1) {
      const f = (x - p0) / (p1 - p0), a = hexRgb(c0), b = hexRgb(c1);
      return '#' + a.map((v0, j) => Math.round(v0 + (b[j] - v0) * f).toString(16).padStart(2, '0')).join('');
    }
  }
  return g[g.length - 1][1];
}
const labelColor = v => (v === null || Math.abs(v) / D.cap < 0.4) ? INK : '#ffffff';
const pct = (v, d = 2) => v === null || v === undefined ? 'n/a' : (v > 0 ? '+' : '') + v.toFixed(d) + '%';
const fmt = v => v === null ? 'n/a' : v;
const ge = (v, x) => v !== null && v >= x;
const le = (v, x) => v !== null && v <= x;
const mean = a => a.length ? a.reduce((s, v) => s + v, 0) / a.length : null;
function median(a) {
  if (!a.length) return null;
  const s = [...a].sort((x, y) => x - y), m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}
const tick = i => D.tickers[C.t[i]];
const byFwdDesc = (a, b) => {
  const va = C[CK][a], vb = C[CK][b];
  if (va === null && vb === null) return 0;
  if (va === null) return 1;
  if (vb === null) return -1;
  return vb - va;
};
function parse(v) {
  return v.toUpperCase().split(/[\s,;]+/).map(x => x.replace('.', '-')).filter(Boolean);
}
function num(id) { const v = $(id).value.trim(); return v === '' ? null : Number(v); }
function weekMid(ds) {           // Wednesday of the Mon-Fri week containing ds (bar position)
  const d = new Date(ds + 'T00:00:00Z'), dow = d.getUTCDay();
  return new Date(d.getTime() + (3 - dow) * DAY).toISOString().slice(0, 10);
}
function weekLabel(mid) {
  const d = new Date(new Date(mid + 'T00:00:00Z').getTime() - 2 * DAY);
  return d.toLocaleDateString('en-US', {month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC'});
}

// ---------- filters ----------
function readFilters() {
  return {
    sector: $('sector').value, tickers: new Set(parse($('tickerInput').value)),
    rsiMin: num('rsiMin'), rsiMax: num('rsiMax'), rvolMin: num('rvolMin'),
    maxDays: num('maxDays'), retMin: num('retMin'),
    macd: $('macd').value, stoch: $('stoch').value, fwdSign: $('fwdSign').value,
  };
}
function passes(i, f) {
  const t = tick(i);
  if (f.sector && D.sector[t] !== f.sector) return false;
  if (f.tickers.size && !f.tickers.has(t)) return false;
  if (f.maxDays !== null && !le(C.dsa[i], f.maxDays)) return false;
  if (f.rvolMin !== null && !ge(C.rvol[i], f.rvolMin)) return false;
  if (f.retMin !== null && !ge(C.ret[i], f.retMin)) return false;
  if (f.rsiMin !== null && !ge(C.rsi[i], f.rsiMin)) return false;
  if (f.rsiMax !== null && !le(C.rsi[i], f.rsiMax)) return false;
  if (f.macd !== 'any' && C.macd[i] !== (f.macd === 'bull' ? 1 : 0)) return false;
  if (f.stoch !== 'any' && C.stoch[i] !== (f.stoch === 'bull' ? 1 : 0)) return false;
  const fv = C[CK][i];
  if (f.fwdSign === 'pos' && !(fv !== null && fv > 0)) return false;
  if (f.fwdSign === 'neg' && !(fv !== null && fv < 0)) return false;
  return true;
}
function setDefaults() {
  const d = D.defaults, s = (id, v) => { $(id).value = v === null ? '' : v; };
  s('rsiMin', d.rsiMin); s('rsiMax', d.rsiMax); s('rvolMin', d.rvolMin);
  s('maxDays', d.maxDays); s('retMin', d.retMin); s('macd', d.macd); s('stoch', d.stoch);
  $('fwdSign').value = 'all'; $('sector').value = ''; $('tickerInput').value = '';
}

// ---------- charts ----------
const scaleTrace = x0 => ({
  type: 'scatter', x: [x0], y: [0], mode: 'markers', hoverinfo: 'skip', showlegend: false,
  marker: {size: 0.1, opacity: 0, color: [0], cmin: -D.cap, cmax: D.cap, showscale: true,
    colorscale: D.gradient.map(([p, c]) => [(p + 1) / 2, c]),
    colorbar: {orientation: 'h', x: 1, xanchor: 'right', y: 1.03, yanchor: 'bottom', len: 0.34,
      thickness: 10, outlinewidth: 0, tickfont: {size: 10},
      tickvals: [-D.cap, -D.cap / 2, 0, D.cap / 2, D.cap],
      ticktext: [`≤ -${D.cap}%`, `-${D.cap / 2}%`, '0%', `+${D.cap / 2}%`, `≥ +${D.cap}%`],
      title: {text: `${D.colorDays}D forward return`, side: 'top', font: {size: 11}}}},
});

function stackTraces(groups, widthMs, hover) {
  // groups: Map x -> [{t, y, v, cd}] sorted best return first (best at the bottom)
  let maxK = 0; groups.forEach(a => { maxK = Math.max(maxK, a.length); });
  const traces = [];
  for (let k = 0; k < maxK; k++) {
    const tr = {type: 'bar', x: [], y: [], text: [], customdata: [], showlegend: false,
      marker: {color: [], line: {color: SURF, width: 0.5},
               pattern: {shape: [], fgcolor: '#a9a8a2', size: 5, solidity: 0.25}},
      textposition: 'inside', insidetextanchor: 'middle', constraintext: 'inside',
      textfont: {size: 10, color: []}, width: widthMs, hovertemplate: hover};
    groups.forEach((a, x) => {
      if (a.length <= k) return;
      const s = a[k];
      tr.x.push(x); tr.y.push(s.y); tr.text.push(s.t); tr.customdata.push(s.cd);
      tr.marker.color.push(color(s.v)); tr.marker.pattern.shape.push(s.v === null ? '/' : '');
      tr.textfont.color.push(labelColor(s.v));
    });
    traces.push(tr);
  }
  return traces;
}

function baseLayout(maxY) {
  return {
    barmode: 'stack', bargap: 0.18, plot_bgcolor: SURF, paper_bgcolor: SURF, height: 520,
    margin: {l: 50, r: 20, t: 70, b: 40}, showlegend: false, uirevision: 'keep',
    font: {family: 'Inter, Segoe UI, Arial, sans-serif', color: INK2, size: 12},
    hoverlabel: {bgcolor: '#ffffff', font: {color: INK}},
    xaxis: {showgrid: false, linecolor: GRIDC, rangeslider: {visible: true, thickness: 0.06}},
    yaxis: {gridcolor: GRIDC, zeroline: false, rangemode: 'tozero', tickformat: ',d',
            dtick: maxY <= 12 ? 1 : null},
  };
}

function drawDaily(idx) {
  const groups = new Map();
  [...idx].sort(byFwdDesc).forEach(i => {
    const x = D.dates[C.d[i]], t = tick(i), v = C[CK][i];
    if (!groups.has(x)) groups.set(x, []);
    groups.get(x).push({t, y: 1, v, cd: [t, D.company[t], D.sector[t], C.dsa[i], fmt(C.rvol[i]),
      fmt(C.rsi[i]), v === null ? 'not yet available' : pct(v), pct(C.ret[i])]});
  });
  const hover = '<b>%{customdata[0]}</b> - %{customdata[1]}<br>%{x|%b %d, %Y} · %{customdata[2]}<br>' +
    'Days since ATH: %{customdata[3]} · RVOL: %{customdata[4]}x · RSI: %{customdata[5]}<br>' +
    `Day's return: %{customdata[7]}<br><b>${D.colorDays}D forward return: %{customdata[6]}</b><extra></extra>`;
  const traces = stackTraces(groups, 0.8 * DAY, hover);
  const tot = D.trading.map(x => (groups.get(x) || []).length);
  traces.push({type: 'scatter', x: D.trading, y: tot, mode: 'markers', showlegend: false,
    marker: {opacity: 0, size: 1},
    hovertemplate: '<b>%{x|%b %d, %Y}</b><br>Hits: %{y}<extra></extra>'});
  traces.push(scaleTrace(D.trading[0]));
  const lay = baseLayout(Math.max(0, ...tot));
  lay.xaxis.rangebreaks = [{bounds: ['sat', 'mon']}, {values: D.holidays}];
  Plotly.react('dailyChart', traces, lay, {displaylogo: false, responsive: true});
}

function drawWeekly(idx) {
  const wk = new Map();   // mid -> Map ticker -> {days, vals, dates}
  idx.forEach(i => {
    const ds = D.dates[C.d[i]], mid = weekMid(ds), t = tick(i);
    if (!wk.has(mid)) wk.set(mid, new Map());
    const m = wk.get(mid);
    if (!m.has(t)) m.set(t, {days: 0, vals: [], dates: []});
    const e = m.get(t); e.days++; e.dates.push(ds);
    if (C[CK][i] !== null) e.vals.push(C[CK][i]);
  });
  const groups = new Map(), totals = {}, uniq = {};
  [...wk.keys()].sort().forEach(mid => {
    const segs = [];
    wk.get(mid).forEach((e, t) => {
      const v = e.vals.length ? mean(e.vals) : null;
      const dl = e.dates.sort().map(d => new Date(d + 'T00:00:00Z')
        .toLocaleDateString('en-US', {weekday: 'short', month: '2-digit', day: '2-digit', timeZone: 'UTC'})).join(', ');
      segs.push({t, y: e.days, v, cd: [t, D.company[t], D.sector[t], dl, weekLabel(mid),
        v === null ? 'not yet available' : pct(v)]});
    });
    segs.sort((a, b) => (a.v === null) - (b.v === null) || (b.v || 0) - (a.v || 0));
    groups.set(mid, segs);
    totals[mid] = segs.reduce((s, x) => s + x.y, 0); uniq[mid] = segs.length;
  });
  const hover = '<b>%{customdata[0]}</b> - %{customdata[1]}<br>Week of %{customdata[4]} · %{customdata[2]}<br>' +
    'Days hit this week: %{y}<br>%{customdata[3]}<br>' +
    `<b>Avg ${D.colorDays}D forward return: %{customdata[5]}</b><extra></extra>`;
  const traces = stackTraces(groups, 0.8 * 5 * DAY, hover);
  const allWeeks = [...new Set(D.trading.map(weekMid))].sort();
  traces.push({type: 'scatter', x: allWeeks, y: allWeeks.map(w => totals[w] || 0), mode: 'markers',
    showlegend: false, marker: {opacity: 0, size: 1}, customdata: allWeeks.map(w => [uniq[w] || 0, weekLabel(w)]),
    hovertemplate: 'Week of %{customdata[1]}<br>Total hits: %{y}<br>Unique tickers: %{customdata[0]}<extra></extra>'});
  traces.push(scaleTrace(allWeeks[0]));
  const lay = baseLayout(Math.max(0, ...Object.values(totals)));
  lay.xaxis.tickformat = '%b %d';
  lay.annotations = Object.entries(totals).map(([x, v]) => ({x, y: v, text: String(v), showarrow: false,
    yshift: 9, font: {size: 10, color: INK2}}));
  Plotly.react('weeklyChart', traces, lay, {displaylogo: false, responsive: true});
}

// ---------- tiles + tables ----------
function tiles(idx) {
  const perDay = {}; idx.forEach(i => { perDay[C.d[i]] = (perDay[C.d[i]] || 0) + 1; });
  const counts = Object.values(perDay), fv = idx.map(i => C[CK][i]).filter(v => v !== null);
  const win = fv.length ? fv.filter(v => v > 0).length / fv.length * 100 : null;
  const t = [
    [idx.length.toLocaleString(), 'total hits'],
    [new Set(idx.map(i => C.t[i])).size, 'unique tickers'],
    [(idx.length / Math.max(D.trading.length, 1)).toFixed(1), 'avg hits / trading day'],
    [`${counts.length}/${D.trading.length}`, 'days with ≥1 hit'],
    [counts.length ? Math.max(...counts) : 0, 'busiest day'],
    [win === null ? 'n/a' : win.toFixed(0) + '%', `${D.colorDays}D win rate`],
    [pct(mean(fv)), `avg ${D.colorDays}D forward return`],
  ];
  $('tiles').innerHTML = t.map(([v, l]) => `<div class="tile"><div class="v">${v}</div><div class="l">${l}</div></div>`).join('');
}

function statRow(label, rows, cls, attr) {
  const fw = n => rows.map(i => C['f' + n][i]).filter(v => v !== null);
  const fc = fw(D.colorDays), avgC = mean(fc);
  const cells = D.fwdDays.map(n => {
    const m = mean(fw(n));
    const sw = n === D.colorDays ? `<span class="sw" style="background:${color(m)}"></span>` : '';
    return `<td class="num">${sw}${pct(m)}</td>`;
  }).join('');
  const win = fc.length ? (fc.filter(v => v > 0).length / fc.length * 100).toFixed(0) + '%' : 'n/a';
  return {avg: avgC, html: `<tr class="${cls}" ${attr}><td>${label}</td><td class="num">${rows.length}</td>` +
    `<td class="num">${new Set(rows.map(i => C.t[i])).size}</td>${cells}` +
    `<td class="num">${pct(median(fc))}</td><td class="num">${win}</td></tr>`};
}

function sectorTable(idx) {
  const by = new Map();
  idx.forEach(i => { const s = D.sector[tick(i)]; if (!by.has(s)) by.set(s, []); by.get(s).push(i); });
  const head = `<tr><th>Sector</th><th class="num">Hits</th><th class="num">Tickers</th>` +
    D.fwdDays.map(n => `<th class="num">Avg ${n}D</th>`).join('') +
    `<th class="num">Median ${D.colorDays}D</th><th class="num">${D.colorDays}D % positive</th></tr>`;
  const rows = [...by.entries()].map(([s, r]) => statRow(
      `<b>${s}</b> <span class="hint">${D.sectorName[s] || ''}</span>`, r, 'click', `data-sector="${s}"`))
    .sort((a, b) => (a.avg === null) - (b.avg === null) || (b.avg || 0) - (a.avg || 0));
  const all = statRow('All sectors', idx, 'all click', 'data-sector=""');
  $('sectorTable').innerHTML = head + (idx.length ? all.html + rows.map(r => r.html).join('')
    : '<tr><td colspan="9" class="hint">No hits match these filters.</td></tr>');
  $('sectorTable').querySelectorAll('tr.click').forEach(tr => tr.onclick = () => {
    $('sector').value = tr.dataset.sector; fillTickerList(); render();
  });
}

function topTable(idx) {
  const by = new Map();
  idx.forEach(i => { const t = tick(i); if (!by.has(t)) by.set(t, []); by.get(t).push(i); });
  const top = [...by.entries()].sort((a, b) => b[1].length - a[1].length).slice(0, 15);
  $('topTable').innerHTML = `<tr><th>Ticker</th><th>Company</th><th>Sector</th><th class="num">Days hit</th>` +
    `<th class="num">Avg ${D.colorDays}D fwd</th></tr>` + top.map(([t, r]) => {
      const m = mean(r.map(i => C[CK][i]).filter(v => v !== null));
      return `<tr class="click" data-t="${t}"><td><b>${t}</b></td><td>${D.company[t]}</td><td>${D.sector[t]}</td>` +
        `<td class="num">${r.length}</td><td class="num"><span class="sw" style="background:${color(m)}"></span>${pct(m)}</td></tr>`;
    }).join('');
  $('topTable').querySelectorAll('tr.click').forEach(tr => tr.onclick = () => {
    $('tickerInput').value = tr.dataset.t; render(); window.scrollTo({top: 0, behavior: 'smooth'});
  });
}

function summary(idx, f) {
  const lines = [];
  if (f.fwdSign !== 'all')
    lines.push(`<div class="warn">Showing only hits whose ${D.colorDays}-day forward return was ` +
      `<b>${f.fwdSign === 'pos' ? 'positive' : 'negative'}</b>. This is a hindsight view for studying ` +
      `winners or losers - the stats below are no longer a fair backtest of the scanner.</div>`);
  f.tickers.forEach(t => {
    if (!D.company[t]) { lines.push(`<b>${t}</b>: never made a new all-time high in this window (or not an S&amp;P 500 ticker).`); return; }
    if (f.sector && D.sector[t] !== f.sector) { lines.push(`<b>${t}</b> is in ${D.sector[t]}, not ${f.sector} - clear the sector to see it.`); return; }
    const r = idx.filter(i => tick(i) === t);
    if (!r.length) { lines.push(`<b>${t}</b> - ${D.company[t]} (${D.sector[t]}): no hits with the current filters.`); return; }
    const ds = r.map(i => D.dates[C.d[i]]).sort();
    const f2 = D.fwdDays.map(n => `${n}D ${pct(mean(r.map(i => C['f' + n][i]).filter(v => v !== null)))}`).join(' · ');
    lines.push(`<b>${t}</b> - ${D.company[t]} (${D.sector[t]}) · <b>${r.length}</b> days hit · first ${ds[0]} · ` +
      `last ${ds[ds.length - 1]} · avg forward return: ${f2}`);
  });
  $('summary').innerHTML = lines.join('<br>');
}

function render() {
  const f = readFilters(), idx = [];
  for (let i = 0; i < N; i++) if (passes(i, f)) idx.push(i);
  drawDaily(idx); drawWeekly(idx); tiles(idx); sectorTable(idx); topTable(idx); summary(idx, f);
}

function fillTickerList() {
  const sec = $('sector').value;
  $('tickerList').innerHTML = D.tickers.filter(t => !sec || D.sector[t] === sec)
    .map(t => `<option value="${t}">${D.company[t]}</option>`).join('');
}

// when embedded in the dashboard (iframe), tell the parent page how tall this report is
function postHeight() {
  if (window.parent !== window)
    window.parent.postMessage({type: 'scanner-height', h: document.documentElement.scrollHeight}, '*');
}

(function init() {
  $('sector').innerHTML += Object.keys(D.sectorName).sort((a, b) => D.sectorName[a].localeCompare(D.sectorName[b]))
    .map(s => `<option value="${s}">${D.sectorName[s]} (${s})</option>`).join('');
  setDefaults(); fillTickerList(); render();
  ['rsiMin', 'rsiMax', 'rvolMin', 'retMin'].forEach(id => $(id).addEventListener('change', render));
  ['maxDays', 'macd', 'stoch', 'fwdSign'].forEach(id => $(id).addEventListener('change', render));
  $('sector').addEventListener('change', () => { fillTickerList(); render(); });
  $('applyBtn').onclick = render;
  $('resetBtn').onclick = () => { setDefaults(); fillTickerList(); render(); };
  $('tickerInput').addEventListener('keydown', e => { if (e.key === 'Enter') render(); });
  $('tickerInput').addEventListener('change', render);
  window.addEventListener('load', postHeight);
  if (window.ResizeObserver) new ResizeObserver(postHeight).observe(document.body);
})();
</script>
</body></html>
"""


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

    cands, trading_days = run_backtest(data, sp500, start)
    if cands.empty:
        print("No all-time-high candidates in the backtest window.")
        return

    cands.to_csv(CSV_OUT, index=False)
    build_report(cands, trading_days, start, end)
    n_pass = int(cands["Passes_Scanner_Filters"].sum())
    print(f"{len(cands):,} all-time-high candidates; {n_pass:,} pass the scanner settings.")
    print(f"Report: {HTML_OUT}\nCSV:    {CSV_OUT}")


if __name__ == "__main__":
    main()
