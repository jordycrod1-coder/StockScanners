"""
Multi-Timeframe Buy/Sell Signal Scanner + Backtest
==================================================

Uses the same indicators as Ticker_Indicator_Graph_DWM.ipynb (MACD 12/26/9, RSI 14,
Stochastic 14/3/3, MFI 14, CMF 20) on the Daily, Weekly and Monthly timeframes, and
answers two questions for one ticker (default TSLA):

  BUY  - which indicator values were followed by the HIGHEST 10-day forward return?
  SELL - which indicator values were followed by the LOWEST 10-day forward return
         (the stock tended to fall after them, i.e. a good time to take profits)?

How the backtest is built
-------------------------
Every trading day in the window is replayed as if the scanner ran after that day's
close. Weekly and Monthly values are read the way the notebook shows them live:
the current week/month is a partial bar (week-to-date / month-to-date), so no
future data leaks in. Forward return = close N trading days later vs. that close.

Two ways to find good values:
  1. Interactive report (HTML): every indicator on every timeframe is a filter for
     a BUY rule and a separate SELL rule. A heatmap shows the average forward
     return for each value range of each indicator (green = price rose after,
     red = price fell after) - click a cell to use that range in the rule.
  2. Suggested rules: the script searches thresholds on the older part of the
     history (training period) and reports how the picked rule did on the most
     recent part it never saw (test period). Trust the test numbers, not the
     training numbers.

Alerts
------
  python mtf_signal_scanner.py            -> builds the report + CSV (no email)
  python mtf_signal_scanner.py --alert    -> checks today's bar and emails a BUY
                                             and/or SELL alert when a rule fires
Email uses EMAIL_USER / EMAIL_PASS / ALERT_TO (same secrets as the ATH scanner).
BUY_RULE / SELL_RULE below decide what triggers an alert ("auto" = the suggested
rule). Once you have picked rules in the report, paste them in as dicts so they
stop changing from day to day.

Outputs (written next to this script unless BACKTEST_OUTPUT_DIR is set):
  mtf_signal_report.html - open in any browser
  mtf_signal_days.csv    - every day with all indicators, forward returns, and
                           Buy_Signal / Sell_Signal flags for the scanner's rules

Requires: pip install yfinance pandas numpy plotly
Not financial advice: past indicator behavior does not guarantee future returns.
"""

import argparse
import json
import os
import smtplib
import warnings
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
from plotly.offline import get_plotlyjs, get_plotlyjs_version

warnings.filterwarnings("ignore")

# ============================================================
# SCANNER SETTINGS
# ============================================================

TICKER = os.environ.get("SIGNAL_TICKER", "TSLA")
DATA_PERIOD = "max"

# Rules that trigger email alerts and are the report's starting filters.
#   "auto" = use the rule the optimizer suggests (see the report's "Suggested rules")
#   or a dict. Keys are <timeframe>_<indicator>; timeframe D = Daily, W = Weekly,
#   M = Monthly. Numbers take (min, max) with None = no limit; yes/no fields take
#   True / False. Every key listed must pass (AND). Example using the notebook's
#   custom overbought lines as a sell rule:
#
#   SELL_RULE = {
#       "D_RSI": (81, None),          # Daily RSI >= 81
#       "W_StochK": (95, None),       # Weekly %K >= 95
#       "D_Hist_Rising": False,       # Daily MACD histogram bar turned red
#   }
#   BUY_RULE = {
#       "D_RSI": (None, 35),
#       "W_MACD_Bull": True,
#   }
#
# Field names: RSI, StochK, StochD, MFI, CMF, MACD, Signal, Hist, MACD_Pct, Hist_Pct,
#              MACD_Bull, MACD_Pos, Hist_Rising, Stoch_Bull, Flow_Bull, Candle_Up
BUY_RULE = "auto"
SELL_RULE = "auto"

# "every" = email every day the rule is true; "new" = only the first day of a streak
ALERT_MODE = "every"

# ============================================================
# BACKTEST / OPTIMIZER SETTINGS
# ============================================================

BACKTEST_YEARS = 10          # days replayed (None = all history after indicator warm-up)
FORWARD_DAYS = [5, 10, 20]   # forward returns in the CSV + report
TARGET_FWD_DAYS = 10         # horizon the optimizer maximizes (buy) / minimizes (sell)
TRAIN_FRACTION = 0.70        # oldest 70% of days to pick rules, newest 30% to test them
MIN_SIGNAL_DAYS = 40         # a rule must fire on at least this many training days...
MIN_EPISODES = 10            # ...spread over at least this many separate streaks
MAX_RULE_CONDITIONS = 3      # conditions in a suggested rule
MIN_IMPROVEMENT_PCT = 0.25   # add another condition only if it improves avg return this much
TOP_SINGLE_CONDITIONS = 12   # rows in the "best single conditions" tables
COLOR_CAP_PCT = 10           # chart markers: +/- this % or beyond = darkest green / red
HEAT_CAP_PCT = 5             # heatmap cells: +/- this % or beyond = darkest

# Output location. Defaults: next to the script. The GitHub Pages workflow overrides.
if os.environ.get("BACKTEST_OUTPUT_DIR"):
    OUT_DIR = Path(os.environ["BACKTEST_OUTPUT_DIR"]).resolve()
else:
    try:
        OUT_DIR = Path(__file__).resolve().parent
    except NameError:
        OUT_DIR = Path.cwd()
HTML_OUT = OUT_DIR / os.environ.get("BACKTEST_HTML_NAME", "mtf_signal_report.html")
CSV_OUT = OUT_DIR / "mtf_signal_days.csv"

SCANNER_TITLE = f"{TICKER} Multi-Timeframe Buy/Sell Signals - Backtest"
SCANNER_ORDER = 20           # after the ATH backtest (10)

PLOTLY_JS = os.environ.get("BACKTEST_PLOTLY_JS", "inline")   # "inline" or "cdn"

GRADIENT = [(-1.0, "#8e1b1b"), (-0.5, "#e0584e"), (0.0, "#e4e2dc"),
            (0.5, "#4fae68"), (1.0, "#0b5a24")]
PENDING_COLOR = "#f1f0ec"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"

MARKET_TZ = "America/New_York"

# ============================================================
# FIELDS
# ============================================================

TIMEFRAMES = [("D", "Daily", "D"), ("W", "Weekly", "W-FRI"), ("M", "Monthly", "M")]

NUM_FIELDS = [   # name, label, step for the filter input, decimals kept
    ("RSI", "RSI (14)", 1, 2),
    ("StochK", "Stochastic %K", 1, 2),
    ("StochD", "Stochastic %D", 1, 2),
    ("MFI", "MFI (14)", 1, 2),
    ("CMF", "CMF (20)", 0.01, 3),
    ("MACD", "MACD line ($)", 0.1, 3),
    ("Signal", "MACD signal ($)", 0.1, 3),
    ("Hist", "MACD histogram ($)", 0.1, 3),
    ("MACD_Pct", "MACD, % of price", 0.1, 3),
    ("Hist_Pct", "Histogram, % of price", 0.1, 3),
]
BOOL_FIELDS = [
    ("MACD_Bull", "MACD above signal"),
    ("MACD_Pos", "MACD above 0"),
    ("Hist_Rising", "Histogram rising (green bar)"),
    ("Stoch_Bull", "%K above %D"),
    ("Flow_Bull", "MFI > 50 and CMF > 0"),
    ("Candle_Up", "Candle up (close > open)"),
]
# The optimizer skips raw-dollar MACD fields: $ values from a $20 stock and a $400 stock
# aren't comparable, so it uses the % of price versions instead.
OPT_NUM_FIELDS = ["RSI", "StochK", "StochD", "MFI", "CMF", "MACD_Pct", "Hist_Pct"]

NUM_KEYS = [f"{tf}_{n}" for tf, _, _ in TIMEFRAMES for n, *_ in NUM_FIELDS]
BOOL_KEYS = [f"{tf}_{n}" for tf, _, _ in TIMEFRAMES for n, _ in BOOL_FIELDS]
ALL_KEYS = NUM_KEYS + BOOL_KEYS
LABEL = {n: lab for n, lab, *_ in NUM_FIELDS} | {n: lab for n, lab in BOOL_FIELDS}
TF_NAME = {tf: name for tf, name, _ in TIMEFRAMES}
DECIMALS = {n: d for n, _, _, d in NUM_FIELDS}


def key_label(k):
    tf, name = k.split("_", 1)
    return f"{TF_NAME[tf]} {LABEL[name]}"


# ============================================================
# INDICATORS (identical formulas to the notebook)
# ============================================================

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_stochastic(df, k_period=14, d_period=3, smooth_k=3):
    low_min = df["Low"].rolling(window=k_period).min()
    high_max = df["High"].rolling(window=k_period).max()
    df["%K"] = 100 * (df["Close"] - low_min) / (high_max - low_min)
    df["%K"] = df["%K"].rolling(window=smooth_k).mean()
    df["%D"] = df["%K"].rolling(window=d_period).mean()
    return df


def calculate_mfi(df, period=14):
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    raw_money_flow = typical_price * df["Volume"]
    up = typical_price > typical_price.shift(1)
    down = typical_price < typical_price.shift(1)
    positive_mf = raw_money_flow.where(up, 0.0).rolling(window=period).sum()
    negative_mf = raw_money_flow.where(down, 0.0).rolling(window=period).sum()
    money_flow_ratio = positive_mf / negative_mf.replace(0, np.nan)
    return 100 - (100 / (1 + money_flow_ratio))


def calculate_cmf(df, period=20):
    hl_range = df["High"] - df["Low"]
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / hl_range.replace(0, np.nan)
    mfm = mfm.fillna(0.0)
    mfv = mfm * df["Volume"]
    return mfv.rolling(window=period).sum() / df["Volume"].rolling(window=period).sum()


def compute_indicators(df):
    """The notebook's fetch_and_process_data() indicator block, on any OHLCV frame."""
    df = df.copy()
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["RSI"] = calculate_rsi(df["Close"], period=14)
    df = calculate_stochastic(df, k_period=14, d_period=3, smooth_k=3)
    df["MFI"] = calculate_mfi(df, period=14)
    df["CMF"] = calculate_cmf(df, period=20)
    return df


# ============================================================
# WEEKLY / MONTHLY AS SEEN LIVE (partial current bar, no look-ahead)
# ============================================================

def partial_bar_indicators(daily: pd.DataFrame, freq: str) -> pd.DataFrame:
    """For every day, the weekly/monthly indicators as they read after that day's close.

    The current week/month is a partial bar (open = first open of the period,
    high/low so far, close = today's close, volume so far). Completed bars before it
    are fixed, so each indicator is updated from the previous bar's state:
    EMAs (MACD, signal, RSI averages) take one step; rolling windows (Stochastic,
    MFI, CMF) combine the previous completed bars with today's partial bar.
    Equivalent to rebuilding the bars up to that day and running the notebook's
    code on them, just vectorized.
    """
    per = daily.index.to_period(freq)
    codes, uniq = pd.factorize(per)          # dates are sorted -> codes are bar positions
    k = codes
    bars = daily.groupby(codes).agg(Open=("Open", "first"), High=("High", "max"), Low=("Low", "min"),
                                    Close=("Close", "last"), Volume=("Volume", "sum"))
    ind = compute_indicators(bars)

    def prev(s, n=1):
        a = np.asarray(s, dtype=float)
        idx = k - n
        out = np.full(len(k), np.nan)
        ok = idx >= 0
        out[ok] = a[idx[ok]]
        return out

    grp = daily.groupby(codes)
    O = grp["Open"].transform("first").values
    H = grp["High"].cummax().values
    L = grp["Low"].cummin().values
    C = daily["Close"].values.astype(float)
    V = grp["Volume"].cumsum().values.astype(float)
    first = k == 0

    with np.errstate(divide="ignore", invalid="ignore"):
        # MACD / signal
        a12, a26, a9 = 2 / 13, 2 / 27, 2 / 10
        e12 = np.where(first, C, a12 * C + (1 - a12) * prev(bars["Close"].ewm(span=12, adjust=False).mean()))
        e26 = np.where(first, C, a26 * C + (1 - a26) * prev(bars["Close"].ewm(span=26, adjust=False).mean()))
        macd = e12 - e26
        sig = np.where(first, macd, a9 * macd + (1 - a9) * prev(ind["Signal"]))

        # RSI (Wilder averages)
        delta = bars["Close"].diff()
        ag = delta.where(delta > 0, 0.0).ewm(alpha=1 / 14, adjust=False).mean()
        al = (-delta.where(delta < 0, 0.0)).ewm(alpha=1 / 14, adjust=False).mean()
        d_p = C - prev(bars["Close"])
        g_p = np.where(d_p > 0, d_p, 0.0)
        l_p = np.where(d_p < 0, -d_p, 0.0)
        ag_p = np.where(first, g_p, g_p / 14 + (13 / 14) * prev(ag))
        al_p = np.where(first, l_p, l_p / 14 + (13 / 14) * prev(al))
        rsi_p = 100 - 100 / (1 + ag_p / al_p)

        # Stochastic 14/3/3
        low_min = np.minimum(L, prev(bars["Low"].rolling(13).min()))
        high_max = np.maximum(H, prev(bars["High"].rolling(13).max()))
        rawk_p = 100 * (C - low_min) / (high_max - low_min)
        rawk = 100 * (bars["Close"] - bars["Low"].rolling(14).min()) / (
            bars["High"].rolling(14).max() - bars["Low"].rolling(14).min())
        k_p = (rawk_p + prev(rawk) + prev(rawk, 2)) / 3
        d_stoch = (k_p + prev(ind["%K"]) + prev(ind["%K"], 2)) / 3

        # MFI 14
        tp = (bars["High"] + bars["Low"] + bars["Close"]) / 3
        rmf = tp * bars["Volume"]
        pos = rmf.where(tp > tp.shift(1), 0.0)
        neg = rmf.where(tp < tp.shift(1), 0.0)
        tp_p = (H + L + C) / 3
        rmf_p = tp_p * V
        tp_prev = prev(tp)
        pos_mf = prev(pos.rolling(13).sum()) + np.where(tp_p > tp_prev, rmf_p, 0.0)
        neg_mf = prev(neg.rolling(13).sum()) + np.where(tp_p < tp_prev, rmf_p, 0.0)
        ratio = pos_mf / np.where(neg_mf == 0, np.nan, neg_mf)
        mfi_p = 100 - 100 / (1 + ratio)

        # CMF 20
        rng = bars["High"] - bars["Low"]
        mfm = (((bars["Close"] - bars["Low"]) - (bars["High"] - bars["Close"])) / rng.replace(0, np.nan)).fillna(0.0)
        mfv = mfm * bars["Volume"]
        rng_p = H - L
        mfm_p = np.where(rng_p == 0, 0.0, ((C - L) - (H - C)) / np.where(rng_p == 0, np.nan, rng_p))
        cmf_p = (prev(mfv.rolling(19).sum()) + mfm_p * V) / (prev(bars["Volume"].rolling(19).sum()) + V)

    return pd.DataFrame({
        "Open": O, "Close": C, "MACD": macd, "Signal": sig, "RSI": rsi_p, "%K": k_p, "%D": d_stoch,
        "MFI": mfi_p, "CMF": cmf_p, "Hist_Prev": prev(ind["MACD"] - ind["Signal"]),
    }, index=daily.index)


def timeframe_fields(tf: str, x: pd.DataFrame) -> pd.DataFrame:
    """Turn one timeframe's indicator columns into the scanner fields (prefix D_/W_/M_)."""
    hist = x["MACD"] - x["Signal"]
    hist_prev = x["Hist_Prev"] if "Hist_Prev" in x else hist.shift(1)
    known = x[["MACD", "Signal", "RSI", "%K", "%D", "MFI", "CMF"]].notna().all(axis=1)

    def flag(cond):
        return cond.astype(float).where(known)

    out = pd.DataFrame({
        "RSI": x["RSI"], "StochK": x["%K"], "StochD": x["%D"], "MFI": x["MFI"], "CMF": x["CMF"],
        "MACD": x["MACD"], "Signal": x["Signal"], "Hist": hist,
        "MACD_Pct": x["MACD"] / x["Close"] * 100, "Hist_Pct": hist / x["Close"] * 100,
        # the notebook colors a histogram bar green when it is >= the previous bar
        "MACD_Bull": flag(x["MACD"] > x["Signal"]),
        "MACD_Pos": flag(x["MACD"] > 0),
        "Hist_Rising": flag(hist_prev.isna() | (hist >= hist_prev)),
        "Stoch_Bull": flag(x["%K"] > x["%D"]),
        "Flow_Bull": flag((x["MFI"] > 50) & (x["CMF"] > 0)),
        "Candle_Up": flag(x["Close"] > x["Open"]),
    }, index=x.index)
    return out.add_prefix(f"{tf}_")


def build_history(daily: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for tf, _, freq in TIMEFRAMES:
        x = compute_indicators(daily) if tf == "D" else partial_bar_indicators(daily, freq)
        parts.append(timeframe_fields(tf, x))
    hist = pd.concat(parts, axis=1)[ALL_KEYS]
    hist.insert(0, "Close", daily["Close"])
    for n in FORWARD_DAYS:
        hist[f"Fwd_{n}D%"] = (daily["Close"].shift(-n) / daily["Close"] - 1) * 100
    return hist


# ============================================================
# RULES: parsing, evaluating, optimizing
# ============================================================

def parse_rule(spec, name):
    """Settings dict -> list of conditions {k, op, v}."""
    conds = []
    for k, v in spec.items():
        if k not in ALL_KEYS:
            raise ValueError(f"{name}: unknown field '{k}'. Valid: {', '.join(ALL_KEYS)}")
        if k in BOOL_KEYS:
            if not isinstance(v, bool):
                raise ValueError(f"{name}: '{k}' is a yes/no field - use True or False")
            conds.append({"k": k, "op": "==", "v": 1 if v else 0})
        else:
            lo, hi = v
            if lo is not None:
                conds.append({"k": k, "op": ">=", "v": float(lo)})
            if hi is not None:
                conds.append({"k": k, "op": "<=", "v": float(hi)})
    return conds


def cond_mask(df, c):
    col = df[c["k"]].values
    with np.errstate(invalid="ignore"):
        if c["op"] == ">=":
            return col >= c["v"]
        if c["op"] == "<=":
            return col <= c["v"]
        return col == c["v"]


def rule_mask(df, conds):
    if not conds:
        return np.zeros(len(df), dtype=bool)
    m = np.ones(len(df), dtype=bool)
    for c in conds:
        m &= cond_mask(df, c)
    return m


def cond_text(c):
    tf, name = c["k"].split("_", 1)
    if c["op"] == "==":
        return f"{TF_NAME[tf]} {LABEL[name]}: {'yes' if c['v'] == 1 else 'no'}"
    return f"{TF_NAME[tf]} {LABEL[name]} {'≥' if c['op'] == '>=' else '≤'} {c['v']:g}"


def rule_text(conds):
    return " AND ".join(cond_text(c) for c in conds) if conds else "(no conditions)"


def stats(mask, y):
    """Days the rule fired, separate streaks, and forward-return stats."""
    mask = np.asarray(mask, dtype=bool)
    episodes = int(mask[0]) + int(np.sum(mask[1:] & ~mask[:-1])) if len(mask) else 0
    v = y[mask & ~np.isnan(y)]
    if not len(v):
        return {"n": int(mask.sum()), "ep": episodes, "avg": None, "med": None, "win": None}
    return {"n": int(mask.sum()), "ep": episodes, "avg": round(float(v.mean()), 2),
            "med": round(float(np.median(v)), 2), "win": round(float((v > 0).mean() * 100), 1)}


def nice(name, v):
    if name in ("RSI", "StochK", "StochD", "MFI"):
        return float(round(v))
    if name == "CMF":
        return float(round(v, 2))
    return float(round(v, 2 if abs(v) < 1 else 1))


def candidates(train):
    out = []
    for tf, _, _ in TIMEFRAMES:
        for name in OPT_NUM_FIELDS:
            k = f"{tf}_{name}"
            qs = np.nanquantile(train[k].values, np.arange(0.05, 0.951, 0.05))
            for q in sorted({nice(name, q) for q in qs}):
                out += [{"k": k, "op": ">=", "v": q}, {"k": k, "op": "<=", "v": q}]
        for name, _ in BOOL_FIELDS:
            k = f"{tf}_{name}"
            out += [{"k": k, "op": "==", "v": 1}, {"k": k, "op": "==", "v": 0}]
    return out


def optimize(train, test, direction):
    """direction +1 = buy (maximize forward return), -1 = sell (minimize it).

    Rules are picked on the training days only; the test days are only scored.
    """
    tgt = f"Fwd_{TARGET_FWD_DAYS}D%"
    y_tr, y_te = train[tgt].values, test[tgt].values
    cands = candidates(train)
    masks = [cond_mask(train, c) for c in cands]
    has_y = ~np.isnan(y_tr)

    def score(m):
        mm = m & has_y
        if mm.sum() < MIN_SIGNAL_DAYS:
            return None
        ep = int(m[0]) + int(np.sum(m[1:] & ~m[:-1]))
        if ep < MIN_EPISODES:
            return None
        return direction * float(y_tr[mm].mean())

    # best single condition per field
    singles = {}
    for c, m in zip(cands, masks):
        s = score(m)
        if s is not None and (c["k"] not in singles or s > singles[c["k"]][0]):
            singles[c["k"]] = (s, c)
    top = sorted(singles.values(), key=lambda x: -x[0])[:TOP_SINGLE_CONDITIONS]
    single_rows = [{"cond": c, "text": cond_text(c),
                    "train": stats(cond_mask(train, c), y_tr), "test": stats(cond_mask(test, c), y_te)}
                   for _, c in top]

    # greedy combined rule
    rule, cur, cur_score = [], np.ones(len(train), dtype=bool), None
    for _ in range(MAX_RULE_CONDITIONS):
        used = {c["k"] for c in rule}
        best = None
        for c, m in zip(cands, masks):
            if c["k"] in used:
                continue
            s = score(cur & m)
            if s is not None and (best is None or s > best[0]):
                best = (s, c, cur & m)
        if best is None or (cur_score is not None and best[0] - cur_score < MIN_IMPROVEMENT_PCT):
            break
        cur_score, cur = best[0], best[2]
        rule.append(best[1])

    return {
        "rule": rule, "text": rule_text(rule),
        "train": stats(rule_mask(train, rule), y_tr),
        "test": stats(rule_mask(test, rule), y_te),
        "singles": single_rows,
    }


def streak(mask):
    n = 0
    for v in mask[::-1]:
        if not v:
            break
        n += 1
    return n


# ============================================================
# DATA
# ============================================================

def load_daily(ticker):
    df = yf.download(ticker, period=DATA_PERIOD, interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise SystemExit(f"No data returned for {ticker}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)[["Open", "High", "Low", "Close", "Volume"]]
    df.index = pd.DatetimeIndex(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(MARKET_TZ).tz_localize(None)
    df.index = df.index.normalize()
    return df[~df.index.duplicated(keep="last")].dropna(subset=["Close"]).sort_index()


def prepare():
    daily = load_daily(TICKER)
    hist = build_history(daily)
    valid = hist[NUM_KEYS].notna().all(axis=1)
    if not valid.any():
        raise SystemExit(f"{TICKER}: not enough history to warm up the monthly indicators.")
    end = hist.index.max()
    start = hist.index[valid.values.argmax()]
    if BACKTEST_YEARS:
        start = max(start, end - pd.DateOffset(years=BACKTEST_YEARS))
    win = hist[(hist.index >= start) & valid].copy()

    tgt = f"Fwd_{TARGET_FWD_DAYS}D%"
    scored = win.index[win[tgt].notna()]
    split_date = scored[int(len(scored) * TRAIN_FRACTION)]
    train, test = win[win.index < split_date], win[win.index >= split_date]

    suggest = {"buy": optimize(train, test, +1), "sell": optimize(train, test, -1)}
    rules = {}
    for side, spec in (("buy", BUY_RULE), ("sell", SELL_RULE)):
        if isinstance(spec, str) and spec.lower() == "auto":
            rules[side] = {"conds": suggest[side]["rule"], "source": "auto"}
        else:
            rules[side] = {"conds": parse_rule(spec, f"{side.upper()}_RULE"), "source": "settings"}
        rules[side]["text"] = rule_text(rules[side]["conds"])
        win[f"{side.title()}_Signal"] = rule_mask(win, rules[side]["conds"])
    win["Period"] = np.where(win.index < split_date, "train", "test")
    return win, train, test, split_date, suggest, rules


# ============================================================
# ALERTS
# ============================================================

def latest_status(win, rules):
    last = win.iloc[-1]
    out = {}
    for side in ("buy", "sell"):
        conds = rules[side]["conds"]
        mask = win[f"{side.title()}_Signal"].values
        checks = [(c, float(last[c["k"]]), bool(cond_mask(win.iloc[[-1]], c)[0])) for c in conds]
        out[side] = {"fires": bool(mask[-1]), "streak": streak(mask), "checks": checks}
    return out


def fmt_val(k, v):
    if k in BOOL_KEYS:
        return "yes" if v == 1 else "no"
    return f"{v:,.{min(DECIMALS[k.split('_', 1)[1]], 2)}f}"


def send_email(subject, body):
    user, pw = os.environ.get("EMAIL_USER"), os.environ.get("EMAIL_PASS")
    to = os.environ.get("ALERT_TO") or user
    if not (user and pw):
        print("EMAIL_USER / EMAIL_PASS not set - alert printed only, no email sent.")
        return False
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, user, to
    msg.set_content(body)
    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "465"))
    with smtplib.SMTP_SSL(host, port, timeout=30) as s:
        s.login(user, pw)
        s.send_message(msg)
    print(f"Alert emailed to {to}.")
    return True


def run_alerts(win, suggest, rules, split_date, force=False):
    day = win.index[-1]
    today_et = datetime.now(ZoneInfo(MARKET_TZ)).date()
    if not force and (today_et - day.date()).days > 1:
        print(f"Latest bar is {day:%Y-%m-%d} (market closed today?) - no alert. Use --force to send anyway.")
        return
    status = latest_status(win, rules)
    last = win.iloc[-1]
    fired = []
    for side in ("buy", "sell"):
        st = status[side]
        if not rules[side]["conds"]:
            print(f"{side.upper()}: no rule conditions - skipped.")
            continue
        print(f"{side.upper()} rule ({rules[side]['source']}): {rules[side]['text']} -> "
              f"{'FIRES' if st['fires'] else 'no signal'}"
              + (f" (day {st['streak']} in a row)" if st["fires"] else ""))
        if st["fires"] and (ALERT_MODE == "every" or st["streak"] == 1):
            fired.append(side)
    if not fired:
        print("No alerts today.")
        return

    lines = [f"{TICKER} close {day:%a %b %d, %Y}: ${last['Close']:,.2f}", ""]
    for side in fired:
        st, r = status[side], rules[side]
        te = suggest[side]["test"] if r["source"] == "auto" else stats(
            win[f"{side.title()}_Signal"].values[win.index >= split_date],
            win[f"Fwd_{TARGET_FWD_DAYS}D%"].values[win.index >= split_date])
        lines += [f"=== {side.upper()} ALERT (day {st['streak']} in a row) ===",
                  f"Rule ({'suggested by the optimizer' if r['source'] == 'auto' else 'from your settings'}):"]
        lines += [f"  - {cond_text(c)}   (today: {fmt_val(c['k'], v)})" for c, v, _ in st["checks"]]
        if te["avg"] is not None:
            lines.append(f"Backtest, test period since {split_date:%b %Y}: fired on {te['n']} days, "
                         f"avg {TARGET_FWD_DAYS}-day forward return {te['avg']:+.2f}%, "
                         f"{te['win']:.0f}% of them positive.")
        lines.append("")
    lines += ["Today's readings (Daily / Weekly / Monthly):"]
    for name, lab, *_ in NUM_FIELDS:
        vals = " / ".join(fmt_val(f"{tf}_{name}", last[f"{tf}_{name}"]) for tf, _, _ in TIMEFRAMES)
        lines.append(f"  {lab:<28} {vals}")
    for name, lab in BOOL_FIELDS:
        vals = " / ".join(fmt_val(f"{tf}_{name}", last[f"{tf}_{name}"]) for tf, _, _ in TIMEFRAMES)
        lines.append(f"  {lab:<28} {vals}")
    if os.environ.get("DASHBOARD_URL"):
        lines += ["", f"Dashboard: {os.environ['DASHBOARD_URL']}"]
    lines += ["", "Automated scanner alert based on historical indicator behavior. Not financial advice."]
    body = "\n".join(lines)
    subject = f"{TICKER} {' + '.join(s.upper() for s in fired)} alert - {day:%b %d, %Y} close ${last['Close']:,.2f}"
    print("\n" + subject + "\n" + body)
    send_email(subject, body)


# ============================================================
# REPORT
# ============================================================

def _col(series, nd):
    return [None if pd.isna(v) else round(float(v), nd) for v in series]


def build_report(win, split_date, suggest, rules):
    tgt = f"Fwd_{TARGET_FWD_DAYS}D%"
    meta = []
    for tf, tfname, _ in TIMEFRAMES:
        for name, lab, step, nd in NUM_FIELDS:
            k = f"{tf}_{name}"
            edges = np.nanquantile(win[k].values, np.linspace(0, 1, 11))
            edges = sorted({round(float(e), nd) for e in edges})
            meta.append({"k": k, "tf": tf, "tfName": tfname, "name": name, "label": lab,
                         "kind": "num", "step": step, "edges": edges})
        for name, lab in BOOL_FIELDS:
            meta.append({"k": f"{tf}_{name}", "tf": tf, "tfName": tfname, "name": name, "label": lab,
                         "kind": "bool"})

    feats = {}
    for k in ALL_KEYS:
        feats[k] = ([None if pd.isna(v) else int(v) for v in win[k]] if k in BOOL_KEYS
                    else _col(win[k], DECIMALS[k.split("_", 1)[1]]))

    def js_stats_rows(rows):
        return [{"cond": r["cond"], "text": r["text"], "train": r["train"], "test": r["test"]} for r in rows]

    data = {
        "ticker": TICKER,
        "dates": win.index.strftime("%Y-%m-%d").tolist(),
        "close": _col(win["Close"], 2),
        "feats": feats,
        "fwd": {str(n): _col(win[f"Fwd_{n}D%"], 2) for n in FORWARD_DAYS},
        "fwdDays": FORWARD_DAYS,
        "target": TARGET_FWD_DAYS,
        "split": int((win.index < split_date).sum()),
        "splitDate": f"{split_date:%Y-%m-%d}",
        "meta": meta,
        "tfs": [{"tf": tf, "name": name} for tf, name, _ in TIMEFRAMES],
        "numNames": [{"name": n, "label": lab, "step": s} for n, lab, s, _ in NUM_FIELDS],
        "boolNames": [{"name": n, "label": lab} for n, lab in BOOL_FIELDS],
        "rules": {s: {"conds": rules[s]["conds"], "source": rules[s]["source"]} for s in rules},
        "suggest": {s: {"rule": suggest[s]["rule"], "text": suggest[s]["text"], "train": suggest[s]["train"],
                        "test": suggest[s]["test"], "singles": js_stats_rows(suggest[s]["singles"])}
                    for s in suggest},
        "opt": {"minDays": MIN_SIGNAL_DAYS, "minEp": MIN_EPISODES, "maxConds": MAX_RULE_CONDITIONS},
        "cap": COLOR_CAP_PCT, "heatCap": HEAT_CAP_PCT, "gradient": GRADIENT, "pending": PENDING_COLOR,
    }

    start, end = win.index.min(), win.index.max()
    plotly_tag = (f"<script>{get_plotlyjs()}</script>" if PLOTLY_JS == "inline" else
                  f'<script src="https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"></script>')
    html = (PAGE_TEMPLATE
            .replace("__TICKER__", TICKER)
            .replace("__PERIOD__", f"{start:%b %d, %Y} → {end:%b %d, %Y}")
            .replace("__SPLIT__", f"{split_date:%b %d, %Y}")
            .replace("__TRAINPCT__", f"{TRAIN_FRACTION * 100:.0f}")
            .replace("__CSV__", CSV_OUT.name)
            .replace("__SURFACE__", SURFACE).replace("__INK2__", INK_2)
            .replace("__INK__", INK).replace("__GRID__", GRID))
    html = html.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    html = html.replace("__PLOTLY__", plotly_tag)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HTML_OUT.write_text(html, encoding="utf-8")

    if os.environ.get("BACKTEST_OUTPUT_DIR"):
        st = latest_status(win, rules)
        now = " + ".join(s.upper() for s in ("buy", "sell") if st[s]["fires"]) or "no signal"
        (OUT_DIR / "scanner.json").write_text(json.dumps({
            "title": SCANNER_TITLE,
            "order": SCANNER_ORDER,
            "page": HTML_OUT.name,
            "subtitle": f"{start:%b %d, %Y} to {end:%b %d, %Y} · buy rule fired {int(win['Buy_Signal'].sum())} days, "
                        f"sell rule {int(win['Sell_Signal'].sum())} days · latest close: {now}",
        }, indent=2), encoding="utf-8")


PAGE_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>__TICKER__ Buy/Sell Signal Backtest</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
__PLOTLY__
<style>
  body { background:__SURFACE__; color:__INK__; font-family:Inter,'Segoe UI',Arial,sans-serif; margin:0; padding:24px 16px; }
  .wrap { max-width:1280px; margin:0 auto; }
  h1 { font-size:22px; margin:0 0 4px; }
  .sub { color:__INK2__; font-size:13px; margin-bottom:16px; line-height:1.55; }
  .card { border:1px solid __GRID__; border-radius:10px; padding:10px 12px; margin-bottom:18px; background:#fff; }
  h2 { font-size:16px; font-weight:600; margin:4px 2px 8px; }
  .note { color:__INK2__; font-size:12px; margin:8px 2px; line-height:1.5; }
  .row { display:flex; flex-wrap:wrap; align-items:center; gap:10px 14px; }
  .ctl { display:flex; align-items:center; gap:6px; font-size:13px; color:__INK2__; }
  select, input { font:inherit; font-size:13px; padding:5px 6px; border:1px solid #c9c8c2; border-radius:6px; background:#fff; color:__INK__; }
  input.n { width:62px; }
  button { font:inherit; font-size:13px; padding:6px 11px; border:1px solid #c9c8c2; border-radius:6px; background:#fff; color:__INK__; cursor:pointer; }
  button:hover { background:#f0efec; }
  .tabs { display:flex; gap:0; margin-bottom:10px; }
  .tab { border-radius:0; padding:8px 18px; font-weight:600; }
  .tab:first-child { border-radius:8px 0 0 8px; } .tab:last-child { border-radius:0 8px 8px 0; border-left:0; }
  .tab.on.buy { background:#0b5a24; color:#fff; border-color:#0b5a24; }
  .tab.on.sell { background:#8e1b1b; color:#fff; border-color:#8e1b1b; }
  .today { display:flex; flex-wrap:wrap; gap:12px; margin-bottom:16px; }
  .badge { border-radius:10px; padding:12px 16px; min-width:260px; flex:1; border:1px solid __GRID__; background:#fff; }
  .badge .t { font-size:12px; color:__INK2__; text-transform:uppercase; letter-spacing:.04em; }
  .badge .v { font-size:22px; font-weight:700; margin:2px 0; }
  .badge .r { font-size:12px; color:__INK2__; line-height:1.45; }
  .badge.fire.buy { background:#e8f5ec; border-color:#4fae68; } .badge.fire.buy .v { color:#0b5a24; }
  .badge.fire.sell { background:#fbeaea; border-color:#e0584e; } .badge.fire.sell .v { color:#8e1b1b; }
  table { border-collapse:collapse; width:100%; font-size:13px; }
  th, td { text-align:left; padding:5px 7px; border-bottom:1px solid __GRID__; white-space:nowrap; }
  th { color:__INK2__; font-weight:500; }
  .num { text-align:right; font-variant-numeric:tabular-nums; }
  .tscroll { overflow-x:auto; }
  table.grid td { vertical-align:top; }
  table.grid td.lab { font-weight:500; padding-top:9px; }
  table.grid th.tf { text-align:center; border-left:1px solid __GRID__; }
  table.grid td.tf { border-left:1px solid __GRID__; }
  .cell { display:flex; align-items:center; gap:4px; }
  .now { font-size:11px; color:__INK2__; margin-top:2px; }
  .now.ok { color:#0b5a24; font-weight:600; }
  .now.bad { color:#8e1b1b; }
  tr.set td.tf.has { background:#f6f5f0; }
  .tiles { display:flex; flex-wrap:wrap; gap:10px; margin:4px 0 6px; }
  .tile { border:1px solid __GRID__; border-radius:8px; padding:8px 14px; min-width:112px; background:#fff; }
  .tile .v { font-size:20px; font-weight:600; } .tile .l { font-size:12px; color:__INK2__; }
  .sidehead { font-size:13px; font-weight:700; margin:8px 2px 2px; }
  .sidehead.buy { color:#0b5a24; } .sidehead.sell { color:#8e1b1b; }
  .rtext { font-size:13px; color:__INK2__; margin:2px 2px 6px; }
  table.heat td.c { text-align:center; cursor:pointer; min-width:54px; font-size:12px; border:2px solid #fff; border-radius:4px; }
  table.heat td.c:hover { outline:2px solid __INK__; }
  table.heat td.c.thin { opacity:.45; }
  table.heat td.c.sel { outline:2px solid #2a78d6; }
  table.heat tr.tfh td { font-weight:700; background:#f7f6f2; }
  table.heat .rng { display:block; font-size:10px; opacity:.8; }
  .sw { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; vertical-align:-1px; border:1px solid rgba(0,0,0,.08); }
  tr.click { cursor:pointer; } tr.click:hover td { background:#f5f4f0; }
  .two { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
  .two > div { min-width:0; overflow-x:auto; }
  select { max-width:100%; }
  .ctl { flex-wrap:wrap; max-width:100%; }
  .hint { color:__INK2__; font-size:12px; }
  @media (max-width: 900px) { .two { grid-template-columns:1fr; } }
  .pill { display:inline-block; font-size:11px; padding:1px 7px; border-radius:10px; background:#f0efec; color:__INK2__; margin-left:6px; }
  a { color:#2a78d6; }
</style></head><body><div class="wrap">
<h1>__TICKER__ multi-timeframe buy &amp; sell signals: backtest</h1>
<div class="sub">__PERIOD__ · every trading day replayed after the close with the notebook's indicators on the Daily,
Weekly (week-to-date) and Monthly (month-to-date) timeframes. <b>Buy rule</b> = conditions followed by the highest forward
returns; <b>sell rule</b> = conditions followed by the lowest. Colors show the forward return after a day:
<span class="sw" style="background:#0b5a24"></span>green = price rose afterwards,
<span class="sw" style="background:#8e1b1b"></span>red = price fell. So good buy signals are green and good sell signals are red.
Rules are suggested from the older __TRAINPCT__% of days and checked on days since __SPLIT__ (out-of-sample).</div>

<div class="today" id="today"></div>

<div class="card">
  <div class="row" style="justify-content:space-between">
    <div class="tabs"><button class="tab buy" data-side="buy">Buy rule</button><button class="tab sell" data-side="sell">Sell rule</button></div>
    <div class="row">
      <label class="ctl">Forward return <select id="horizon"></select></label>
      <label class="ctl">Days <select id="period">
        <option value="all">all days</option><option value="train">training period only</option>
        <option value="test">test period only (since __SPLIT__)</option></select></label>
    </div>
  </div>
  <div class="row" style="margin-bottom:8px">
    <button id="resetBtn" title="The rule the scanner uses for email alerts">Reset to scanner rule</button>
    <button id="suggestBtn">Load suggested rule</button>
    <button id="clearBtn">Clear rule</button>
    <span class="note" id="ruleText" style="margin:0"></span>
  </div>
  <div class="tscroll"><table class="grid" id="grid"></table></div>
  <div class="note">All filled-in conditions must pass (AND). Leave a box empty for no limit. The small line under each box is
  the latest close's value (green = passes this rule's condition). Numbers are on the notebook's scales: RSI, %K, %D and MFI 0 to 100,
  CMF -1 to 1, MACD fields in dollars (the "% of price" versions compare better across years).</div>
</div>

<div class="card"><h2>Rule results <span class="pill" id="periodPill"></span></h2>
  <div class="sidehead buy">Buy rule</div><div class="rtext" id="buyText"></div><div class="tiles" id="buyTiles"></div>
  <div class="sidehead sell">Sell rule</div><div class="rtext" id="sellText"></div><div class="tiles" id="sellTiles"></div>
  <div class="note">"Edge" = rule's average forward return minus the average of all days in the same period. A buy rule wants a
  positive edge, a sell rule a negative one. Streaks = separate runs of consecutive signal days; consecutive days overlap in
  their forward windows, so streaks are the more honest count.</div>
</div>

<div class="card"><h2>Price with buy ▲ and sell ▼ signals</h2>
  <div class="row"><label class="ctl"><input type="checkbox" id="logY" checked> log price scale</label></div>
  <div id="priceChart"></div></div>

<div class="card"><h2 id="heatTitle">Where the forward returns are: average by indicator value</h2>
  <div class="row" style="margin-bottom:6px">
    <label class="ctl">Days included <select id="heatCtx">
      <option value="all">all days in the period</option>
      <option value="rule">days passing the active rule's OTHER conditions</option></select></label>
    <span class="note" style="margin:0">Each row splits that indicator's values into 10 equal-count ranges. Click a cell to set that range in the
    active rule (the blue outline marks its current range). Faded = under 20 days.</span>
  </div>
  <div class="tscroll"><table class="heat" id="heat"></table></div>
  <h2 style="margin-top:14px">Yes/no fields</h2>
  <div class="tscroll"><table class="heat" id="heatBool"></table></div>
</div>

<div class="card"><h2>Suggested rules</h2>
  <div class="note">Searched on the training period only (before __SPLIT__): single thresholds at every 5th percentile of each
  indicator plus the yes/no fields, then combined greedily up to <span id="optMax"></span> conditions. Each rule had to fire on at least
  <span id="optDays"></span> days in <span id="optEp"></span>+ separate streaks. The <b>test</b> columns are days the search never saw, so they're the fair check. A rule that
  looks great in training but not in test is probably luck. Raw-dollar MACD fields are left out of the search.</div>
  <div class="two">
    <div><div class="sidehead buy">Buy: combined rule</div><div id="sugBuy"></div></div>
    <div><div class="sidehead sell">Sell: combined rule</div><div id="sugSell"></div></div>
  </div>
  <div class="two" style="margin-top:12px">
    <div><div class="sidehead buy">Buy: best single conditions (click to add to buy rule)</div><div class="tscroll"><table id="singBuy"></table></div></div>
    <div><div class="sidehead sell">Sell: best single conditions (click to add to sell rule)</div><div class="tscroll"><table id="singSell"></table></div></div>
  </div>
</div>

<div class="card"><h2 id="hitsTitle">Recent signal days</h2>
  <div class="tscroll"><table id="hits"></table></div>
  <div class="note"><a href="__CSV__" download>Download every day with all indicators (CSV)</a>. Buy_Signal / Sell_Signal = the scanner's
  alert rules. Backtest uses split-adjusted prices, ignores costs and taxes, and is not financial advice.</div>
</div>
</div>

<script>
const D = __DATA__;
const N = D.dates.length, F = D.feats;
const SURF = '__SURFACE__', INK = '__INK__', INK2 = '__INK2__', GRIDC = '__GRID__';
const $ = id => document.getElementById(id);
const MK = Object.fromEntries(D.meta.map(m => [m.k, m]));
let side = 'buy', H = String(D.target);
const state = {buy: {}, sell: {}};

// ---------- helpers ----------
const hexRgb = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
function color(v, cap) {
  if (v === null || v === undefined || Number.isNaN(v)) return D.pending;
  const x = Math.max(-1, Math.min(1, v / cap)), g = D.gradient;
  for (let k = 0; k < g.length - 1; k++) {
    const [p0, c0] = g[k], [p1, c1] = g[k + 1];
    if (x <= p1) {
      const f = (x - p0) / (p1 - p0), a = hexRgb(c0), b = hexRgb(c1);
      return '#' + a.map((v0, j) => Math.round(v0 + (b[j] - v0) * f).toString(16).padStart(2, '0')).join('');
    }
  }
  return g[g.length - 1][1];
}
const txtColor = (v, cap) => (v === null || Math.abs(v) / cap < 0.45) ? INK : '#fff';
const pct = (v, d = 2) => v === null || v === undefined ? 'n/a' : (v > 0 ? '+' : '') + v.toFixed(d) + '%';
const mean = a => a.length ? a.reduce((s, v) => s + v, 0) / a.length : null;
function median(a) { if (!a.length) return null; const s = [...a].sort((x, y) => x - y), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; }
const fmtNum = v => v === null ? 'n/a' : (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));
const fmtVal = (k, v) => v === null ? 'n/a' : MK[k].kind === 'bool' ? (v ? 'yes' : 'no') : fmtNum(v);
const sideColor = s => s === 'buy' ? '#0b5a24' : '#8e1b1b';
const inPeriod = i => { const p = $('period').value; return p === 'all' || (p === 'train' ? i < D.split : i >= D.split); };

// ---------- rules ----------
function condsToState(conds) {
  const s = {};
  conds.forEach(c => { const e = s[c.k] || (s[c.k] = {});
    if (c.op === '>=') e.min = c.v; else if (c.op === '<=') e.max = c.v; else e.sel = c.v; });
  return s;
}
function active(e) { return e && (e.min != null || e.max != null || e.sel != null); }
function condOk(v, e) {
  if (v === null) return false;
  if (e.sel != null) return v === e.sel;
  if (e.min != null && v < e.min) return false;
  if (e.max != null && v > e.max) return false;
  return true;
}
function ruleKeys(s, exclude) { return Object.keys(state[s]).filter(k => k !== exclude && active(state[s][k])); }
function mask(s, exclude, emptyAll) {
  const keys = ruleKeys(s, exclude), m = new Array(N);
  if (!keys.length) return m.fill(!!emptyAll);
  for (let i = 0; i < N; i++) { let ok = true;
    for (const k of keys) if (!condOk(F[k][i], state[s][k])) { ok = false; break; }
    m[i] = ok; }
  return m;
}
function ruleDesc(s) {
  const keys = ruleKeys(s);
  if (!keys.length) return 'no conditions: the rule never fires';
  return keys.map(k => { const e = state[s][k], m = MK[k], lab = `${m.tfName} ${m.label}`;
    if (e.sel != null) return `${lab}: ${e.sel ? 'yes' : 'no'}`;
    if (e.min != null && e.max != null) return `${lab} ${e.min} to ${e.max}`;
    return e.min != null ? `${lab} ≥ ${e.min}` : `${lab} ≤ ${e.max}`; }).join(' AND ');
}

// ---------- rule grid ----------
function drawGrid() {
  document.querySelectorAll('.tab').forEach(b => b.classList.toggle('on', b.dataset.side === side));
  let h = '<tr><th>Indicator</th>' + D.tfs.map(t => `<th class="tf">${t.name}</th>`).join('') + '</tr>';
  D.numNames.forEach(n => {
    h += `<tr class="set"><td class="lab">${n.label}</td>` + D.tfs.map(t => {
      const k = `${t.tf}_${n.name}`;
      return `<td class="tf" data-k="${k}"><div class="cell"><input class="n" type="number" step="${n.step}" data-k="${k}" data-b="min" placeholder="min">` +
        `to <input class="n" type="number" step="${n.step}" data-k="${k}" data-b="max" placeholder="max"></div><div class="now" data-now="${k}"></div></td>`;
    }).join('') + '</tr>';
  });
  D.boolNames.forEach(n => {
    h += `<tr class="set"><td class="lab">${n.label}</td>` + D.tfs.map(t => {
      const k = `${t.tf}_${n.name}`;
      return `<td class="tf" data-k="${k}"><select data-k="${k}" data-b="sel"><option value="">any</option><option value="1">yes</option><option value="0">no</option></select>` +
        `<div class="now" data-now="${k}"></div></td>`;
    }).join('') + '</tr>';
  });
  $('grid').innerHTML = h;
  $('grid').querySelectorAll('input, select').forEach(el => el.addEventListener('change', () => {
    const k = el.dataset.k, b = el.dataset.b, e = state[side][k] || (state[side][k] = {});
    const v = el.value.trim();
    if (b === 'sel') e.sel = v === '' ? null : Number(v); else e[b] = v === '' ? null : Number(v);
    render();
  }));
  fillGrid();
}
function fillGrid() {
  $('grid').querySelectorAll('input, select').forEach(el => {
    const e = state[side][el.dataset.k] || {}, v = e[el.dataset.b];
    el.value = v == null ? '' : String(v);
  });
  $('grid').querySelectorAll('td.tf').forEach(td => td.classList.toggle('has', active(state[side][td.dataset.k])));
  $('grid').querySelectorAll('[data-now]').forEach(el => {
    const k = el.dataset.now, v = F[k][N - 1], e = state[side][k];
    el.textContent = 'latest: ' + fmtVal(k, v);
    el.className = 'now' + (active(e) ? (condOk(v, e) ? ' ok' : ' bad') : '');
  });
}

// ---------- stats ----------
function statsOf(m, h) {
  const idx = []; for (let i = 0; i < N; i++) if (m[i] && inPeriod(i)) idx.push(i);
  const set = new Set(idx); let ep = 0; idx.forEach(i => { if (!set.has(i - 1)) ep++; });
  const v = idx.map(i => D.fwd[h][i]).filter(x => x !== null);
  return {idx, n: idx.length, ep, avg: mean(v), med: median(v), win: v.length ? v.filter(x => x > 0).length / v.length * 100 : null, nres: v.length};
}
function baseline(h) { const all = new Array(N).fill(true); return statsOf(all, h); }

function tiles(s, m) {
  const st = statsOf(m, H), b = baseline(H), edge = st.avg === null || b.avg === null ? null : st.avg - b.avg;
  const t = [
    [st.n.toLocaleString(), 'signal days'],
    [st.ep, 'separate streaks'],
    [pct(st.avg), `avg ${H}D forward`],
    [pct(st.med), `median ${H}D`],
    [st.win === null ? 'n/a' : st.win.toFixed(0) + '%', `${H}D positive`],
    [edge === null ? 'n/a' : pct(edge), 'edge vs all days'],
    [pct(b.avg), `all days avg ${H}D`],
  ];
  $(s + 'Tiles').innerHTML = t.map(([v, l], j) => {
    const sw = (j === 2 && st.avg !== null) ? `<span class="sw" style="background:${color(st.avg, D.heatCap)}"></span>` : '';
    return `<div class="tile"><div class="v">${sw}${v}</div><div class="l">${l}</div></div>`; }).join('');
  $(s + 'Text').textContent = ruleDesc(s);
  return st;
}

// ---------- today ----------
function today(mb, ms) {
  const d = new Date(D.dates[N - 1] + 'T00:00:00Z').toLocaleDateString('en-US', {weekday: 'short', month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC'});
  const one = (s, m) => {
    const keys = ruleKeys(s); let streak = 0; for (let i = N - 1; i >= 0 && m[i]; i--) streak++;
    const fires = keys.length && m[N - 1];
    const detail = keys.length ? keys.map(k => `${MK[k].tfName} ${MK[k].label}: ${fmtVal(k, F[k][N - 1])} ${condOk(F[k][N - 1], state[s][k]) ? '✓' : '✗'}`).join(' · ') : 'no conditions set';
    return `<div class="badge ${s} ${fires ? 'fire' : ''}"><div class="t">${s} rule on the latest close (${d}, $${D.close[N - 1].toFixed(2)})</div>` +
      `<div class="v">${fires ? (s === 'buy' ? 'BUY signal' : 'SELL signal') + (streak > 1 ? ` · day ${streak}` : '') : 'No ' + s + ' signal'}</div><div class="r">${detail}</div></div>`;
  };
  $('today').innerHTML = one('buy', mb) + one('sell', ms);
}

// ---------- price chart ----------
function scaleTrace(x0) {
  const cap = D.cap;
  return {type: 'scatter', x: [x0], y: [null], mode: 'markers', hoverinfo: 'skip', showlegend: false,
    marker: {size: 0.1, opacity: 0, color: [0], cmin: -cap, cmax: cap, showscale: true,
      colorscale: D.gradient.map(([p, c]) => [(p + 1) / 2, c]),
      colorbar: {orientation: 'h', x: 1, xanchor: 'right', y: 1.02, yanchor: 'bottom', len: 0.34, thickness: 10, outlinewidth: 0,
        tickfont: {size: 10}, tickvals: [-cap, -cap / 2, 0, cap / 2, cap],
        ticktext: [`≤ -${cap}%`, `-${cap / 2}%`, '0%', `+${cap / 2}%`, `≥ +${cap}%`],
        title: {text: `${H}D forward return`, side: 'top', font: {size: 11}}}}};
}
function chart(sb, ss) {
  const tr = [{type: 'scatter', mode: 'lines', x: D.dates, y: D.close, name: 'Close', line: {color: '#8a8984', width: 1.2},
    hovertemplate: '%{x|%b %d, %Y}<br>Close $%{y:,.2f}<extra></extra>', showlegend: false}];
  const mk = (st, s) => {
    const x = [], y = [], c = [], cd = [];
    st.idx.forEach(i => { const v = D.fwd[H][i]; x.push(D.dates[i]); y.push(D.close[i]); c.push(color(v, D.cap));
      cd.push(D.fwdDays.map(n => pct(D.fwd[String(n)][i]))); });
    return {type: 'scatter', mode: 'markers', x, y, customdata: cd, name: s === 'buy' ? 'Buy ▲' : 'Sell ▼', showlegend: false,
      marker: {symbol: s === 'buy' ? 'triangle-up' : 'triangle-down', size: 10, color: c, line: {color: sideColor(s), width: 1.2}},
      hovertemplate: `<b>${s.toUpperCase()}</b> %{x|%b %d, %Y}<br>Close $%{y:,.2f}<br>` +
        D.fwdDays.map((n, j) => `${n}D fwd: %{customdata[${j}]}`).join(' · ') + '<extra></extra>'};
  };
  tr.push(mk(sb, 'buy'), mk(ss, 'sell'), scaleTrace(D.dates[0]));
  const lay = {height: 520, margin: {l: 60, r: 20, t: 60, b: 40}, plot_bgcolor: SURF, paper_bgcolor: '#fff', uirevision: 'keep',
    font: {family: 'Inter, Segoe UI, Arial, sans-serif', color: INK2, size: 12}, hoverlabel: {bgcolor: '#fff', font: {color: INK}},
    xaxis: {showgrid: false, linecolor: GRIDC, rangeslider: {visible: true, thickness: 0.06}},
    yaxis: {type: $('logY').checked ? 'log' : 'linear', gridcolor: GRIDC, tickprefix: '$'},
    shapes: [{type: 'rect', xref: 'x', yref: 'paper', x0: D.splitDate, x1: D.dates[N - 1], y0: 0, y1: 1, fillcolor: '#2a78d6', opacity: 0.05, line: {width: 0}}],
    annotations: [{x: D.splitDate, y: 1, xref: 'x', yref: 'paper', text: 'test period →', showarrow: false, xanchor: 'left', yanchor: 'top', font: {size: 11, color: '#2a78d6'}}]};
  Plotly.react('priceChart', tr, lay, {displaylogo: false, responsive: true});
}

// ---------- heatmap ----------
function binOf(edges, v) {
  if (v === null) return -1;
  const nb = edges.length - 1;
  for (let b = 0; b < nb; b++) if (v < edges[b + 1] || b === nb - 1) return v >= edges[b] || b === 0 ? b : -1;
  return -1;
}
function heat() {
  const ctx = $('heatCtx').value, h = H, cap = D.heatCap;
  $('heatTitle').textContent = `Where the forward returns are: average ${h}-day forward return by indicator value` +
    (ctx === 'rule' ? ` (${side} rule context)` : '');
  const maxB = Math.max(...D.meta.filter(m => m.kind === 'num').map(m => m.edges.length - 1));
  let out = '';
  D.tfs.forEach(t => {
    out += `<tr class="tfh"><td colspan="${maxB + 1}">${t.name}</td></tr>`;
    D.numNames.forEach(n => {
      const k = `${t.tf}_${n.name}`, m = MK[k], nb = m.edges.length - 1;
      const base = ctx === 'rule' ? mask(side, k, true) : null;
      const vals = Array.from({length: nb}, () => []), cnt = new Array(nb).fill(0);
      for (let i = 0; i < N; i++) {
        if (!inPeriod(i) || (base && !base[i])) continue;
        const b = binOf(m.edges, F[k][i]); if (b < 0) continue;
        cnt[b]++; const f = D.fwd[h][i]; if (f !== null) vals[b].push(f);
      }
      const e = state[side][k] || {};
      out += `<tr><td>${n.label}</td>` + Array.from({length: maxB}, (_, b) => {
        if (b >= nb) return '<td></td>';
        const lo = m.edges[b], hi = m.edges[b + 1], av = mean(vals[b]);
        const win = vals[b].length ? (vals[b].filter(x => x > 0).length / vals[b].length * 100).toFixed(0) + '%' : 'n/a';
        const sel = (e.min != null || e.max != null) && (e.min == null || e.min <= lo) && (e.max == null || e.max >= hi) ? ' sel' : '';
        return `<td class="c${cnt[b] < 20 ? ' thin' : ''}${sel}" data-k="${k}" data-lo="${lo}" data-hi="${hi}" data-b="${b}" data-nb="${nb}" ` +
          `style="background:${color(av, cap)};color:${txtColor(av, cap)}" title="${t.name} ${n.label} ${fmtNum(lo)} to ${fmtNum(hi)}\n${cnt[b]} days · avg ${pct(av)} · ${win} positive">` +
          `${av === null ? '·' : pct(av, 1)}<span class="rng">${fmtNum(lo)}–${fmtNum(hi)}</span></td>`;
      }).join('') + '</tr>';
    });
  });
  $('heat').innerHTML = out;
  $('heat').querySelectorAll('td.c').forEach(td => td.onclick = () => {
    const k = td.dataset.k, b = +td.dataset.b, nb = +td.dataset.nb;
    state[side][k] = {min: b === 0 ? null : Number(td.dataset.lo), max: b === nb - 1 ? null : Number(td.dataset.hi)};
    fillGrid(); render();
  });

  let ob = '<tr><th></th>' + D.tfs.map(t => `<th colspan="2" style="text-align:center">${t.name}</th>`).join('') + '</tr>' +
    '<tr><th></th>' + D.tfs.map(() => '<th style="text-align:center">yes</th><th style="text-align:center">no</th>').join('') + '</tr>';
  D.boolNames.forEach(n => {
    ob += `<tr><td>${n.label}</td>` + D.tfs.map(t => {
      const k = `${t.tf}_${n.name}`, base = ctx === 'rule' ? mask(side, k, true) : null;
      return [1, 0].map(val => {
        const v = []; let c = 0;
        for (let i = 0; i < N; i++) { if (!inPeriod(i) || (base && !base[i]) || F[k][i] !== val) continue; c++; const f = D.fwd[h][i]; if (f !== null) v.push(f); }
        const av = mean(v), e = state[side][k] || {}, sel = e.sel === val ? ' sel' : '';
        return `<td class="c${c < 20 ? ' thin' : ''}${sel}" data-k="${k}" data-v="${val}" style="background:${color(av, cap)};color:${txtColor(av, cap)}" ` +
          `title="${t.name} ${n.label}: ${val ? 'yes' : 'no'}\n${c} days · avg ${pct(av)}">${av === null ? '·' : pct(av, 1)}<span class="rng">${c} days</span></td>`;
      }).join('');
    }).join('') + '</tr>';
  });
  $('heatBool').innerHTML = ob;
  $('heatBool').querySelectorAll('td.c').forEach(td => td.onclick = () => {
    state[side][td.dataset.k] = {sel: Number(td.dataset.v)}; fillGrid(); render();
  });
}

// ---------- suggestions ----------
function sugBlock(s) {
  const g = D.suggest[s], row = (lab, st) => `<tr><td>${lab}</td><td class="num">${st.n}</td><td class="num">${st.ep}</td>` +
    `<td class="num"><span class="sw" style="background:${color(st.avg, D.heatCap)}"></span>${pct(st.avg)}</td><td class="num">${st.win === null ? 'n/a' : st.win.toFixed(0) + '%'}</td></tr>`;
  if (!g.rule.length) return '<div class="note">No rule met the minimum signal-day and streak requirements.</div>';
  const bTr = baselineRange(0, D.split), bTe = baselineRange(D.split, N);
  return `<div class="rtext"><b>${g.text}</b></div><table><tr><th></th><th class="num">Days</th><th class="num">Streaks</th>` +
    `<th class="num">Avg ${D.target}D</th><th class="num">% positive</th></tr>` + row('Training', g.train) + row('Test (unseen)', g.test) +
    `<tr><td class="hint">All days, training</td><td></td><td></td><td class="num">${pct(bTr)}</td><td></td></tr>` +
    `<tr><td class="hint">All days, test</td><td></td><td></td><td class="num">${pct(bTe)}</td><td></td></tr></table>` +
    `<div style="margin-top:8px"><button data-load="${s}">Load into ${s} rule</button></div>`;
}
function baselineRange(a, b) { const v = []; for (let i = a; i < b; i++) { const f = D.fwd[String(D.target)][i]; if (f !== null) v.push(f); } return mean(v); }
function singTable(s) {
  const rows = D.suggest[s].singles;
  $(s === 'buy' ? 'singBuy' : 'singSell').innerHTML = `<tr><th>Condition</th><th class="num">Train days</th><th class="num">Train avg</th>` +
    `<th class="num">Test days</th><th class="num">Test avg</th><th class="num">Test % pos</th></tr>` +
    (rows.length ? rows.map((r, j) => `<tr class="click" data-s="${s}" data-j="${j}"><td>${r.text}</td><td class="num">${r.train.n}</td>` +
      `<td class="num">${pct(r.train.avg)}</td><td class="num">${r.test.n}</td>` +
      `<td class="num"><span class="sw" style="background:${color(r.test.avg, D.heatCap)}"></span>${pct(r.test.avg)}</td>` +
      `<td class="num">${r.test.win === null ? 'n/a' : r.test.win.toFixed(0) + '%'}</td></tr>`).join('')
      : '<tr><td colspan="6" class="note">None met the minimums.</td></tr>');
}
function addCond(s, c) {
  const e = state[s][c.k] || (state[s][c.k] = {});
  if (c.op === '>=') e.min = c.v; else if (c.op === '<=') e.max = c.v; else e.sel = c.v;
}

// ---------- signal list ----------
function hits(st) {
  const rows = st.idx.slice(-40).reverse();
  $('hitsTitle').textContent = `Recent ${side} signal days (${st.n} in period, newest first, last 40 shown)`;
  const keys = ruleKeys(side);
  $('hits').innerHTML = `<tr><th>Date</th><th class="num">Close</th>` + D.fwdDays.map(n => `<th class="num">${n}D fwd</th>`).join('') +
    keys.map(k => `<th class="num">${MK[k].tf} ${MK[k].name}</th>`).join('') + '</tr>' +
    (rows.length ? rows.map(i => `<tr><td>${D.dates[i]}</td><td class="num">$${D.close[i].toFixed(2)}</td>` +
      D.fwdDays.map(n => { const v = D.fwd[String(n)][i];
        return `<td class="num" style="background:${color(v, D.cap)};color:${txtColor(v, D.cap)}">${v === null ? 'pending' : pct(v)}</td>`; }).join('') +
      keys.map(k => `<td class="num">${fmtVal(k, F[k][i])}</td>`).join('') + '</tr>').join('')
      : `<tr><td colspan="9" class="note">The ${side} rule didn't fire in this period.</td></tr>`);
}

// ---------- render ----------
function render() {
  const mb = mask('buy'), ms = mask('sell');
  const p = $('period');
  $('periodPill').textContent = p.options[p.selectedIndex].text;
  const sb = tiles('buy', mb), ss = tiles('sell', ms);
  today(mb, ms); chart(sb, ss); heat(); hits(side === 'buy' ? sb : ss); fillGrid();
  $('ruleText').textContent = `Editing the ${side} rule` + (ruleKeys(side).length ? '' : ' (empty)');
}
function postHeight() {
  if (window.parent !== window) window.parent.postMessage({type: 'scanner-height', h: document.documentElement.scrollHeight}, '*');
}

(function init() {
  $('horizon').innerHTML = D.fwdDays.map(n => `<option value="${n}" ${String(n) === H ? 'selected' : ''}>${n} days</option>`).join('');
  $('optMax').textContent = D.opt.maxConds; $('optDays').textContent = D.opt.minDays; $('optEp').textContent = D.opt.minEp;
  state.buy = condsToState(D.rules.buy.conds); state.sell = condsToState(D.rules.sell.conds);
  $('sugBuy').innerHTML = sugBlock('buy'); $('sugSell').innerHTML = sugBlock('sell');
  singTable('buy'); singTable('sell');
  document.querySelectorAll('[data-load]').forEach(b => b.onclick = () => {
    side = b.dataset.load; state[side] = condsToState(D.suggest[side].rule); drawGrid(); render();
    window.scrollTo({top: 0, behavior: 'smooth'}); });
  document.querySelectorAll('#singBuy tr.click, #singSell tr.click').forEach(tr => tr.onclick = () => {
    side = tr.dataset.s; addCond(side, D.suggest[side].singles[+tr.dataset.j].cond); drawGrid(); render(); });
  document.querySelectorAll('.tab').forEach(b => b.onclick = () => { side = b.dataset.side; drawGrid(); render(); });
  $('resetBtn').onclick = () => { state[side] = condsToState(D.rules[side].conds); fillGrid(); render(); };
  $('suggestBtn').onclick = () => { state[side] = condsToState(D.suggest[side].rule); fillGrid(); render(); };
  $('clearBtn').onclick = () => { state[side] = {}; fillGrid(); render(); };
  $('horizon').onchange = () => { H = $('horizon').value; render(); };
  ['period', 'heatCtx', 'logY'].forEach(id => $(id).addEventListener('change', render));
  drawGrid(); render();
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
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--alert", action="store_true", help="check the latest close and email buy/sell alerts (no report)")
    ap.add_argument("--force", action="store_true", help="with --alert: send even if the latest bar isn't from today")
    args = ap.parse_args()

    print(f"Downloading {TICKER} daily history and replaying the scanner...")
    win, train, test, split_date, suggest, rules = prepare()
    print(f"Window {win.index.min():%Y-%m-%d} to {win.index.max():%Y-%m-%d} ({len(win):,} days); "
          f"training before {split_date:%Y-%m-%d}, test after.")
    for side in ("buy", "sell"):
        g = suggest[side]
        tr, te = g["train"], g["test"]
        print(f"Suggested {side.upper()}: {g['text']}")
        if tr["avg"] is not None:
            te_avg = "n/a" if te["avg"] is None else f"{te['avg']:+.2f}%"
            print(f"   train {tr['n']} days avg {tr['avg']:+.2f}% | test {te['n']} days avg {te_avg}")

    if args.alert:
        run_alerts(win, suggest, rules, split_date, force=args.force)
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["Close", *[f"Fwd_{n}D%" for n in FORWARD_DAYS], "Buy_Signal", "Sell_Signal", "Period", *ALL_KEYS]
    win[cols].round(4).to_csv(CSV_OUT, index_label="Date")
    build_report(win, split_date, suggest, rules)
    st = latest_status(win, rules)
    for side in ("buy", "sell"):
        print(f"Latest close {win.index[-1]:%Y-%m-%d}: {side.upper()} rule "
              f"{'FIRES' if st[side]['fires'] else 'no signal'} ({rules[side]['source']}: {rules[side]['text']})")
    print(f"Report: {HTML_OUT}\nCSV:    {CSV_OUT}")


if __name__ == "__main__":
    main()
