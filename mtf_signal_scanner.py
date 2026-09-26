"""
Multi-Timeframe Buy/Sell Signal Scanner + Backtest
==================================================

Uses the same indicators as Ticker_Indicator_Graph_DWM.ipynb (MACD 12/26/9, RSI 14,
Stochastic 14/3/3, MFI 14, CMF 20) plus Relative Volume, Price vs 20/50 SMA,
Relative Strength vs SPY, RS Acceleration and Price Momentum (ROC) on the Daily,
Weekly and Monthly timeframes, and answers two questions for one ticker (default TSLA):

  BUY  - which indicator values were followed by the HIGHEST 10-day forward return?
  SELL - which indicator values were followed by the LOWEST 10-day forward return
         (the stock tended to fall after them, i.e. a good time to take profits)?

How the backtest is built
-------------------------
Every trading day in the window is replayed as if the scanner ran after that day's
close. Weekly and Monthly values are read the way the notebook shows them live:
the current week/month is a partial bar (week-to-date / month-to-date), so no
future data leaks in. Forward return = close N trading days later vs. that close.

Added indicators (every one is computed on D, W and M, is a filter, and is used by
the optimizer). Lookbacks are in bars of that timeframe (20 bars = 20 days on
Daily, 20 weeks on Weekly, 20 months on Monthly), like the notebook's RSI 14 etc.:
  RVOL        relative volume = average daily volume inside the bar / average daily
              volume of the previous RVOL_PERIOD bars. 1.0 = normal, 2.0 = double.
              (Week/month-to-date bars are pace-adjusted, so Monday isn't "low".)
  SMA5_Pct, SMA10_Pct, SMA20_Pct, SMA50_Pct
              % the close is above (+) / below (-) its 5/10/20/50-bar simple moving
              average, one filter each; plus yes/no "close above" for each SMA and
              "stacked bullish" (5 > 10 > 20 > 50) / "stacked bearish" (5 < 10 < 20 < 50)
  RS_SPY      relative strength vs BENCHMARK: % change of the ratio close / SPY close
              over RS_PERIOD bars. +5 = beat SPY by ~5% over that stretch.
  RS_Accel    RS acceleration = RS_SPY now minus RS_SPY RS_ACCEL_PERIOD bars ago.
              Positive = relative strength is improving.
  ROC         price momentum, rate of change = % change of the close over ROC_PERIOD bars

Hover over any buy/sell marker on the price chart to see every indicator's Daily,
Weekly and Monthly value on that day (also shown in the panel under the chart).

Two ways to find good values:
  1. Interactive report (HTML): every indicator on every timeframe is a filter for
     a BUY rule and a separate SELL rule. A heatmap shows the average forward
     return for each value range of each indicator (green = price rose after,
     red = price fell after) - click a cell to use that range in the rule.
  2. Suggested rules: the script searches thresholds on the older part of the
     history (training period) and reports how the picked rule did on the most
     recent part it never saw (test period). Trust the test numbers, not the
     training numbers.

Two buy rules and two sell rules
--------------------------------
There are four rule slots: Buy 1, Buy 2, Sell 1, Sell 2, so you can run two
different strategies side by side. The report also pairs them up as round trips
(buy on a Buy signal, sell on the next Sell signal) and shows how long those
trades were held, bucketed into the forward-return horizons (5, 10, 20, 42, 63
trading days = 1 week, 2 weeks, ~1 month, ~2 months, ~3 months).

Saving rules
------------
The report autosaves your inputs in the browser and has a small library of named
rules. "Export rules" downloads mtf_signal_rules_<TICKER>.json. Put that file next
to this script and the scanner uses those rules for the report's starting filters
and for email alerts (it wins over the BUY_RULE / SELL_RULE settings below).

Alerts
------
  python mtf_signal_scanner.py                 -> builds the report + CSV (no email)
  python mtf_signal_scanner.py --ticker NVDA   -> same for another ticker
  python mtf_signal_scanner.py --alert         -> checks today's bar and emails an
                                                  alert when any rule fires
Email uses EMAIL_USER / EMAIL_PASS / ALERT_TO (same secrets as the ATH scanner).

Suggested rules are recomputed on every run from the latest OPTIMIZER_YEARS (5)
years of data, so they adapt to whichever ticker you run.

Outputs (written next to this script unless BACKTEST_OUTPUT_DIR is set):
  mtf_signal_report.html - open in any browser
  mtf_signal_days.csv    - every day with all indicators, forward returns, and
                           Buy_Signal / Buy2_Signal / Sell_Signal / Sell2_Signal flags

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
#              RVOL, SMA5_Pct, SMA10_Pct, SMA20_Pct, SMA50_Pct, RS_SPY, RS_Accel, ROC,
#              MACD_Bull, MACD_Pos, Hist_Rising, Stoch_Bull, Flow_Bull, Candle_Up,
#              Above_SMA5, Above_SMA10, Above_SMA20, Above_SMA50, SMA_Stack_Bull, SMA_Stack_Bear
#   e.g. BUY_RULE = {"D_RVOL": (1.5, None), "W_RS_SPY": (0, None), "D_Above_SMA50": True}
BUY_RULE = "auto"
SELL_RULE = "auto"
# Second strategy. "auto" = the optimizer's alternative rule, built from different
# indicators than rule 1. None = slot left empty (never fires, no alerts).
BUY_RULE_2 = "auto"
SELL_RULE_2 = "auto"

# Rules saved from the report ("Export rules"). When this file exists it overrides
# the four settings above. {ticker} is replaced with the ticker symbol.
RULES_FILE = os.environ.get("SIGNAL_RULES_FILE", "mtf_signal_rules_{ticker}.json")

# "every" = email every day the rule is true; "new" = only the first day of a streak
ALERT_MODE = "every"

# ============================================================
# ADDED INDICATOR SETTINGS (lookbacks in bars of each timeframe)
# ============================================================

BENCHMARK = os.environ.get("SIGNAL_BENCHMARK", "SPY")   # relative strength is measured against this
RVOL_PERIOD = 20             # relative volume: compare with the average of the previous N bars
RS_PERIOD = 20               # RS vs SPY: change of the price ratio over N bars
RS_ACCEL_PERIOD = 5          # RS acceleration: RS now minus RS this many bars ago
ROC_PERIOD = 12              # price momentum: % change over N bars (12 = the classic ROC setting)
SMA_PERIODS = [5, 10, 20, 50] # "Price vs SMA": one % field + one yes/no field per SMA
                             # (field names follow the numbers: SMA5_Pct, Above_SMA5, ...)

# ============================================================
# BACKTEST / OPTIMIZER SETTINGS
# ============================================================

BACKTEST_YEARS = 10          # days in the report (None = all history after indicator warm-up)
DEFAULT_VIEW_YEARS = 5       # date range the report opens on (you can change it in the page)
OPTIMIZER_YEARS = 5          # suggested rules are searched on the latest N years only
# Forward returns in the CSV + report, in trading days (~21 per month):
#   5 = 1 week, 10 = 2 weeks, 20 = ~1 month, 42 = ~2 months, 63 = ~3 months
FORWARD_DAYS = [5, 10, 20, 42, 63]
HORIZON_NAMES = {5: "1 week", 10: "2 weeks", 20: "~1 month", 42: "~2 months", 63: "~3 months"}
TARGET_FWD_DAYS = 10         # horizon used by "auto" rules for alerts (report shows every horizon)
TRAIN_FRACTION = 0.70        # oldest 70% of the optimizer window to pick rules, newest 30% to test
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

SCANNER_ORDER = 20           # after the ATH backtest (10)

# Rule slots: id, label, direction (+1 buy = want high forward returns, -1 sell = low),
# settings variable, CSV column
SLOTS = [("buy1", "Buy 1", +1, "BUY_RULE", "Buy_Signal"),
         ("buy2", "Buy 2", +1, "BUY_RULE_2", "Buy2_Signal"),
         ("sell1", "Sell 1", -1, "SELL_RULE", "Sell_Signal"),
         ("sell2", "Sell 2", -1, "SELL_RULE_2", "Sell2_Signal")]
SLOT_LABEL = {s: lab for s, lab, *_ in SLOTS}
SLOT_COL = {s: col for s, *_, col in SLOTS}

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
    ("RVOL", f"Relative volume (x avg of prior {RVOL_PERIOD})", 0.1, 2),
    *[(f"SMA{p}_Pct", f"Price vs {p} SMA, %", 1, 2) for p in SMA_PERIODS],
    ("RS_SPY", f"RS vs {BENCHMARK}, {RS_PERIOD}-bar %", 1, 2),
    ("RS_Accel", f"RS acceleration ({RS_ACCEL_PERIOD}-bar change)", 1, 2),
    ("ROC", f"Price momentum, ROC {ROC_PERIOD} %", 1, 2),
]
BOOL_FIELDS = [
    ("MACD_Bull", "MACD above signal"),
    ("MACD_Pos", "MACD above 0"),
    ("Hist_Rising", "Histogram rising (green bar)"),
    ("Stoch_Bull", "%K above %D"),
    ("Flow_Bull", "MFI > 50 and CMF > 0"),
    ("Candle_Up", "Candle up (close > open)"),
    *[(f"Above_SMA{p}", f"Close above {p} SMA") for p in SMA_PERIODS],
    ("SMA_Stack_Bull", "SMAs stacked bullish (" + " > ".join(map(str, SMA_PERIODS)) + ")"),
    ("SMA_Stack_Bear", "SMAs stacked bearish (" + " < ".join(map(str, SMA_PERIODS)) + ")"),
]
# Short names for the chart tooltip
SHORT = {"RSI": "RSI 14", "StochK": "Stoch %K", "StochD": "Stoch %D", "MFI": "MFI 14", "CMF": "CMF 20",
         "MACD": "MACD $", "Signal": "MACD sig $", "Hist": "MACD hist $", "MACD_Pct": "MACD %px",
         "Hist_Pct": "Hist %px", "RVOL": "Rel volume", "RS_SPY": f"RS vs {BENCHMARK}"[:12], "RS_Accel": "RS accel",
         "ROC": f"ROC {ROC_PERIOD} %", "MACD_Bull": "MACD>sig", "MACD_Pos": "MACD>0",
         "Hist_Rising": "Hist rising", "Stoch_Bull": "%K>%D", "Flow_Bull": "Flow bull", "Candle_Up": "Candle up",
         "SMA_Stack_Bull": "SMA stack up", "SMA_Stack_Bear": "SMA stack dn",
         **{f"SMA{p}_Pct": f"vs SMA{p} %" for p in SMA_PERIODS}, **{f"Above_SMA{p}": f">SMA{p}" for p in SMA_PERIODS}}
# The optimizer skips raw-dollar MACD fields: $ values from a $20 stock and a $400 stock
# aren't comparable, so it uses the % of price versions instead.
OPT_NUM_FIELDS = ["RSI", "StochK", "StochD", "MFI", "CMF", "MACD_Pct", "Hist_Pct",
                  "RVOL", *[f"SMA{p}_Pct" for p in SMA_PERIODS], "RS_SPY", "RS_Accel", "ROC"]
# The backtest window starts once these are warmed up on all three timeframes. The added
# indicators can need more history (a 50-month SMA needs 4+ years), so they may be blank
# (n/a) early in the window instead of shortening it; a blank value never passes a filter.
CORE_FIELDS = ["RSI", "StochK", "StochD", "MFI", "CMF", "MACD", "Signal", "Hist", "MACD_Pct", "Hist_Pct"]

NUM_KEYS = [f"{tf}_{n}" for tf, _, _ in TIMEFRAMES for n, *_ in NUM_FIELDS]
BOOL_KEYS = [f"{tf}_{n}" for tf, _, _ in TIMEFRAMES for n, _ in BOOL_FIELDS]
CORE_KEYS = [f"{tf}_{n}" for tf, _, _ in TIMEFRAMES for n in CORE_FIELDS]
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


def added_indicators(df):
    """Relative volume, SMAs, RS vs benchmark, RS acceleration, ROC on complete bars.

    Needs Close, Volume, Bench (benchmark close) and NDays (trading days in the bar).
    """
    out = pd.DataFrame(index=df.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        adv = df["Volume"] / df["NDays"]                       # average daily volume inside each bar
        out["RVOL"] = (adv / adv.shift(1).rolling(RVOL_PERIOD).mean()).replace([np.inf, -np.inf], np.nan)
        for p in SMA_PERIODS:
            out[f"SMA{p}"] = df["Close"].rolling(p).mean()
        out["ROC"] = (df["Close"] / df["Close"].shift(ROC_PERIOD) - 1) * 100
        ratio = df["Close"] / df["Bench"]
        out["RS"] = (ratio / ratio.shift(RS_PERIOD) - 1) * 100
        out["RS_Accel"] = out["RS"] - out["RS"].shift(RS_ACCEL_PERIOD)
    return out


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
                                    Close=("Close", "last"), Volume=("Volume", "sum"),
                                    Bench=("Bench", "last"), NDays=("Close", "size"))
    ind = compute_indicators(bars)
    add = added_indicators(bars)

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
    ND = (grp.cumcount() + 1).values.astype(float)     # trading days so far in the current bar
    B = daily["Bench"].values.astype(float)
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

        # Relative volume, pace-adjusted: avg daily volume so far this bar vs the previous bars' average
        adv = bars["Volume"] / bars["NDays"]
        rvol_p = (V / ND) / prev(adv.rolling(RVOL_PERIOD).mean())
        rvol_p = np.where(np.isfinite(rvol_p), rvol_p, np.nan)

        # SMAs including today's partial close
        smas = {f"SMA{p}": (prev(bars["Close"].rolling(p - 1).sum()) + C) / p if p > 1 else C
                for p in SMA_PERIODS}

        # Momentum and relative strength vs benchmark
        roc_p = (C / prev(bars["Close"], ROC_PERIOD) - 1) * 100
        ratio = bars["Close"] / bars["Bench"]
        rs_p = (C / B / prev(ratio, RS_PERIOD) - 1) * 100
        rs_acc_p = rs_p - prev(add["RS"], RS_ACCEL_PERIOD)

    return pd.DataFrame({
        "Open": O, "Close": C, "MACD": macd, "Signal": sig, "RSI": rsi_p, "%K": k_p, "%D": d_stoch,
        "MFI": mfi_p, "CMF": cmf_p, "Hist_Prev": prev(ind["MACD"] - ind["Signal"]),
        "RVOL": rvol_p, **smas, "ROC": roc_p, "RS": rs_p, "RS_Accel": rs_acc_p,
    }, index=daily.index)


def timeframe_fields(tf: str, x: pd.DataFrame) -> pd.DataFrame:
    """Turn one timeframe's indicator columns into the scanner fields (prefix D_/W_/M_)."""
    hist = x["MACD"] - x["Signal"]
    hist_prev = x["Hist_Prev"] if "Hist_Prev" in x else hist.shift(1)
    known = x[["MACD", "Signal", "RSI", "%K", "%D", "MFI", "CMF"]].notna().all(axis=1)

    def flag(cond, ok=known):
        return cond.astype(float).where(ok)

    sma = {p: x[f"SMA{p}"] for p in SMA_PERIODS}
    sma_known = pd.concat(sma.values(), axis=1).notna().all(axis=1)
    per = sorted(SMA_PERIODS)
    stack_bull = pd.Series(True, index=x.index)
    stack_bear = pd.Series(True, index=x.index)
    for a, b in zip(per[:-1], per[1:]):
        stack_bull &= sma[a] > sma[b]
        stack_bear &= sma[a] < sma[b]

    out = pd.DataFrame({
        "RSI": x["RSI"], "StochK": x["%K"], "StochD": x["%D"], "MFI": x["MFI"], "CMF": x["CMF"],
        "MACD": x["MACD"], "Signal": x["Signal"], "Hist": hist,
        "MACD_Pct": x["MACD"] / x["Close"] * 100, "Hist_Pct": hist / x["Close"] * 100,
        "RVOL": x["RVOL"],
        **{f"SMA{p}_Pct": (x["Close"] / sma[p] - 1) * 100 for p in SMA_PERIODS},
        "RS_SPY": x["RS"], "RS_Accel": x["RS_Accel"], "ROC": x["ROC"],
        # the notebook colors a histogram bar green when it is >= the previous bar
        "MACD_Bull": flag(x["MACD"] > x["Signal"]),
        "MACD_Pos": flag(x["MACD"] > 0),
        "Hist_Rising": flag(hist_prev.isna() | (hist >= hist_prev)),
        "Stoch_Bull": flag(x["%K"] > x["%D"]),
        "Flow_Bull": flag((x["MFI"] > 50) & (x["CMF"] > 0)),
        "Candle_Up": flag(x["Close"] > x["Open"]),
        **{f"Above_SMA{p}": flag(x["Close"] > sma[p], sma[p].notna()) for p in SMA_PERIODS},
        "SMA_Stack_Bull": flag(stack_bull, sma_known),
        "SMA_Stack_Bear": flag(stack_bear, sma_known),
    }, index=x.index)
    return out.add_prefix(f"{tf}_")


def build_history(daily: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for tf, _, freq in TIMEFRAMES:
        if tf == "D":
            x = compute_indicators(daily)
            x = x.join(added_indicators(daily.assign(NDays=1.0)))
        else:
            x = partial_bar_indicators(daily, freq)
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
            if v is None:
                continue
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
    if name in ("CMF", "RVOL"):
        return float(round(v, 2))
    if name in ("RS_SPY", "RS_Accel", "ROC") or name.startswith("SMA"):
        return float(round(v, 1))
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


def optimize(train, test, direction, horizon, exclude=(), cands=None, with_singles=True):
    """direction +1 = buy (maximize forward return), -1 = sell (minimize it).

    Rules are picked on the training days only; the test days are only scored.
    The last `horizon` training days are dropped from scoring because their forward
    window reaches into the test period. `exclude` = fields the rule may not use
    (used to build a second, different strategy).
    """
    tgt = f"Fwd_{horizon}D%"
    y_tr, y_te = train[tgt].values.copy(), test[tgt].values
    y_tr[max(0, len(y_tr) - horizon):] = np.nan
    cands = [c for c in (cands or candidates(train)) if c["k"] not in exclude]
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
    single_rows = []
    if with_singles:
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


def load_benchmark(index):
    """Benchmark closes on the ticker's trading days (NaN if it can't be downloaded)."""
    if BENCHMARK.upper() == TICKER.upper():
        return load_daily(TICKER)["Close"].reindex(index)
    try:
        b = load_daily(BENCHMARK)["Close"]
        return b.reindex(b.index.union(index)).ffill().reindex(index)
    except (SystemExit, Exception) as e:
        print(f"Warning: couldn't download {BENCHMARK} ({e}); RS fields will be blank.")
        return pd.Series(np.nan, index=index)


def prepare():
    daily = load_daily(TICKER)
    daily["Bench"] = load_benchmark(daily.index)
    hist = build_history(daily)
    valid = hist[CORE_KEYS].notna().all(axis=1)
    if not valid.any():
        raise SystemExit(f"{TICKER}: not enough history to warm up the monthly indicators.")
    end = hist.index.max()
    start = hist.index[valid.values.argmax()]
    if BACKTEST_YEARS:
        start = max(start, end - pd.DateOffset(years=BACKTEST_YEARS))
    win = hist[(hist.index >= start) & valid].copy()

    # Optimizer: latest OPTIMIZER_YEARS only, oldest TRAIN_FRACTION to pick, rest to test
    opt_start = win.index.min()
    if OPTIMIZER_YEARS:
        opt_start = max(opt_start, end - pd.DateOffset(years=OPTIMIZER_YEARS))
    opt = win[win.index >= opt_start]
    split_date = opt.index[int(len(opt) * TRAIN_FRACTION)]
    train, test = opt[opt.index < split_date], opt[opt.index >= split_date]

    # Suggestions for every horizon: rule 1 = best, rule 2 = best using other indicators
    cands = candidates(train)
    suggest = {}
    for h in FORWARD_DAYS:
        suggest[h] = {}
        for side, direction in (("buy", +1), ("sell", -1)):
            first = optimize(train, test, direction, h, cands=cands)
            second = optimize(train, test, direction, h, exclude={c["k"] for c in first["rule"]},
                              cands=cands, with_singles=False)
            suggest[h][f"{side}1"], suggest[h][f"{side}2"] = first, second

    saved, saved_path = load_rules_file()
    rules = {}
    for slot, lab, _, var, col in SLOTS:
        if saved is not None and slot in saved:
            spec, source = saved[slot], "saved"
        else:
            spec, source = globals()[var], "settings"
        if spec is None:
            conds = []
        elif isinstance(spec, str) and spec.lower() == "auto":
            conds, source = suggest[TARGET_FWD_DAYS][slot]["rule"], "auto"
        else:
            conds = parse_rule(spec, f"{lab} rule ({saved_path.name if source == 'saved' else var})")
        rules[slot] = {"conds": conds, "source": source, "text": rule_text(conds)}
        win[col] = rule_mask(win, conds)
    win["Period"] = np.select([win.index < opt_start, win.index < split_date], ["before_optimizer", "train"], "test")
    return win, opt_start, split_date, suggest, rules


def rules_path():
    p = Path(RULES_FILE.replace("{ticker}", TICKER))
    if not p.is_absolute():
        try:
            p = Path(__file__).resolve().parent / p
        except NameError:
            p = Path.cwd() / p
    return p


def load_rules_file():
    """Rules exported from the report (dict of slot -> {field: [min, max] | true/false})."""
    p = rules_path()
    if not p.exists():
        return None, p
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("ticker") and data["ticker"].upper() != TICKER.upper():
        print(f"Note: {p.name} was saved from the {data['ticker']} report; using it for {TICKER} anyway.")
    rules = {k: v for k, v in (data.get("rules") or {}).items() if k in SLOT_LABEL}
    print(f"Using saved rules from {p.name}: {', '.join(SLOT_LABEL[k] for k in rules) or 'none'}")
    return rules, p


# ============================================================
# ALERTS
# ============================================================

def latest_status(win, rules):
    last = win.iloc[-1]
    out = {}
    for slot, *_ in SLOTS:
        conds = rules[slot]["conds"]
        mask = win[SLOT_COL[slot]].values
        checks = [(c, float(last[c["k"]]), bool(cond_mask(win.iloc[[-1]], c)[0])) for c in conds]
        out[slot] = {"fires": bool(mask[-1]), "streak": streak(mask), "checks": checks}
    return out


def fmt_val(k, v):
    if v is None or pd.isna(v):
        return "n/a"
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


SOURCE_TEXT = {"auto": "suggested by the optimizer", "settings": "from your settings", "saved": "saved from the report"}


def run_alerts(win, suggest, rules, split_date, force=False):
    day = win.index[-1]
    today_et = datetime.now(ZoneInfo(MARKET_TZ)).date()
    if not force and (today_et - day.date()).days > 1:
        print(f"Latest bar is {day:%Y-%m-%d} (market closed today?) - no alert. Use --force to send anyway.")
        return
    status = latest_status(win, rules)
    last = win.iloc[-1]
    fired = []
    for slot, lab, *_ in SLOTS:
        st = status[slot]
        if not rules[slot]["conds"]:
            print(f"{lab.upper()}: no rule conditions - skipped.")
            continue
        print(f"{lab.upper()} rule ({rules[slot]['source']}): {rules[slot]['text']} -> "
              f"{'FIRES' if st['fires'] else 'no signal'}"
              + (f" (day {st['streak']} in a row)" if st["fires"] else ""))
        if st["fires"] and (ALERT_MODE == "every" or st["streak"] == 1):
            fired.append(slot)
    if not fired:
        print("No alerts today.")
        return

    lines = [f"{TICKER} close {day:%a %b %d, %Y}: ${last['Close']:,.2f}", ""]
    for slot in fired:
        st, r = status[slot], rules[slot]
        te = suggest[TARGET_FWD_DAYS][slot]["test"] if r["source"] == "auto" else stats(
            win[SLOT_COL[slot]].values[win.index >= split_date],
            win[f"Fwd_{TARGET_FWD_DAYS}D%"].values[win.index >= split_date])
        lines += [f"=== {SLOT_LABEL[slot].upper()} ALERT (day {st['streak']} in a row) ===",
                  f"Rule ({SOURCE_TEXT[r['source']]}):"]
        lines += [f"  - {cond_text(c)}   (today: {fmt_val(c['k'], v)})" for c, v, _ in st["checks"]]
        if te["avg"] is not None:
            lines.append(f"Backtest, test period since {split_date:%b %Y}: fired on {te['n']} days, "
                         f"avg {TARGET_FWD_DAYS}-day forward return {te['avg']:+.2f}%, "
                         f"{te['win']:.0f}% of them positive.")
        lines.append("")
    lines += ["Today's readings (Daily / Weekly / Monthly):"]
    for name, lab, *_ in NUM_FIELDS:
        vals = " / ".join(fmt_val(f"{tf}_{name}", last[f"{tf}_{name}"]) for tf, _, _ in TIMEFRAMES)
        lines.append(f"  {lab:<40} {vals}")
    for name, lab in BOOL_FIELDS:
        vals = " / ".join(fmt_val(f"{tf}_{name}", last[f"{tf}_{name}"]) for tf, _, _ in TIMEFRAMES)
        lines.append(f"  {lab:<40} {vals}")
    if os.environ.get("DASHBOARD_URL"):
        lines += ["", f"Dashboard: {os.environ['DASHBOARD_URL']}"]
    lines += ["", "Automated scanner alert based on historical indicator behavior. Not financial advice."]
    body = "\n".join(lines)
    subject = f"{TICKER} {' + '.join(SLOT_LABEL[s].upper() for s in fired)} alert - {day:%b %d, %Y} close ${last['Close']:,.2f}"
    print("\n" + subject + "\n" + body)
    send_email(subject, body)


# ============================================================
# REPORT
# ============================================================

def _col(series, nd):
    return [None if pd.isna(v) else round(float(v), nd) for v in series]




def build_report(win, opt_start, split_date, suggest, rules):
    meta = []
    for tf, tfname, _ in TIMEFRAMES:
        for name, lab, step, nd in NUM_FIELDS:
            meta.append({"k": f"{tf}_{name}", "tf": tf, "tfName": tfname, "name": name, "label": lab,
                         "kind": "num", "step": step, "nd": nd})
        for name, lab in BOOL_FIELDS:
            meta.append({"k": f"{tf}_{name}", "tf": tf, "tfName": tfname, "name": name, "label": lab,
                         "kind": "bool"})

    feats = {}
    for k in ALL_KEYS:
        feats[k] = ([None if pd.isna(v) else int(v) for v in win[k]] if k in BOOL_KEYS
                    else _col(win[k], DECIMALS[k.split("_", 1)[1]]))

    def sug(g):
        return {"rule": g["rule"], "text": g["text"], "train": g["train"], "test": g["test"],
                "singles": [{"cond": r["cond"], "text": r["text"], "train": r["train"], "test": r["test"]}
                            for r in g["singles"]]}

    data = {
        "ticker": TICKER,
        "dates": win.index.strftime("%Y-%m-%d").tolist(),
        "close": _col(win["Close"], 2),
        "feats": feats,
        "fwd": {str(n): _col(win[f"Fwd_{n}D%"], 2) for n in FORWARD_DAYS},
        "fwdDays": FORWARD_DAYS,
        "horizonNames": {str(n): HORIZON_NAMES.get(n, f"{n} days") for n in FORWARD_DAYS},
        "target": TARGET_FWD_DAYS,
        "optStart": int((win.index < opt_start).sum()),
        "optStartDate": f"{opt_start:%Y-%m-%d}",
        "split": int((win.index < split_date).sum()),
        "splitDate": f"{split_date:%Y-%m-%d}",
        "viewYears": DEFAULT_VIEW_YEARS,
        "optYears": OPTIMIZER_YEARS,
        "meta": meta,
        "tfs": [{"tf": tf, "name": name} for tf, name, _ in TIMEFRAMES],
        "numNames": [{"name": n, "label": lab, "step": s, "short": SHORT.get(n, n)} for n, lab, s, _ in NUM_FIELDS],
        "boolNames": [{"name": n, "label": lab, "short": SHORT.get(n, n)} for n, lab in BOOL_FIELDS],
        "bench": BENCHMARK,
        "slots": [{"id": s, "label": lab, "dir": d, "var": var} for s, lab, d, var, _ in SLOTS],
        "rules": {s: {"conds": rules[s]["conds"], "source": rules[s]["source"]} for s in rules},
        "suggest": {str(h): {s: sug(g) for s, g in by_slot.items()} for h, by_slot in suggest.items()},
        "opt": {"minDays": MIN_SIGNAL_DAYS, "minEp": MIN_EPISODES, "maxConds": MAX_RULE_CONDITIONS,
                "trainPct": round(TRAIN_FRACTION * 100)},
        "cap": COLOR_CAP_PCT, "heatCap": HEAT_CAP_PCT, "gradient": GRADIENT, "pending": PENDING_COLOR,
    }

    start, end = win.index.min(), win.index.max()
    plotly_tag = (f"<script>{get_plotlyjs()}</script>" if PLOTLY_JS == "inline" else
                  f'<script src="https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"></script>')
    html = (PAGE_TEMPLATE
            .replace("__TICKER__", TICKER)
            .replace("__PERIOD__", f"{start:%b %d, %Y} → {end:%b %d, %Y}")
            .replace("__CSV__", CSV_OUT.name)
            .replace("__SURFACE__", SURFACE).replace("__INK2__", INK_2)
            .replace("__INK__", INK).replace("__GRID__", GRID))
    html = html.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    html = html.replace("__PLOTLY__", plotly_tag)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HTML_OUT.write_text(html, encoding="utf-8")

    if os.environ.get("BACKTEST_OUTPUT_DIR"):
        st = latest_status(win, rules)
        now = " + ".join(SLOT_LABEL[s].upper() for s, *_ in SLOTS if st[s]["fires"]) or "no signal"
        counts = ", ".join(f"{SLOT_LABEL[s]} {int(win[SLOT_COL[s]].sum())}" for s, *_ in SLOTS if rules[s]["conds"])
        (OUT_DIR / "scanner.json").write_text(json.dumps({
            "title": f"{TICKER} Multi-Timeframe Buy/Sell Signals - Backtest",
            "order": SCANNER_ORDER,
            "page": HTML_OUT.name,
            "subtitle": f"{start:%b %d, %Y} to {end:%b %d, %Y} · signal days: {counts or 'no rules'} · latest close: {now}",
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
  .ctl { display:flex; align-items:center; gap:6px; font-size:13px; color:__INK2__; flex-wrap:wrap; max-width:100%; }
  select, input { font:inherit; font-size:13px; padding:5px 6px; border:1px solid #c9c8c2; border-radius:6px; background:#fff; color:__INK__; }
  select { max-width:100%; }
  input.n { width:62px; }
  input[type=checkbox] { padding:0; }
  button { font:inherit; font-size:13px; padding:6px 11px; border:1px solid #c9c8c2; border-radius:6px; background:#fff; color:__INK__; cursor:pointer; }
  button:hover { background:#f0efec; }
  button.sm { font-size:12px; padding:3px 8px; }
  .tabs { display:flex; gap:0; flex-wrap:wrap; }
  .tab { border-radius:0; padding:8px 16px; font-weight:600; border-left-width:0; }
  .tab:first-child { border-radius:8px 0 0 8px; border-left-width:1px; } .tab:last-child { border-radius:0 8px 8px 0; }
  .tab.on { color:#fff; }
  .savebar { background:#f7f6f2; border-radius:8px; padding:8px 10px; margin:8px 0; }
  .savebar .lbl { font-size:12px; font-weight:600; color:__INK2__; text-transform:uppercase; letter-spacing:.04em; }
  .ok-msg { font-size:12px; color:#0b5a24; }
  .today { display:grid; grid-template-columns:repeat(auto-fit, minmax(240px, 1fr)); gap:12px; margin-bottom:16px; }
  .badge { border-radius:10px; padding:12px 16px; border:1px solid __GRID__; background:#fff; }
  .badge .t { font-size:12px; color:__INK2__; text-transform:uppercase; letter-spacing:.04em; }
  .badge .v { font-size:20px; font-weight:700; margin:2px 0; }
  .badge .r { font-size:12px; color:__INK2__; line-height:1.45; }
  .badge.fire.buy { background:#e8f5ec; border-color:#4fae68; } .badge.fire.buy .v { color:#0b5a24; }
  .badge.fire.sell { background:#fbeaea; border-color:#e0584e; } .badge.fire.sell .v { color:#8e1b1b; }
  table { border-collapse:collapse; width:100%; font-size:13px; }
  th, td { text-align:left; padding:5px 7px; border-bottom:1px solid __GRID__; white-space:nowrap; }
  th { color:__INK2__; font-weight:500; }
  .num { text-align:right; font-variant-numeric:tabular-nums; }
  .wrapcell { white-space:normal; min-width:220px; max-width:420px; }
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
  .chip { display:inline-block; font-size:11px; font-weight:700; padding:1px 7px; border-radius:10px; color:#fff; margin-right:3px; }
  table.heat td.c { text-align:center; cursor:pointer; min-width:54px; font-size:12px; border:2px solid #fff; border-radius:4px; }
  table.heat td.c:hover { outline:2px solid __INK__; }
  table.heat td.c.thin { opacity:.45; }
  table.heat td.c.sel { outline:2px solid #2a78d6; }
  table.heat tr.tfh td { font-weight:700; background:#f7f6f2; }
  table.heat .rng { display:block; font-size:10px; opacity:.8; }
  .sw { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; vertical-align:-1px; border:1px solid rgba(0,0,0,.08); }
  tr.click { cursor:pointer; } tr.click:hover td { background:#f5f4f0; }
  tr.pick td { background:#eef4fb; }
  .two { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
  .two > div { min-width:0; overflow-x:auto; }
  .hint { color:__INK2__; font-size:12px; }
  @media (max-width: 900px) { .two { grid-template-columns:1fr; } }
  .pill { display:inline-block; font-size:11px; padding:1px 7px; border-radius:10px; background:#f0efec; color:__INK2__; margin-left:6px; font-weight:500; }
  .bk { display:inline-block; min-width:62px; text-align:center; border-radius:4px; padding:2px 4px; }
  .callout { border-left:3px solid #2a78d6; background:#f3f7fc; padding:8px 12px; border-radius:4px; font-size:13px; margin:8px 0; line-height:1.5; }
  td.b { font-weight:700; }
  table.det { width:auto; min-width:420px; } table.det td, table.det th { padding:3px 10px; }
  table.det tr.sep td { background:#f7f6f2; font-weight:600; font-size:12px; color:__INK2__; }
  table.det tr.used td { background:#eef4fb; font-weight:600; }
  a { color:#2a78d6; }
</style></head><body><div class="wrap">
<h1>__TICKER__ multi-timeframe buy &amp; sell signals: backtest</h1>
<div class="sub">Data __PERIOD__ · every trading day replayed after the close with the notebook's indicators on the Daily,
Weekly (week-to-date) and Monthly (month-to-date) timeframes. Two <b>buy rules</b> (conditions followed by the highest forward
returns) and two <b>sell rules</b> (followed by the lowest). Colors show the forward return after a day:
<span class="sw" style="background:#0b5a24"></span>green = price rose afterwards,
<span class="sw" style="background:#8e1b1b"></span>red = price fell. Good buy signals are green, good sell signals are red.
Forward horizons are in trading days (about 21 per month). Everything below follows the date range you pick.</div>

<div class="today" id="today"></div>

<div class="card">
  <div class="row" style="justify-content:space-between; margin-bottom:8px">
    <div class="tabs" id="tabs"></div>
    <div class="row">
      <label class="ctl">Forward return <select id="horizon"></select></label>
    </div>
  </div>
  <div class="row" style="margin-bottom:8px">
    <label class="ctl">Date range <select id="period"></select></label>
    <label class="ctl">from <input type="date" id="from"></label>
    <label class="ctl">to <input type="date" id="to"></label>
    <span class="pill" id="rangePill"></span>
  </div>
  <div class="row" style="margin-bottom:4px">
    <button id="resetBtn" title="The rule the scanner uses for email alerts">Reset to scanner rule</button>
    <button id="suggestBtn">Load suggested rule</button>
    <button id="clearBtn">Clear rule</button>
    <span class="note" id="ruleText" style="margin:0"></span>
  </div>
  <div class="savebar">
    <div class="row">
      <span class="lbl">Save</span>
      <input id="saveName" placeholder="name this rule" style="width:180px">
      <button id="saveBtn">Save rule</button>
      <select id="libSel" style="min-width:200px"></select>
      <button id="libLoad">Load into this slot</button>
      <button id="libDel">Delete</button>
      <span style="flex:1"></span>
      <button id="exportBtn" title="Download all four rules + your saved library as JSON">Export rules</button>
      <button id="importBtn">Import rules</button><input type="file" id="importFile" accept=".json,application/json" style="display:none">
      <button id="copyPy" title="Copy the four rules as Python settings">Copy as Python</button>
    </div>
    <div class="note" id="saveNote" style="margin:6px 0 0"></div>
  </div>
  <div class="tscroll"><table class="grid" id="grid"></table></div>
  <div class="note">All filled-in conditions must pass (AND). Leave a box empty for no limit; an empty rule never fires. The small line
  under each box is the latest close's value (green = passes this rule's condition). Numbers are on the notebook's scales: RSI, %K, %D
  and MFI 0 to 100, CMF -1 to 1, MACD fields in dollars (the "% of price" versions compare better across years). Relative volume is a
  multiple (1 = normal, 2 = double the recent average). Price vs SMA, RS, RS acceleration and ROC are in %: +3 = 3% above the SMA / 3%
  ahead of the benchmark / up 3%. Lookbacks count bars of each timeframe. "n/a" = not enough history yet for that indicator.</div>
</div>

<div class="card"><h2>Rule results <span class="pill" id="resPill"></span></h2>
  <div class="tscroll"><table id="results"></table></div>
  <div class="note">"Edge" = the rule's average forward return minus the average of all days in the range. A buy rule wants a
  positive edge, a sell rule a negative one. Streaks = separate runs of consecutive signal days; consecutive days overlap in
  their forward windows, so streaks are the more honest count.</div>
</div>

<div class="card"><h2>Price with buy and sell signals</h2>
  <div class="row">
    <label class="ctl"><input type="checkbox" id="logY" checked> log price scale</label>
    <label class="ctl"><input type="checkbox" id="showTrips"> draw round trips for</label>
    <select id="chartPair"></select>
  </div>
  <div id="priceChart"></div>
  <div class="note">Marker shape = rule (▲ Buy 1, ● Buy 2, ▼ Sell 1, ■ Sell 2), fill = forward return at the selected horizon.
  Markers are drawn a little below (buys) or above (sells) the close so signals on the same day don't hide each other; hover shows the
  actual close, forward returns and every indicator's Daily / Weekly / Monthly value that day (bold = used in that rule). Hovering or
  clicking a marker also fills the table below. Click a legend entry to hide a rule. Shaded area = the optimizer's test period.</div>
  <div class="row" style="margin:4px 0"><label class="ctl"><input type="checkbox" id="fullHover" checked> full indicator readout in the tooltip</label></div>
  <div id="sigDetail" class="tscroll"><div class="note">Hover over or click a buy/sell marker to see its full Daily / Weekly / Monthly indicator readout here.</div></div>
</div>

<div class="card"><h2>Round trips: how long from buy to sell <span class="pill" id="tripPill"></span></h2>
  <div class="note" style="margin-top:0">Each trade buys at the close of a buy signal day and sells at the close of the next sell signal day
  (later buy signals while holding are ignored). Hold time is in trading days, bucketed by the forward-return horizons, so you can see which
  horizon matches how long a strategy actually holds.</div>
  <div class="tscroll"><table id="trips"></table></div>
  <div id="tripCallout"></div>
  <div class="row" style="margin-top:6px"><label class="ctl">Trades for <select id="tripPair"></select></label></div>
  <div class="tscroll"><table id="tradeList"></table></div>
</div>

<div class="card"><h2 id="heatTitle">Where the forward returns are: average by indicator value</h2>
  <div class="row" style="margin-bottom:6px">
    <label class="ctl">Days included <select id="heatCtx">
      <option value="all">all days in the range</option>
      <option value="rule">days passing the active rule's OTHER conditions</option></select></label>
    <span class="note" style="margin:0">Each row splits that indicator's values in the date range into 10 equal-count ranges. Click a cell to
    set that range in the active rule (blue outline = its current range). Faded = under 20 days.</span>
  </div>
  <div class="tscroll"><table class="heat" id="heat"></table></div>
  <h2 style="margin-top:14px">Yes/no fields</h2>
  <div class="tscroll"><table class="heat" id="heatBool"></table></div>
</div>

<div class="card"><h2>Suggested rules <span class="pill" id="sugPill"></span></h2>
  <div class="note" id="sugNote"></div>
  <div class="two">
    <div><div id="sug_buy1"></div></div>
    <div><div id="sug_sell1"></div></div>
  </div>
  <div class="two" style="margin-top:12px">
    <div><div id="sug_buy2"></div></div>
    <div><div id="sug_sell2"></div></div>
  </div>
  <div class="two" style="margin-top:12px">
    <div><h2 style="font-size:14px;color:#0b5a24">Buy: best single conditions (click to add to the active buy rule)</h2><div class="tscroll"><table id="singBuy"></table></div></div>
    <div><h2 style="font-size:14px;color:#8e1b1b">Sell: best single conditions (click to add to the active sell rule)</h2><div class="tscroll"><table id="singSell"></table></div></div>
  </div>
</div>

<div class="card"><h2 id="hitsTitle">Signal days</h2>
  <div class="row" id="hitSlots" style="margin-bottom:6px"></div>
  <div class="tscroll"><table id="hits"></table></div>
  <div class="note">Every day in the range where a selected rule fired, newest first (up to 150 rows; tick "first day of each streak only" to see further back). Indicator columns cover every field used
  by the selected rules; bold = that field is part of a rule that fired that day. <a href="__CSV__" download>Download every day with all
  indicators (CSV)</a>. Backtest uses split-adjusted prices, ignores costs and taxes, and is not financial advice.</div>
</div>
</div>

<script>
const D = __DATA__;
const N = D.dates.length, F = D.feats;
const SURF = '__SURFACE__', INK = '__INK__', INK2 = '__INK2__', GRIDC = '__GRID__';
const $ = id => document.getElementById(id);
const MK = Object.fromEntries(D.meta.map(m => [m.k, m]));
const SLOTS = D.slots, SL = Object.fromEntries(SLOTS.map(s => [s.id, s]));
const STY = {
  buy1: {sym: 'triangle-up', col: '#0b5a24', off: 0.96, short: 'B1', mark: '▲'},
  buy2: {sym: 'circle', col: '#2a78d6', off: 0.92, short: 'B2', mark: '●'},
  sell1: {sym: 'triangle-down', col: '#8e1b1b', off: 1.04, short: 'S1', mark: '▼'},
  sell2: {sym: 'square', col: '#7b3fb0', off: 1.08, short: 'S2', mark: '■'},
};
const BUYS = SLOTS.filter(s => s.dir > 0).map(s => s.id), SELLS = SLOTS.filter(s => s.dir < 0).map(s => s.id);
let side = 'buy1', H = String(D.target), RA = 0, RB = N - 1;
const state = Object.fromEntries(SLOTS.map(s => [s.id, {}]));

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
// longer horizons have bigger moves, so the color scale widens with sqrt(time)
const capFor = (h, base) => Math.max(1, Math.round(base * Math.sqrt(Number(h) / 10)));
const txtColor = (v, cap) => (v === null || Math.abs(v) / cap < 0.45) ? INK : '#fff';
const pct = (v, d = 2) => v === null || v === undefined ? 'n/a' : (v > 0 ? '+' : '') + v.toFixed(d) + '%';
const mean = a => a.length ? a.reduce((s, v) => s + v, 0) / a.length : null;
function quant(a, q) { if (!a.length) return null; const s = [...a].sort((x, y) => x - y), p = (s.length - 1) * q, lo = Math.floor(p), hi = Math.ceil(p); return s[lo] + (s[hi] - s[lo]) * (p - lo); }
const median = a => quant(a, 0.5);
const fmtNum = v => v === null ? 'n/a' : (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));
const fmtVal = (k, v) => v === null ? 'n/a' : MK[k].kind === 'bool' ? (v ? 'yes' : 'no') : fmtNum(v);
const fmtDate = s => new Date(s + 'T00:00:00Z').toLocaleDateString('en-US', {month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC'});
const hName = h => `${h} days (${D.horizonNames[String(h)]})`;
const chip = s => `<span class="chip" style="background:${STY[s].col}">${STY[s].short}</span>`;
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));

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
function ruleKeys(s, exclude) { return Object.keys(state[s]).filter(k => k !== exclude && MK[k] && active(state[s][k])); }
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

// ---------- saving: python-style rule format {field: [min, max] | true/false} ----------
function toPy(s) {
  const o = {};
  ruleKeys(s).forEach(k => { const e = state[s][k];
    o[k] = e.sel != null ? e.sel === 1 : [e.min ?? null, e.max ?? null]; });
  return o;
}
function fromPy(o) {
  const s = {};
  Object.entries(o || {}).forEach(([k, v]) => {
    if (!MK[k] || v === null) return;
    if (typeof v === 'boolean') s[k] = {sel: v ? 1 : 0};
    else if (Array.isArray(v)) s[k] = {min: v[0] ?? null, max: v[1] ?? null};
  });
  return s;
}
const STORE = (() => { try { const t = '__mtf'; localStorage.setItem(t, '1'); localStorage.removeItem(t); return localStorage; } catch (e) { return null; } })();
const KEY_CUR = `mtfScanner:${D.ticker}:current`, KEY_LIB = 'mtfScanner:library';
function sget(k) { try { return STORE ? JSON.parse(STORE.getItem(k)) : null; } catch (e) { return null; } }
function sset(k, v) { try { if (!STORE) return false; STORE.setItem(k, JSON.stringify(v)); return true; } catch (e) { return false; } }
let library = sget(KEY_LIB) || [];
function autosave() {
  sset(KEY_CUR, {rules: Object.fromEntries(SLOTS.map(s => [s.id, toPy(s.id)])), H, period: $('period').value,
    from: $('from').value, to: $('to').value, side});
}
function note(msg) { $('saveNote').innerHTML = msg; }
function baseNote() {
  note(STORE ? `Your inputs autosave in this browser. Saved rules (${library.length}) are shared across tickers. <b>Export rules</b> downloads
    <code>mtf_signal_rules_${esc(D.ticker)}.json</code>: put it next to the script and the scanner uses those four rules for alerts.`
    : `This browser is blocking storage, so inputs won't be remembered here. Use <b>Export rules</b> to keep them in a file.`);
}
function drawLib() {
  $('libSel').innerHTML = library.length ? library.map((r, j) => `<option value="${j}">[${r.kind}] ${esc(r.name)}</option>`).join('')
    : '<option value="">no saved rules yet</option>';
}
function download(name, text) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], {type: 'application/json'}));
  a.download = name; document.body.appendChild(a); a.click(); a.remove();
}
function pyText() {
  const lit = v => v === null ? 'None' : v === true ? 'True' : v === false ? 'False' : String(v);
  return SLOTS.map(s => { const o = toPy(s.id), ks = Object.keys(o);
    if (!ks.length) return `${s.var} = None`;
    return `${s.var} = {\n` + ks.map(k => `    "${k}": ${Array.isArray(o[k]) ? `(${lit(o[k][0])}, ${lit(o[k][1])})` : lit(o[k])},`).join('\n') + '\n}';
  }).join('\n');
}

// ---------- date range ----------
function firstOnOrAfter(d) { let lo = 0, hi = N - 1; while (lo < hi) { const m = (lo + hi) >> 1; if (D.dates[m] < d) lo = m + 1; else hi = m; } return lo; }
function lastOnOrBefore(d) { let lo = 0, hi = N - 1; while (lo < hi) { const m = (lo + hi + 1) >> 1; if (D.dates[m] > d) hi = m - 1; else lo = m; } return lo; }
function yearsBack(y) { const d = new Date(D.dates[N - 1] + 'T00:00:00Z'); d.setUTCFullYear(d.getUTCFullYear() - y); return d.toISOString().slice(0, 10); }
function periodOptions() {
  const yrs = (new Date(D.dates[N - 1]) - new Date(D.dates[0])) / 3.156e10, o = [];
  [1, 3, 5, 10].forEach(y => { if (y < yrs - 0.05) o.push([`${y}y`, `last ${y} year${y > 1 ? 's' : ''}`]); });
  o.push(['all', `all data (since ${fmtDate(D.dates[0])})`]);
  o.push(['train', `optimizer training days (${fmtDate(D.optStartDate)} to ${fmtDate(D.dates[D.split - 1])})`]);
  o.push(['test', `optimizer test days (since ${fmtDate(D.splitDate)})`]);
  o.push(['custom', 'custom dates']);
  $('period').innerHTML = o.map(([v, t]) => `<option value="${v}">${t}</option>`).join('');
  const def = `${D.viewYears}y`;
  $('period').value = o.some(x => x[0] === def) ? def : 'all';
}
function applyPeriod() {
  const p = $('period').value;
  if (p === 'custom') {
    RA = firstOnOrAfter($('from').value || D.dates[0]); RB = lastOnOrBefore($('to').value || D.dates[N - 1]);
    if (RB < RA) [RA, RB] = [RB, RA];
  } else {
    RA = 0; RB = N - 1;
    if (p.endsWith('y')) RA = firstOnOrAfter(yearsBack(parseInt(p)));
    else if (p === 'train') { RA = D.optStart; RB = D.split - 1; }
    else if (p === 'test') RA = D.split;
  }
  $('from').value = D.dates[RA]; $('to').value = D.dates[RB];
  $('rangePill').textContent = `${fmtDate(D.dates[RA])} → ${fmtDate(D.dates[RB])} · ${(RB - RA + 1).toLocaleString()} trading days`;
}

// ---------- rule grid ----------
function drawTabs() {
  $('tabs').innerHTML = SLOTS.map(s => `<button class="tab${s.id === side ? ' on' : ''}" data-side="${s.id}" style="${s.id === side ?
    `background:${STY[s.id].col};border-color:${STY[s.id].col}` : `color:${STY[s.id].col}`}">${STY[s.id].mark} ${s.label}</button>`).join('');
  $('tabs').querySelectorAll('.tab').forEach(b => b.onclick = () => { side = b.dataset.side; drawTabs(); fillGrid(); render(); });
}
function drawGrid() {
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
    e[b] = v === '' ? null : Number(v);
    render();
  }));
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
  const src = {auto: 'optimizer suggestion', settings: 'script settings', saved: 'saved rules file'}[D.rules[side].source];
  $('ruleText').textContent = `Editing ${SL[side].label}` + (ruleKeys(side).length ? '' : ' (empty)') + ` · scanner's alert rule comes from the ${src}`;
}

// ---------- stats (inside the date range) ----------
function statsOf(m, h) {
  const idx = []; let ep = 0;
  for (let i = RA; i <= RB; i++) if (m[i]) { idx.push(i); if (i === RA || !m[i - 1]) ep++; }
  const v = idx.map(i => D.fwd[h][i]).filter(x => x !== null);
  return {idx, n: idx.length, ep, avg: mean(v), med: median(v), win: v.length ? v.filter(x => x > 0).length / v.length * 100 : null};
}
function baseline(h) { const v = []; for (let i = RA; i <= RB; i++) if (D.fwd[h][i] !== null) v.push(D.fwd[h][i]); return mean(v); }

function results(ST) {
  const b = baseline(H), hc = capFor(H, D.heatCap);
  $('resPill').textContent = `${hName(H)} forward · ${$('rangePill').textContent}`;
  $('results').innerHTML = `<tr><th>Rule</th><th>Conditions</th><th class="num">Signal days</th><th class="num">Streaks</th><th class="num">Avg ${H}D</th>` +
    `<th class="num">Median ${H}D</th><th class="num">% positive</th><th class="num">Edge vs all days</th></tr>` +
    SLOTS.map(s => { const st = ST[s.id], edge = st.avg === null || b === null ? null : st.avg - b;
      const good = edge === null ? '' : (edge * s.dir > 0 ? 'color:#0b5a24' : 'color:#8e1b1b');
      return `<tr><td>${chip(s.id)} ${s.label}</td><td class="wrapcell hint">${esc(ruleDesc(s.id))}</td><td class="num">${st.n.toLocaleString()}</td>` +
        `<td class="num">${st.ep}</td><td class="num"><span class="sw" style="background:${color(st.avg, hc)}"></span>${pct(st.avg)}</td>` +
        `<td class="num">${pct(st.med)}</td><td class="num">${st.win === null ? 'n/a' : st.win.toFixed(0) + '%'}</td>` +
        `<td class="num" style="${good};font-weight:600">${pct(edge)}</td></tr>`; }).join('') +
    `<tr><td class="hint">All days</td><td class="hint">every trading day in the range</td><td></td><td></td><td class="num">${pct(b)}</td><td></td><td></td><td></td></tr>`;
}

// ---------- today ----------
function today(M) {
  const d = new Date(D.dates[N - 1] + 'T00:00:00Z').toLocaleDateString('en-US', {weekday: 'short', month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC'});
  $('today').innerHTML = SLOTS.map(s => {
    const m = M[s.id], keys = ruleKeys(s.id), kind = s.dir > 0 ? 'buy' : 'sell';
    let streak = 0; for (let i = N - 1; i >= 0 && m[i]; i--) streak++;
    const fires = keys.length && m[N - 1];
    const detail = keys.length ? keys.map(k => `${MK[k].tfName} ${MK[k].label}: ${fmtVal(k, F[k][N - 1])} ${condOk(F[k][N - 1], state[s.id][k]) ? '✓' : '✗'}`).join(' · ') : 'no conditions set';
    return `<div class="badge ${kind} ${fires ? 'fire' : ''}"><div class="t">${chip(s.id)}${s.label} on ${d} close ($${D.close[N - 1].toFixed(2)})</div>` +
      `<div class="v">${fires ? s.label.toUpperCase() + ' signal' + (streak > 1 ? ` · day ${streak}` : '') : 'No signal'}</div><div class="r">${detail}</div></div>`;
  }).join('');
}

// ---------- round trips ----------
function pairs() {
  const out = [];
  BUYS.forEach(b => SELLS.forEach(s => out.push({id: `${b}>${s}`, b: [b], s: [s], label: `${SL[b].label} → ${SL[s].label}`})));
  out.push({id: 'any>any', b: BUYS, s: SELLS, label: 'Either buy → either sell'});
  return out;
}
const PAIRS = pairs();
function orMask(M, ids) { const m = new Array(N).fill(false); ids.forEach(id => { for (let i = 0; i < N; i++) if (M[id][i]) m[i] = true; }); return m; }
function tradesFor(M, p) {
  const bm = orMask(M, p.b), sm = orMask(M, p.s), out = [];
  const mk = (e, x, open) => ({e, x, open, hold: x - e, cal: Math.round((new Date(D.dates[x]) - new Date(D.dates[e])) / 864e5),
    ret: (D.close[x] / D.close[e] - 1) * 100});
  let e = -1;
  for (let i = RA; i <= RB; i++) {
    if (e < 0) { if (bm[i]) e = i; }
    else if (sm[i]) { out.push(mk(e, i, false)); e = -1; }
  }
  if (e >= 0) out.push(mk(e, RB, true));
  return out;
}
function buckets() {
  const b = []; let lo = 1;
  D.fwdDays.forEach(n => { b.push({lo, hi: n, label: lo === 1 ? `≤${n}` : `${lo}–${n}`, h: n}); lo = n + 1; });
  b.push({lo, hi: Infinity, label: `${lo}+`, h: null});
  return b;
}
const BK = buckets();
function trips(M) {
  const hasRule = id => ruleKeys(id).length > 0;
  $('tripPill').textContent = $('rangePill').textContent;
  let best = null;
  const rows = PAIRS.map(p => {
    const ok = p.b.some(hasRule) && p.s.some(hasRule);
    if (!ok) return `<tr><td>${p.label}</td><td colspan="${6 + BK.length}" class="hint">one of these rules is empty</td></tr>`;
    const T = tradesFor(M, p), C = T.filter(t => !t.open), open = T.find(t => t.open);
    const holds = C.map(t => t.hold), rets = C.map(t => t.ret);
    const cnt = BK.map(k => holds.filter(h => h >= k.lo && h <= k.hi).length), mx = Math.max(1, ...cnt);
    const med = median(holds);
    if (p.id === $('tripPair').value && med !== null) best = {p, med, q1: quant(holds, .25), q3: quant(holds, .75), n: C.length, avgRet: mean(rets)};
    return `<tr class="click${p.id === $('tripPair').value ? ' pick' : ''}" data-pair="${p.id}"><td>${p.label}</td>` +
      `<td class="num">${C.length}${open ? ' <span class="hint">+1 open</span>' : ''}</td>` +
      `<td class="num">${med === null ? 'n/a' : med.toFixed(0)}</td>` +
      `<td class="num">${holds.length ? `${quant(holds, .25).toFixed(0)}–${quant(holds, .75).toFixed(0)}` : 'n/a'}</td>` +
      `<td class="num">${C.length ? mean(C.map(t => t.cal)).toFixed(0) : 'n/a'}</td>` +
      `<td class="num"><span class="sw" style="background:${color(mean(rets), capFor(med || 10, D.heatCap))}"></span>${pct(mean(rets))}</td>` +
      `<td class="num">${rets.length ? (rets.filter(r => r > 0).length / rets.length * 100).toFixed(0) + '%' : 'n/a'}</td>` +
      cnt.map(c => `<td class="num"><span class="bk" style="background:rgba(42,120,214,${(c / mx * 0.35).toFixed(2)})">${c}${holds.length ? ` · ${(c / holds.length * 100).toFixed(0)}%` : ''}</span></td>`).join('') + '</tr>';
  });
  $('trips').innerHTML = `<tr><th>Pair</th><th class="num">Trades</th><th class="num">Median hold</th><th class="num">Middle 50%</th>` +
    `<th class="num">Avg cal. days</th><th class="num">Avg return</th><th class="num">% winners</th>` +
    BK.map(k => `<th class="num">${k.label} d</th>`).join('') + '</tr>' + rows.join('');
  $('trips').querySelectorAll('tr.click').forEach(tr => tr.onclick = () => { $('tripPair').value = tr.dataset.pair; $('chartPair').value = tr.dataset.pair; render(); });

  if (best) {
    const near = D.fwdDays.reduce((a, n) => Math.abs(n - best.med) < Math.abs(a - best.med) ? n : a, D.fwdDays[0]);
    $('tripCallout').innerHTML = `<div class="callout"><b>${best.p.label}</b>: median hold <b>${best.med.toFixed(0)} trading days</b>
      (middle half of trades ${best.q1.toFixed(0)} to ${best.q3.toFixed(0)} days) over ${best.n} closed trades, avg return ${pct(best.avgRet)}.
      The closest forward-return horizon is <b>${hName(near)}</b>${String(near) === H ? ' (already selected)' : ` <button class="sm" id="useH" data-h="${near}">use it</button>`}.
      Tuning the buy rule on that horizon makes its forward returns match how long this pair actually holds.</div>`;
    const u = $('useH'); if (u) u.onclick = () => { H = u.dataset.h; $('horizon').value = H; render(); };
  } else $('tripCallout').innerHTML = '';

  const p = PAIRS.find(x => x.id === $('tripPair').value);
  const T = p && p.b.some(hasRule) && p.s.some(hasRule) ? tradesFor(M, p) : [];
  $('tradeList').innerHTML = `<tr><th>Bought</th><th class="num">Buy close</th><th>Sold</th><th class="num">Sell close</th>` +
    `<th class="num">Hold (trading days)</th><th class="num">Calendar days</th><th class="num">Return</th></tr>` +
    (T.length ? T.slice(-30).reverse().map(t => { const cp = capFor(Math.max(t.hold, 5), D.cap);
      return `<tr><td>${D.dates[t.e]}</td><td class="num">$${D.close[t.e].toFixed(2)}</td><td>${t.open ? '<i>still open</i>' : D.dates[t.x]}</td>` +
        `<td class="num">$${D.close[t.x].toFixed(2)}${t.open ? ' <span class="hint">(latest)</span>' : ''}</td><td class="num">${t.hold}</td><td class="num">${t.cal}</td>` +
        `<td class="num" style="background:${color(t.ret, cp)};color:${txtColor(t.ret, cp)}">${pct(t.ret)}</td></tr>`; }).join('')
      : '<tr><td colspan="7" class="note">No trades for this pair in the range.</td></tr>');
  return T;
}

// ---------- price chart ----------
function scaleTrace(x0, cap) {
  return {type: 'scatter', x: [x0], y: [null], mode: 'markers', hoverinfo: 'skip', showlegend: false,
    marker: {size: 0.1, opacity: 0, color: [0], cmin: -cap, cmax: cap, showscale: true,
      colorscale: D.gradient.map(([p, c]) => [(p + 1) / 2, c]),
      colorbar: {orientation: 'h', x: 1, xanchor: 'right', y: 1.02, yanchor: 'bottom', len: 0.34, thickness: 10, outlinewidth: 0,
        tickfont: {size: 10}, tickvals: [-cap, -cap / 2, 0, cap / 2, cap],
        ticktext: [`≤ -${cap}%`, `-${cap / 2}%`, '0%', `+${cap / 2}%`, `≥ +${cap}%`],
        title: {text: `${H}D forward return`, side: 'top', font: {size: 11}}}}};
}
function chart(ST, M) {
  const cap = capFor(H, D.cap), xs = D.dates.slice(RA, RB + 1);
  const tr = [{type: 'scatter', mode: 'lines', x: xs, y: D.close.slice(RA, RB + 1), name: 'Close', line: {color: '#8a8984', width: 1.2},
    hovertemplate: '%{x|%b %d, %Y}<br>Close $%{y:,.2f}<extra></extra>', showlegend: false}];
  SLOTS.forEach(s => {
    if (!ruleKeys(s.id).length) return;
    const st = ST[s.id], y = STY[s.id], X = [], Y = [], c = [], cd = [], tx = [], full = $('fullHover').checked;
    st.idx.forEach(i => { X.push(D.dates[i]); Y.push(D.close[i] * y.off); c.push(color(D.fwd[H][i], cap));
      cd.push([i, s.id]); tx.push(hoverText(i, s.id, full)); });
    tr.push({type: 'scatter', mode: 'markers', x: X, y: Y, customdata: cd, text: tx, name: `${s.label} (${st.n})`,
      marker: {symbol: y.sym, size: s.id.endsWith('2') ? 9 : 10, color: c, line: {color: y.col, width: 1.4}},
      hoverlabel: {align: 'left', font: {family: 'Consolas, Menlo, "DejaVu Sans Mono", monospace', size: 11, color: INK}, bgcolor: '#fff', bordercolor: y.col},
      hovertemplate: '%{text}<extra></extra>'});
  });
  if ($('showTrips').checked) {
    const p = PAIRS.find(x => x.id === $('chartPair').value);
    if (p && p.b.some(id => ruleKeys(id).length) && p.s.some(id => ruleKeys(id).length)) {
      const T = tradesFor(M, p);
      [[true, '#1f8a44', 'wins'], [false, '#c0392b', 'losses']].forEach(([w, col, lab]) => {
        const X = [], Y = [], tx = [];
        T.filter(t => (t.ret > 0) === w).forEach(t => { X.push(D.dates[t.e], D.dates[t.x], null); Y.push(D.close[t.e], D.close[t.x], null);
          const s = `${D.dates[t.e]} → ${t.open ? 'open' : D.dates[t.x]} · ${t.hold} trading days · ${pct(t.ret)}`; tx.push(s, s, null); });
        tr.push({type: 'scatter', mode: 'lines', x: X, y: Y, text: tx, name: `${p.label}: ${lab}`, line: {color: col, width: 3},
          opacity: 0.75, hovertemplate: '%{text}<extra></extra>', connectgaps: false});
      });
    }
  }
  tr.push(scaleTrace(xs[0], cap));
  const shapes = [], ann = [];
  if (D.split <= RB) {
    const x0 = D.dates[Math.max(D.split, RA)];
    shapes.push({type: 'rect', xref: 'x', yref: 'paper', x0, x1: D.dates[RB], y0: 0, y1: 1, fillcolor: '#2a78d6', opacity: 0.05, line: {width: 0}});
    ann.push({x: x0, y: 1, xref: 'x', yref: 'paper', text: 'test period →', showarrow: false, xanchor: 'left', yanchor: 'top', font: {size: 11, color: '#2a78d6'}});
  }
  const lay = {height: 680, margin: {l: 60, r: 20, t: 80, b: 40}, plot_bgcolor: SURF, paper_bgcolor: '#fff', uirevision: `${RA}-${RB}`,
    font: {family: 'Inter, Segoe UI, Arial, sans-serif', color: INK2, size: 12}, hoverlabel: {bgcolor: '#fff', font: {color: INK}},
    legend: {orientation: 'h', x: 0, xanchor: 'left', y: 1.02, yanchor: 'bottom', font: {size: 11}},
    xaxis: {showgrid: false, linecolor: GRIDC, range: [D.dates[RA], D.dates[RB]], rangeslider: {visible: true, thickness: 0.06, range: [D.dates[RA], D.dates[RB]]}},
    yaxis: {type: $('logY').checked ? 'log' : 'linear', gridcolor: GRIDC, tickprefix: '$'}, shapes, annotations: ann};
  Plotly.react('priceChart', tr, lay, {displaylogo: false, responsive: true});
  const el = $('priceChart');
  if (!el._detailHooked && el.on) {
    el._detailHooked = true;
    const pick = ev => { const p = ev && ev.points && ev.points.find(q => Array.isArray(q.customdata)); if (p) detail(p.customdata[0], p.customdata[1]); };
    el.on('plotly_hover', pick); el.on('plotly_click', pick);
  }
}

// ---------- full D/W/M readout for one signal day (tooltip + panel) ----------
const NBSP = '\u00a0';
const padR = (t, w) => t.length >= w ? t.slice(0, w) : t + NBSP.repeat(w - t.length);
const padL = (t, w) => t.length >= w ? t : NBSP.repeat(w - t.length) + t;
const esh = t => String(t).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
function firedOn(i) { return lastM ? SLOTS.filter(s => ruleKeys(s.id).length && lastM[s.id][i]).map(s => s.id) : []; }
function hoverText(i, sid, full) {
  let t = `<b>${SL[sid].label.toUpperCase()}</b> ${fmtDate(D.dates[i])} · Close $${D.close[i].toFixed(2)}<br>` +
    D.fwdDays.map(n => `${n}D ${pct(D.fwd[String(n)][i], 1)}`).join(' · ');
  if (!full) return t;
  const used = new Set(ruleKeys(sid)), W = 13, C = 9;
  const row = (n, isBool) => { const ks = D.tfs.map(tf => `${tf.tf}_${n.name}`);
    const line = esh(padR(n.short, W)) + ks.map(k => padL(fmtVal(k, F[k][i]), C)).join('');
    return ks.some(k => used.has(k)) ? `<b>${line}</b>` : line; };
  t += '<br>' + padR('', W) + D.tfs.map(tf => padL(tf.name, C)).join('');
  t += '<br>' + D.numNames.map(n => row(n)).join('<br>');
  t += '<br>' + D.boolNames.map(n => row(n, true)).join('<br>');
  return t;
}
function detail(i, sid) {
  const used = new Set(ruleKeys(sid)), fired = firedOn(i);
  const row = n => { const ks = D.tfs.map(tf => `${tf.tf}_${n.name}`);
    return `<tr class="${ks.some(k => used.has(k)) ? 'used' : ''}"><td>${esc(n.label)}</td>` +
      ks.map(k => `<td class="num${used.has(k) ? ' b' : ''}">${fmtVal(k, F[k][i])}</td>`).join('') + '</tr>'; };
  $('sigDetail').innerHTML = `<h2 style="font-size:14px;margin-top:10px">${fired.map(chip).join('')} ${fmtDate(D.dates[i])} · close $${D.close[i].toFixed(2)}
    <span class="pill">${D.fwdDays.map(n => `${n}D ${pct(D.fwd[String(n)][i])}`).join(' · ')}</span></h2>
    <table class="det"><tr><th>Indicator</th>${D.tfs.map(t => `<th class="num">${t.name}</th>`).join('')}</tr>
    <tr class="sep"><td colspan="${D.tfs.length + 1}">Values</td></tr>${D.numNames.map(row).join('')}
    <tr class="sep"><td colspan="${D.tfs.length + 1}">Yes / no</td></tr>${D.boolNames.map(row).join('')}</table>
    <div class="note">Highlighted rows = fields in the ${esc(SL[sid].label)} rule. Weekly and Monthly are the week-to-date and month-to-date
    values as they read after this day's close.</div>`;
}

// ---------- heatmap (bins from the values inside the date range) ----------
let edgeCache = {key: '', E: {}};
function edgesFor(k) {
  const ck = `${RA}-${RB}`;
  if (edgeCache.key !== ck) edgeCache = {key: ck, E: {}};
  if (edgeCache.E[k]) return edgeCache.E[k];
  const v = []; for (let i = RA; i <= RB; i++) if (F[k][i] !== null) v.push(F[k][i]);
  v.sort((a, b) => a - b);
  const nd = MK[k].nd, e = [];
  if (v.length) for (let q = 0; q <= 10; q++) {
    const p = (v.length - 1) * q / 10, lo = Math.floor(p), x = Number((v[lo] + (v[Math.ceil(p)] - v[lo]) * (p - lo)).toFixed(nd));
    if (!e.length || x > e[e.length - 1]) e.push(x);
  }
  return (edgeCache.E[k] = e.length >= 2 ? e : (e.length ? [e[0], e[0]] : [0, 0]));
}
function binOf(edges, v) {
  if (v === null) return -1;
  const nb = edges.length - 1;
  for (let b = 0; b < nb; b++) if (v < edges[b + 1] || b === nb - 1) return v >= edges[b] || b === 0 ? b : -1;
  return -1;
}
function heat() {
  const ctx = $('heatCtx').value, h = H, cap = capFor(H, D.heatCap);
  $('heatTitle').textContent = `Where the forward returns are: average ${h}-day forward return by indicator value` +
    (ctx === 'rule' ? ` (${SL[side].label} rule context)` : '');
  const E = Object.fromEntries(D.meta.filter(m => m.kind === 'num').map(m => [m.k, edgesFor(m.k)]));
  const maxB = Math.max(...Object.values(E).map(e => e.length - 1));
  let out = '';
  D.tfs.forEach(t => {
    out += `<tr class="tfh"><td colspan="${maxB + 1}">${t.name}</td></tr>`;
    D.numNames.forEach(n => {
      const k = `${t.tf}_${n.name}`, edges = E[k], nb = edges.length - 1;
      const base = ctx === 'rule' ? mask(side, k, true) : null;
      const vals = Array.from({length: nb}, () => []), cnt = new Array(nb).fill(0);
      for (let i = RA; i <= RB; i++) {
        if (base && !base[i]) continue;
        const b = binOf(edges, F[k][i]); if (b < 0) continue;
        cnt[b]++; const f = D.fwd[h][i]; if (f !== null) vals[b].push(f);
      }
      const e = state[side][k] || {};
      out += `<tr><td>${n.label}</td>` + Array.from({length: maxB}, (_, b) => {
        if (b >= nb) return '<td></td>';
        const lo = edges[b], hi = edges[b + 1], av = mean(vals[b]);
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
    render();
  });

  let ob = '<tr><th></th>' + D.tfs.map(t => `<th colspan="2" style="text-align:center">${t.name}</th>`).join('') + '</tr>' +
    '<tr><th></th>' + D.tfs.map(() => '<th style="text-align:center">yes</th><th style="text-align:center">no</th>').join('') + '</tr>';
  D.boolNames.forEach(n => {
    ob += `<tr><td>${n.label}</td>` + D.tfs.map(t => {
      const k = `${t.tf}_${n.name}`, base = ctx === 'rule' ? mask(side, k, true) : null;
      return [1, 0].map(val => {
        const v = []; let c = 0;
        for (let i = RA; i <= RB; i++) { if ((base && !base[i]) || F[k][i] !== val) continue; c++; const f = D.fwd[h][i]; if (f !== null) v.push(f); }
        const av = mean(v), e = state[side][k] || {}, sel = e.sel === val ? ' sel' : '';
        return `<td class="c${c < 20 ? ' thin' : ''}${sel}" data-k="${k}" data-v="${val}" style="background:${color(av, cap)};color:${txtColor(av, cap)}" ` +
          `title="${t.name} ${n.label}: ${val ? 'yes' : 'no'}\n${c} days · avg ${pct(av)}">${av === null ? '·' : pct(av, 1)}<span class="rng">${c} days</span></td>`;
      }).join('');
    }).join('') + '</tr>';
  });
  $('heatBool').innerHTML = ob;
  $('heatBool').querySelectorAll('td.c').forEach(td => td.onclick = () => {
    state[side][td.dataset.k] = {sel: Number(td.dataset.v)}; render();
  });
}

// ---------- suggestions (computed by the script on the latest optimizer years, per horizon) ----------
function baselineRange(a, b) { const v = []; for (let i = a; i < b; i++) { const f = D.fwd[H][i]; if (f !== null) v.push(f); } return mean(v); }
function sugg() {
  const S = D.suggest[H], hc = capFor(H, D.heatCap);
  $('sugPill').textContent = `for the ${hName(H)} horizon`;
  $('sugNote').innerHTML = `Recomputed every time the script runs, from the latest ${D.optYears} years only: picked on
    ${fmtDate(D.optStartDate)} to ${fmtDate(D.dates[D.split - 1])} (${D.opt.trainPct}%), then checked on ${fmtDate(D.splitDate)} to
    ${fmtDate(D.dates[N - 1])}, days the search never saw. So running it for another ticker gives that ticker's own suggestions.
    Search: single thresholds at every 5th percentile of each indicator plus the yes/no fields, combined greedily up to ${D.opt.maxConds}
    conditions; each rule had to fire on ${D.opt.minDays}+ days in ${D.opt.minEp}+ separate streaks. Rule 2 is the best rule that uses
    <i>different</i> indicators from rule 1, so it's a genuinely separate strategy. Change the forward-return horizon above to see the
    suggestions for 1 week up to ~3 months. Trust the test columns, not training. Raw-dollar MACD fields are left out.`;
  const bTr = baselineRange(D.optStart, D.split), bTe = baselineRange(D.split, N);
  SLOTS.forEach(s => {
    const g = S[s.id], row = (lab, st) => `<tr><td>${lab}</td><td class="num">${st.n}</td><td class="num">${st.ep}</td>` +
      `<td class="num"><span class="sw" style="background:${color(st.avg, hc)}"></span>${pct(st.avg)}</td><td class="num">${st.win === null ? 'n/a' : st.win.toFixed(0) + '%'}</td></tr>`;
    const kindWord = s.dir > 0 ? 'buy' : 'sell', twin = s.dir > 0 ? BUYS : SELLS;
    const head = `<h2 style="font-size:14px;color:${STY[s.id].col}">${STY[s.id].mark} Suggested ${kindWord} rule ${s.id.endsWith('1') ? '1 (best)' : '2 (alternative, different indicators)'}</h2>`;
    if (!g.rule.length) { $('sug_' + s.id).innerHTML = head + '<div class="note">No rule met the minimum signal-day and streak requirements.</div>'; return; }
    $('sug_' + s.id).innerHTML = head + `<div class="hint" style="margin-bottom:6px"><b>${esc(g.text)}</b></div><table><tr><th></th><th class="num">Days</th><th class="num">Streaks</th>` +
      `<th class="num">Avg ${H}D</th><th class="num">% positive</th></tr>` + row('Training', g.train) + row('Test (unseen)', g.test) +
      `<tr><td class="hint">All days, training</td><td></td><td></td><td class="num">${pct(bTr)}</td><td></td></tr>` +
      `<tr><td class="hint">All days, test</td><td></td><td></td><td class="num">${pct(bTe)}</td><td></td></tr></table>` +
      `<div class="row" style="margin-top:8px">` + twin.map(t => `<button class="sm" data-load="${s.id}" data-into="${t}">Load into ${SL[t].label}</button>`).join('') + '</div>';
  });
  document.querySelectorAll('[data-load]').forEach(b => b.onclick = () => {
    side = b.dataset.into; state[side] = condsToState(D.suggest[H][b.dataset.load].rule); drawTabs(); render();
    window.scrollTo({top: 0, behavior: 'smooth'}); });
  [['buy1', 'singBuy'], ['sell1', 'singSell']].forEach(([s, id]) => {
    const rows = S[s].singles;
    $(id).innerHTML = `<tr><th>Condition</th><th class="num">Train days</th><th class="num">Train avg</th>` +
      `<th class="num">Test days</th><th class="num">Test avg</th><th class="num">Test % pos</th></tr>` +
      (rows.length ? rows.map((r, j) => `<tr class="click" data-s="${s}" data-j="${j}"><td>${esc(r.text)}</td><td class="num">${r.train.n}</td>` +
        `<td class="num">${pct(r.train.avg)}</td><td class="num">${r.test.n}</td>` +
        `<td class="num"><span class="sw" style="background:${color(r.test.avg, hc)}"></span>${pct(r.test.avg)}</td>` +
        `<td class="num">${r.test.win === null ? 'n/a' : r.test.win.toFixed(0) + '%'}</td></tr>`).join('')
        : '<tr><td colspan="6" class="note">None met the minimums.</td></tr>');
  });
  document.querySelectorAll('#singBuy tr.click, #singSell tr.click').forEach(tr => tr.onclick = () => {
    const isBuy = tr.dataset.s === 'buy1', target = isBuy ? (BUYS.includes(side) ? side : 'buy1') : (SELLS.includes(side) ? side : 'sell1');
    side = target; addCond(side, D.suggest[H][tr.dataset.s].singles[+tr.dataset.j].cond); drawTabs(); render(); });
}
function addCond(s, c) {
  const e = state[s][c.k] || (state[s][c.k] = {});
  if (c.op === '>=') e.min = c.v; else if (c.op === '<=') e.max = c.v; else e.sel = c.v;
}

// ---------- all signal days, every rule ----------
let hitOn = null, hitFirst = false;
let lastM = null;
function hitControls() {
  if (!hitOn) hitOn = Object.fromEntries(SLOTS.map(s => [s.id, true]));
  $('hitSlots').innerHTML = '<span class="hint">Show:</span>' + SLOTS.map(s => `<label class="ctl"><input type="checkbox" data-hs="${s.id}" ${hitOn[s.id] ? 'checked' : ''}>` +
    `${chip(s.id)}${s.label}${ruleKeys(s.id).length ? '' : ' <span class="hint">(empty)</span>'}</label>`).join('') +
    `<label class="ctl" style="margin-left:12px"><input type="checkbox" id="hitFirst" ${hitFirst ? 'checked' : ''}> first day of each streak only</label>`;
  $('hitSlots').querySelectorAll('input[data-hs]').forEach(el => el.onchange = () => { hitOn[el.dataset.hs] = el.checked; hits(lastM); });
  $('hitFirst').onchange = () => { hitFirst = $('hitFirst').checked; hits(lastM); };
}
function hits(M) {
  lastM = M;
  const on = SLOTS.map(s => s.id).filter(id => hitOn[id] && ruleKeys(id).length);
  const keys = [...new Set(on.flatMap(id => ruleKeys(id)))];
  keys.sort((a, b) => D.meta.findIndex(m => m.k === a) - D.meta.findIndex(m => m.k === b));
  const fires = (id, i) => M[id][i] && (!hitFirst || i === 0 || !M[id][i - 1]);
  const rows = []; for (let i = RB; i >= RA; i--) if (on.some(id => fires(id, i))) rows.push(i);
  $('hitsTitle').textContent = `Signal days for every rule (${rows.length.toLocaleString()} ${hitFirst ? 'streak starts' : 'days'} in the range, newest first)`;
  $('hits').innerHTML = `<tr><th>Date</th><th>Signals</th><th class="num">Close</th>` +
    D.fwdDays.map(n => `<th class="num">${n}D fwd</th>`).join('') +
    keys.map(k => `<th class="num" title="${MK[k].tfName} ${MK[k].label}">${MK[k].tf} ${MK[k].name}<br>${on.filter(id => ruleKeys(id).includes(k)).map(id => `<span class="hint">${STY[id].short}</span>`).join(' ')}</th>`).join('') + '</tr>' +
    (rows.length ? rows.slice(0, 150).map(i => {
      const fired = on.filter(id => fires(id, i));
      return `<tr><td>${D.dates[i]}</td><td>${fired.map(chip).join('')}</td><td class="num">$${D.close[i].toFixed(2)}</td>` +
        D.fwdDays.map(n => { const v = D.fwd[String(n)][i], cp = capFor(n, D.cap);
          return `<td class="num" style="background:${color(v, cp)};color:${txtColor(v, cp)}">${v === null ? 'pending' : pct(v)}</td>`; }).join('') +
        keys.map(k => `<td class="num${fired.some(id => ruleKeys(id).includes(k)) ? ' b' : ''}">${fmtVal(k, F[k][i])}</td>`).join('') + '</tr>';
    }).join('') : `<tr><td colspan="${3 + D.fwdDays.length + keys.length}" class="note">No selected rule fired in this range.</td></tr>`);
}

// ---------- render ----------
function render() {
  const M = Object.fromEntries(SLOTS.map(s => [s.id, mask(s.id)]));
  const ST = Object.fromEntries(SLOTS.map(s => [s.id, statsOf(M[s.id], H)]));
  fillGrid(); results(ST); today(M); chart(ST, M); trips(M); heat(); sugg(); hitControls(); hits(M);
  autosave();
}
function postHeight() {
  if (window.parent !== window) window.parent.postMessage({type: 'scanner-height', h: document.documentElement.scrollHeight}, '*');
}

(function init() {
  $('horizon').innerHTML = D.fwdDays.map(n => `<option value="${n}">${hName(n)}</option>`).join('');
  $('from').min = $('to').min = D.dates[0]; $('from').max = $('to').max = D.dates[N - 1];
  periodOptions();
  SLOTS.forEach(s => state[s.id] = condsToState(D.rules[s.id].conds));
  const pairOpts = PAIRS.map(p => `<option value="${p.id}">${p.label}</option>`).join('');
  $('tripPair').innerHTML = $('chartPair').innerHTML = pairOpts;

  const saved = sget(KEY_CUR);
  if (saved && saved.rules) {
    SLOTS.forEach(s => { if (saved.rules[s.id]) state[s.id] = fromPy(saved.rules[s.id]); });
    if (saved.H && D.fwdDays.map(String).includes(String(saved.H))) H = String(saved.H);
    if (saved.period && [...$('period').options].some(o => o.value === saved.period)) $('period').value = saved.period;
    if (saved.period === 'custom') { $('from').value = saved.from || ''; $('to').value = saved.to || ''; }
    if (saved.side && SL[saved.side]) side = saved.side;
  }
  $('horizon').value = H;
  applyPeriod(); drawTabs(); drawGrid(); drawLib(); baseNote();

  $('resetBtn').onclick = () => { state[side] = condsToState(D.rules[side].conds); render(); };
  $('suggestBtn').onclick = () => { state[side] = condsToState(D.suggest[H][side].rule); render(); };
  $('clearBtn').onclick = () => { state[side] = {}; render(); };
  $('horizon').onchange = () => { H = $('horizon').value; render(); };
  $('period').onchange = () => { applyPeriod(); render(); };
  ['from', 'to'].forEach(id => $(id).addEventListener('change', () => { $('period').value = 'custom'; applyPeriod(); render(); }));
  ['heatCtx', 'logY', 'showTrips', 'chartPair', 'fullHover'].forEach(id => $(id).addEventListener('change', render));
  $('tripPair').onchange = () => { $('chartPair').value = $('tripPair').value; render(); };

  $('saveBtn').onclick = () => {
    if (!ruleKeys(side).length) { note('This rule is empty. Set some conditions first.'); return; }
    const name = $('saveName').value.trim() || `${SL[side].label}: ${ruleDesc(side)}`.slice(0, 80);
    const kind = SL[side].dir > 0 ? 'buy' : 'sell', j = library.findIndex(r => r.name === name && r.kind === kind);
    const rec = {name, kind, rule: toPy(side), ticker: D.ticker, saved: new Date().toISOString().slice(0, 10)};
    if (j >= 0) library[j] = rec; else library.push(rec);
    const ok = sset(KEY_LIB, library); drawLib(); $('libSel').value = String(j >= 0 ? j : library.length - 1); $('saveName').value = '';
    note(ok ? `<span class="ok-msg">Saved “${esc(name)}”.</span> Load it into any slot, for any ticker. Use Export to keep a file copy.`
      : `Saved for this page visit only (browser storage is blocked). Use <b>Export rules</b> to keep it.`);
  };
  $('libLoad').onclick = () => { const r = library[+$('libSel').value]; if (!r) return;
    state[side] = fromPy(r.rule); render(); note(`<span class="ok-msg">Loaded “${esc(r.name)}” into ${SL[side].label}.</span>`); };
  $('libDel').onclick = () => { const j = +$('libSel').value, r = library[j]; if (!r || !confirm(`Delete saved rule “${r.name}”?`)) return;
    library.splice(j, 1); sset(KEY_LIB, library); drawLib(); baseNote(); };
  $('exportBtn').onclick = () => {
    download(`mtf_signal_rules_${D.ticker}.json`, JSON.stringify({ticker: D.ticker, saved: new Date().toISOString(), horizon: Number(H),
      rules: Object.fromEntries(SLOTS.map(s => [s.id, toPy(s.id)])), library}, null, 2));
    note(`<span class="ok-msg">Downloaded mtf_signal_rules_${esc(D.ticker)}.json.</span> Put it next to mtf_signal_scanner.py and the next run
      uses these four rules for alerts and as the report's starting rules (an empty slot = no alerts for it).`);
  };
  $('importBtn').onclick = () => $('importFile').click();
  $('importFile').onchange = async () => {
    const f = $('importFile').files[0]; if (!f) return;
    try {
      const o = JSON.parse(await f.text());
      SLOTS.forEach(s => { if (o.rules && o.rules[s.id]) state[s.id] = fromPy(o.rules[s.id]); });
      (o.library || []).forEach(r => { if (!library.some(x => x.name === r.name && x.kind === r.kind)) library.push(r); });
      sset(KEY_LIB, library); drawLib(); render();
      note(`<span class="ok-msg">Imported ${esc(f.name)}.</span>`);
    } catch (e) { note(`Couldn't read ${esc(f.name)}: ${esc(e.message)}`); }
    $('importFile').value = '';
  };
  $('copyPy').onclick = async () => {
    const t = pyText();
    try { await navigator.clipboard.writeText(t); note('<span class="ok-msg">Copied.</span> Paste over BUY_RULE / SELL_RULE / BUY_RULE_2 / SELL_RULE_2 in the script.'); }
    catch (e) { note(`<pre style="margin:4px 0;font-size:12px">${esc(t)}</pre>`); }
  };

  render();
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
    global TICKER
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--ticker", help=f"symbol to scan (default {TICKER}, or the SIGNAL_TICKER env variable)")
    ap.add_argument("--alert", action="store_true", help="check the latest close and email alerts for any rule that fires (no report)")
    ap.add_argument("--force", action="store_true", help="with --alert: send even if the latest bar isn't from today")
    args = ap.parse_args()
    if args.ticker:
        TICKER = args.ticker.strip().upper()

    print(f"Downloading {TICKER} daily history and replaying the scanner...")
    win, opt_start, split_date, suggest, rules = prepare()
    print(f"Window {win.index.min():%Y-%m-%d} to {win.index.max():%Y-%m-%d} ({len(win):,} days). Optimizer uses "
          f"{opt_start:%Y-%m-%d} onward: training before {split_date:%Y-%m-%d}, test after.")
    for slot, lab, *_ in SLOTS:
        g = suggest[TARGET_FWD_DAYS][slot]
        tr, te = g["train"], g["test"]
        print(f"Suggested {lab.upper()} ({TARGET_FWD_DAYS}D): {g['text']}")
        if tr["avg"] is not None:
            te_avg = "n/a" if te["avg"] is None else f"{te['avg']:+.2f}%"
            print(f"   train {tr['n']} days avg {tr['avg']:+.2f}% | test {te['n']} days avg {te_avg}")

    if args.alert:
        run_alerts(win, suggest, rules, split_date, force=args.force)
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["Close", *[f"Fwd_{n}D%" for n in FORWARD_DAYS], *[SLOT_COL[s] for s, *_ in SLOTS], "Period", *ALL_KEYS]
    win[cols].round(4).to_csv(CSV_OUT, index_label="Date")
    build_report(win, opt_start, split_date, suggest, rules)
    st = latest_status(win, rules)
    for slot, lab, *_ in SLOTS:
        print(f"Latest close {win.index[-1]:%Y-%m-%d}: {lab.upper()} rule "
              f"{'FIRES' if st[slot]['fires'] else 'no signal'} ({rules[slot]['source']}: {rules[slot]['text']})")
    print(f"Report: {HTML_OUT}\nCSV:    {CSV_OUT}")


if __name__ == "__main__":
    main()
