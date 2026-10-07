"""
ATH Scanner - dashboard + all-time-high runs with separate ENTRY and EXIT rules
================================================================================

One script, two jobs:

  python ath_scanner_backtest.py            -> builds the interactive dashboard page
  python ath_scanner_backtest.py --alert    -> sends the daily email (the Scanner_All-Time-Highs
                                               job calls this through scanner_All-Time-High_1.py)

Both read the SAME rules file, ath_scanner_rules.json (next to this script), so what the
dashboard previews is exactly what the email sends. Edit the rules in the dashboard, copy or
download the JSON, commit it to the repo, and the next run emails with the new rules.

How a run works (a simple on/off state per ticker, day by day):
  * Not in a run  -> check the ENTRY conditions. If they pass, the run starts ("new run").
                     The breakout level is frozen at that moment: the old all-time high the
                     run broke through.
  * In a run      -> check the EXIT conditions. If they pass, the run ends ("run ended").
Entry is meant to be strict and exit looser (hysteresis), so a healthy pause doesn't end a run.

Each condition can be switched on/off and joined to the previous one with AND or OR.
AND binds tighter than OR, like normal logic:  A AND B OR C  =  (A AND B) OR C.

Distances can be in % or in multiples of the ticker's own 14-day ATR, so a volatile stock
gets more room than a quiet one.

Entry conditions:  min new ATH closes in window, latest ATH within N days, close within X of
                   the ATH, holding the breakout (with tolerance), stochastic %K.
Exit conditions:   close more than X below the ATH, no new ATH for N days, closes below the
                   breakout (with tolerance), too few ATH closes in window, stochastic %K.

Email sections:
  NEW RUNS            - entered today (indicator filters, if any, apply here)
  RUN ENDED           - exited today: a possible peak, with the exit condition(s) that fired
  STILL RUNNING       - everything else currently in a run (optional)

Requires: pip install yfinance pandas requests plotly lxml
"""

import json
import math
import os
import smtplib
import sys
import warnings
from email.mime.text import MIMEText
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

# ============================================================
# SETTINGS
# ============================================================

DATA_PERIOD = "max"          # full history needed for a TRUE all-time high
RVOL_LOOKBACK = 20
ATR_PERIOD = 14

BACKTEST_MONTHS = 12
MAX_WINDOW = 60              # largest trailing window the page lets you pick
FORWARD_DAYS = [5, 10, 20]
COLOR_FWD_DAYS = 10          # default forward return used for colors
COLOR_CAP_PCT = 10           # +/- this % (or beyond) = darkest color
HEATMAP_DAYS = 60
HEATMAP_ROWS = 40

SCANNER_NAME = "ATH Scanner"
SCANNER_TITLE = "All-Time-High Runs - dashboard + email rules"
SCANNER_ORDER = 10

RULES_FILE_NAME = "ath_scanner_rules.json"

# Starting rules, used when ath_scanner_rules.json doesn't exist yet (or is the old format).
# join = how this condition connects to the previous switched-on one ("and" / "or").
# unit = "pct" (percent) or "atr" (multiples of the ticker's 14-day ATR).
# stoch op: ge (%K >= x), le (%K <= x), kd_up (%K above %D), kd_dn (%K below %D).
DEFAULT_RULES = {
    "window_days": 30,
    "entry": {
        "count":    {"on": True,  "join": "and", "n": 3},
        "recent":   {"on": True,  "join": "and", "n": 5},
        "near":     {"on": True,  "join": "and", "x": 5.0, "unit": "pct"},
        "breakout": {"on": True,  "join": "and", "x": 1.0, "unit": "pct", "n": 1},
        "stoch":    {"on": False, "join": "and", "op": "ge", "x": 50.0},
    },
    "exit": {
        "drawdown": {"on": True,  "join": "or", "x": 3.0, "unit": "atr"},
        "stale":    {"on": True,  "join": "or", "n": 15},
        "breakout": {"on": True,  "join": "or", "x": 1.0, "unit": "pct", "n": 1},
        "count":    {"on": False, "join": "or", "n": 1},
        "stoch":    {"on": False, "join": "or", "op": "le", "x": 50.0},
    },
    "filters": {
        "rsi_min": None,
        "rsi_max": None,
        "rvol_min": None,
        "ret_min": None,
        "macd": "any",       # any | bull | bear
        "stoch": "any",
    },
    "email": {
        "send_when": "changes",   # changes (new or ended) | new | always
        "list_running": True,
    },
}

# Limits for each condition's fields. "K" = the window, "K-1" = window minus one.
COND_SPEC = {
    "entry": {
        "count":    {"n": [1, "K"]},
        "recent":   {"n": [0, "K-1"]},
        "near":     {"x": [0, None], "unit": True},
        "breakout": {"x": [0, None], "unit": True, "n": [1, 10]},
        "stoch":    {"x": [0, 100], "op": True},
    },
    "exit": {
        "drawdown": {"x": [0, None], "unit": True},
        "stale":    {"n": [1, 250]},
        "breakout": {"x": [0, None], "unit": True, "n": [1, 10]},
        "count":    {"n": [1, "K"]},
        "stoch":    {"x": [0, 100], "op": True},
    },
}
STOCH_OPS = ("ge", "le", "kd_up", "kd_dn")

SECTOR_ETF_MAP = {
    "Energy": "XLE", "Materials": "XLB", "Industrials": "XLI",
    "Consumer Discretionary": "XLY", "Consumer Staples": "XLP", "Health Care": "XLV",
    "Financials": "XLF", "Information Technology": "XLK", "Communication Services": "XLC",
    "Utilities": "XLU", "Real Estate": "XLRE",
}

EMAIL_USER = os.environ.get("EMAIL_USER")
EMAIL_PASS = os.environ.get("EMAIL_PASS")
ALERT_TO = os.environ.get("ALERT_TO")

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()
RULES_FILE = SCRIPT_DIR / RULES_FILE_NAME

if os.environ.get("BACKTEST_OUTPUT_DIR"):
    OUT_DIR = Path(os.environ["BACKTEST_OUTPUT_DIR"]).resolve()
else:
    OUT_DIR = SCRIPT_DIR
HTML_OUT = OUT_DIR / os.environ.get("BACKTEST_HTML_NAME", "ath_backtest_report.html")
CSV_OUT = OUT_DIR / "ath_backtest_days.csv"
PLOTLY_JS = os.environ.get("BACKTEST_PLOTLY_JS", "inline")

GRADIENT = [(-1.0, "#8e1b1b"), (-0.5, "#e0584e"), (0.0, "#e4e2dc"),
            (0.5, "#4fae68"), (1.0, "#0b5a24")]
RVOL_SCALE = [(0.0, "#fff7f3"), (0.25, "#fcc5c0"), (0.5, "#f768a1"), (0.75, "#ae017e"), (1.0, "#49006a")]
RVOL_RANGE = [0.5, 3.0]
PENDING_COLOR = "#f1f0ec"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"


# ============================================================
# RULES  (the page's clampRules() does the same thing)
# ============================================================

def _round(v):
    return int(math.floor(float(v) + 0.5))          # same as JavaScript Math.round


def _numval(v, default):
    if v is None or v == "" or isinstance(v, bool):
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(f) else f


def _lim(v, K):
    return K if v == "K" else (K - 1 if v == "K-1" else v)


def normalize_rules(raw):
    """Merge a (possibly partial) rules dict over the defaults and clamp values."""
    raw = raw if isinstance(raw, dict) else {}
    r = json.loads(json.dumps(DEFAULT_RULES))
    K = min(max(_round(_numval(raw.get("window_days"), r["window_days"])), 2), MAX_WINDOW)
    r["window_days"] = K
    for side, conds in COND_SPEC.items():
        src = raw.get(side) if isinstance(raw.get(side), dict) else {}
        for key, spec in conds.items():
            c = r[side][key]
            s = src.get(key) if isinstance(src.get(key), dict) else {}
            c["on"] = bool(s.get("on", c["on"]))
            c["join"] = "or" if s.get("join", c["join"]) == "or" else "and"
            if "n" in spec:
                lo, hi = _lim(spec["n"][0], K), _lim(spec["n"][1], K)
                c["n"] = min(max(_round(_numval(s.get("n"), c["n"])), lo), hi)
            if "x" in spec:
                lo, hi = spec["x"]
                v = max(float(_numval(s.get("x"), c["x"])), lo)
                c["x"] = min(v, hi) if hi is not None else v
            if "unit" in spec:
                c["unit"] = "atr" if s.get("unit", c["unit"]) == "atr" else "pct"
            if "op" in spec:
                op = s.get("op", c["op"])
                c["op"] = op if op in STOCH_OPS else c["op"]
    for sec in ("filters", "email"):
        if isinstance(raw.get(sec), dict):
            for k in r[sec]:
                if k in raw[sec]:
                    r[sec][k] = raw[sec][k]
    f = r["filters"]
    for k in ("rsi_min", "rsi_max", "rvol_min", "ret_min"):
        v = _numval(f[k], None)
        f[k] = None if v is None else float(v)
    for k in ("macd", "stoch"):
        f[k] = f[k] if f[k] in ("any", "bull", "bear") else "any"
    e = r["email"]
    e["send_when"] = e["send_when"] if e["send_when"] in ("changes", "new", "always") else "changes"
    e["list_running"] = bool(e["list_running"])
    return r


def load_rules():
    if RULES_FILE.exists():
        try:
            raw = json.loads(RULES_FILE.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and "confirm" in raw and "entry" not in raw:
                print(f"{RULES_FILE.name} is the old single-rule format - using the new entry/exit "
                      "defaults (filters and email settings kept). Commit the new rules file to update it.")
            rules = normalize_rules(raw)
            print(f"Rules: {RULES_FILE.name}")
            return rules, True
        except Exception as exc:
            print(f"Could not read {RULES_FILE.name} ({exc}) - using default rules.")
    else:
        print(f"No {RULES_FILE.name} found - using default rules.")
    return normalize_rules(DEFAULT_RULES), False


# ============================================================
# DATA + INDICATORS
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


def download(tickers):
    return yf.download(tickers, period=DATA_PERIOD, interval="1d", group_by="ticker",
                       auto_adjust=False, threads=True)


def rsi(series, period=14):
    delta = series.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - (100 / (1 + avg_gain / avg_loss))


def macd(series, fast=12, slow=26, signal_period=9):
    line = series.ewm(span=fast, adjust=False).mean() - series.ewm(span=slow, adjust=False).mean()
    return line, line.ewm(span=signal_period, adjust=False).mean()


def stochastic(high, low, close, k_period=14, smooth_k=3, d_period=3):
    lo, hi = low.rolling(k_period).min(), high.rolling(k_period).max()
    k = (100 * (close - lo) / (hi - lo)).rolling(smooth_k).mean()
    return k, k.rolling(d_period).mean()


def atr(high, low, close, period=ATR_PERIOD):
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _num(v, nd):
    return None if v is None or pd.isna(v) else round(float(v), nd)


def prepare(data, sp500, start):
    """Per-ticker arrays on a shared trading-day calendar.

    The calendar starts MAX_WINDOW+1 trading days before `start` so trailing-window
    counts are complete from the first displayed day. Values are rounded (prices and ATR
    to cents, stochastic to 0.1) and every rule is evaluated on these rounded values, in
    Python (email) and in the page, so both always agree.
    """
    idx = pd.DatetimeIndex(data.index).sort_values()
    s0 = int(idx.searchsorted(start))
    cal = idx[max(0, s0 - (MAX_WINDOW + 1)):]
    out = {}
    for _, row in sp500.iterrows():
        t = row["Ticker"]
        try:
            full = data[t].dropna(subset=["Close"])
        except KeyError:
            continue
        if len(full) < 2:
            continue
        close, high, low, vol = full["Close"], full["High"], full["Low"], full["Volume"]
        avg_vol = vol.shift(1).rolling(RVOL_LOOKBACK, min_periods=1).mean()
        ml, sl = macd(close)
        k, d = stochastic(high, low, close)
        ind = pd.DataFrame({
            "c": close,
            "rsi": rsi(close),
            "rvol": vol / avg_vol.where(avg_vol > 0),
            "atr": atr(high, low, close),
            "kv": k,
            "dv": d,
            "m": (ml > sl).astype(int),
            "k": (k > d).astype(int),
        })
        before = close[close.index < cal[0]]
        sl_df = ind.reindex(cal)
        out[t] = {
            "a0": _num(before.max(), 2) if len(before) else None,
            "c": [_num(v, 2) for v in sl_df["c"]],
            "r": [_num(v, 1) for v in sl_df["rsi"]],
            "v": [_num(v, 2) for v in sl_df["rvol"]],
            "atr": [_num(v, 2) for v in sl_df["atr"]],
            "kv": [_num(v, 1) for v in sl_df["kv"]],
            "dv": [_num(v, 1) for v in sl_df["dv"]],
            "m": [None if pd.isna(v) else int(v) for v in sl_df["m"]],
            "k": [None if pd.isna(v) else int(v) for v in sl_df["k"]],
            "company": str(row["Company"]),
            "sector": str(row["Sector_ETF"]),
            "sectorName": str(row["Sector"]),
        }
    return cal, s0 - max(0, s0 - (MAX_WINDOW + 1)), out


# ============================================================
# RUN LOGIC  (mirrored line by line in the page's computeTicker())
# ============================================================

def enabled(rules, side):
    return [(key, rules[side][key]) for key in COND_SPEC[side] if rules[side][key]["on"]]


def eval_expr(items, bits):
    """AND binds tighter than OR. No conditions switched on = never true."""
    if not items:
        return False
    result, acc = False, None
    for idx, (_, cfg) in enumerate(items):
        b = bits[idx]
        if idx == 0:
            acc = b
        elif cfg["join"] == "or":
            result = result or acc
            acc = b
        else:
            acc = acc and b
    return result or acc


def stoch_ok(op, thr, k, d):
    if op == "ge":
        return k is not None and k >= thr
    if op == "le":
        return k is not None and k <= thr
    if k is None or d is None:
        return False
    return k > d if op == "kd_up" else k < d


def below_brk(s, j, brk, cfg):
    c, a = s["c"], s["atr"]
    if c[j] is None or brk is None:
        return False
    if cfg["unit"] == "atr":
        return a[j] is not None and c[j] < brk - cfg["x"] * a[j]
    return c[j] < brk * (1 - cfg["x"] / 100)


def entry_pass(key, cfg, s, st, i):
    px, a = s["c"][i], s["atr"][i]
    if key == "count":
        return st["cnt"][i] >= cfg["n"]
    if key == "recent":
        return st["dsa"][i] is not None and st["dsa"][i] <= cfg["n"]
    if key == "near":
        if cfg["unit"] == "atr":
            return a is not None and (st["lvl"][i] - px) <= cfg["x"] * a
        return st["below"][i] <= cfg["x"]
    if key == "breakout":
        f = st["first"][i]
        if f is None:
            return True
        b, run = st["pmb"][f], 0
        for j in range(f, i + 1):
            if s["c"][j] is None:
                continue
            if below_brk(s, j, b, cfg):
                run += 1
                if run >= cfg["n"]:
                    return False
            else:
                run = 0
        return True
    if key == "stoch":
        return stoch_ok(cfg["op"], cfg["x"], s["kv"][i], s["dv"][i])
    return False


def exit_hit(key, cfg, s, st, i, rbrk):
    px, a = s["c"][i], s["atr"][i]
    if key == "drawdown":
        if cfg["unit"] == "atr":
            return a is not None and (st["lvl"][i] - px) > cfg["x"] * a
        return st["below"][i] > cfg["x"]
    if key == "stale":
        return st["dsa"][i] is None or st["dsa"][i] > cfg["n"]
    if key == "breakout":
        if rbrk is None:
            return False
        need, j = cfg["n"], i
        while j >= 0 and need > 0:
            if s["c"][j] is not None:
                if not below_brk(s, j, rbrk, cfg):
                    return False
                need -= 1
            j -= 1
        return need == 0
    if key == "count":
        return st["cnt"][i] < cfg["n"]
    if key == "stoch":
        return stoch_ok(cfg["op"], cfg["x"], s["kv"][i], s["dv"][i])
    return False


def compute_states(s, rules):
    c = s["c"]
    n, K = len(c), rules["window_days"]
    E, X = enabled(rules, "entry"), enabled(rules, "exit")
    st = {key: [None] * n for key in ("lvl", "below", "dsa", "brk", "first", "start", "why", "wk", "pmb")}
    st["ath"], st["cnt"], st["conf"] = [False] * n, [0] * n, [False] * n
    pm, last, inrun, rstart, rbrk = s["a0"], None, False, None, None
    for i in range(n):
        x = c[i]
        st["pmb"][i] = pm
        if x is not None:
            a = pm is not None and x > pm
            st["ath"][i] = a
            if pm is None or x > pm:
                pm = x
            st["lvl"][i] = pm
            st["below"][i] = (1 - x / pm) * 100
            if a:
                last = i
        k, f = 0, None
        for j in range(max(0, i - K + 1), i + 1):
            if st["ath"][j]:
                k += 1
                if f is None:
                    f = j
        st["cnt"][i], st["first"][i] = k, f
        st["dsa"][i] = None if last is None else i - last
        if x is None:
            if i > 0:
                for key in ("conf", "brk", "start", "why", "wk"):
                    st[key][i] = st[key][i - 1]
            continue
        wbrk = st["pmb"][f] if f is not None else None
        if inrun:
            bits = [exit_hit(key, cfg, s, st, i, rbrk) for key, cfg in X]
            st["brk"][i], st["start"][i] = rbrk, rstart
            if eval_expr(X, bits):
                inrun = False
                st["why"][i] = [key for (key, _), b in zip(X, bits) if b]
                st["wk"][i] = "exit"
        else:
            bits = [entry_pass(key, cfg, s, st, i) for key, cfg in E]
            if eval_expr(E, bits):
                inrun = True
                rstart = f if f is not None else i
                rbrk = wbrk
                st["brk"][i], st["start"][i] = rbrk, rstart
            else:
                st["brk"][i], st["start"][i] = wbrk, f
                st["why"][i] = [key for (key, _), b in zip(E, bits) if not b]
                st["wk"][i] = "entry"
        st["conf"][i] = inrun
    return st


def day_return(c, i):
    if i < 1 or c[i] is None or c[i - 1] is None:
        return None
    return (c[i] / c[i - 1] - 1) * 100


def fwd_return(c, i, n):
    if i + n >= len(c) or c[i] is None or c[i + n] is None:
        return None
    return (c[i + n] / c[i] - 1) * 100


def above_brk(s, st, i):
    b, px = st["brk"][i], s["c"][i]
    if b is None or px is None or b <= 0:
        return None
    return (px / b - 1) * 100


def passes_filters(s, i, f):
    def ge(v, x):
        return v is not None and v >= x

    def le(v, x):
        return v is not None and v <= x
    if f["rsi_min"] is not None and not ge(s["r"][i], f["rsi_min"]):
        return False
    if f["rsi_max"] is not None and not le(s["r"][i], f["rsi_max"]):
        return False
    if f["rvol_min"] is not None and not ge(s["v"][i], f["rvol_min"]):
        return False
    if f["ret_min"] is not None and not ge(day_return(s["c"], i), f["ret_min"]):
        return False
    if f["macd"] != "any" and s["m"][i] != (1 if f["macd"] == "bull" else 0):
        return False
    if f["stoch"] != "any" and s["k"][i] != (1 if f["stoch"] == "bull" else 0):
        return False
    return True


def events(s, st, i, rules):
    """(is_new_alert, is_new_but_filtered, is_ended) for day i."""
    if i < 1 or s["c"][i] is None:
        return False, False, False
    became = st["conf"][i] and not st["conf"][i - 1]
    ended = (not st["conf"][i]) and st["conf"][i - 1]
    ok = passes_filters(s, i, rules["filters"])
    return became and ok, became and not ok, ended


# ============================================================
# TEXT  (the page builds identical strings)
# ============================================================

def _g(v):
    return "n/a" if v is None else f"{v:g}"


def _money(v):
    return "n/a" if v is None else f"${v:,.2f}"


def _pct(v, nd=1, sign=True):
    if v is None:
        return "n/a"
    return f"{v:+.{nd}f}%" if sign else f"{v:.{nd}f}%"


def _unit(cfg):
    return "%" if cfg["unit"] == "pct" else "x ATR"


def stoch_desc(cfg):
    op = cfg["op"]
    if op == "ge":
        return f"Stoch %K >= {_g(cfg['x'])}"
    if op == "le":
        return f"Stoch %K <= {_g(cfg['x'])}"
    return "Stoch %K above %D" if op == "kd_up" else "Stoch %K below %D"


def cond_desc(side, key, cfg, K):
    if key == "stoch":
        return stoch_desc(cfg)
    if side == "entry":
        if key == "count":
            return f"{cfg['n']}+ new ATH closes in {K}D"
        if key == "recent":
            return "a new ATH today" if cfg["n"] == 0 else f"latest ATH within {cfg['n']}D"
        if key == "near":
            return f"close within {_g(cfg['x'])}{_unit(cfg)} of ATH"
        if key == "breakout":
            times = f"{cfg['n']} straight closes" if cfg["n"] > 1 else "close"
            tol = f" by more than {_g(cfg['x'])}{_unit(cfg)}" if cfg["x"] > 0 else ""
            return f"no {times} below breakout{tol}"
    else:
        if key == "drawdown":
            return f"close more than {_g(cfg['x'])}{_unit(cfg)} below ATH"
        if key == "stale":
            return f"no new ATH in over {cfg['n']}D"
        if key == "breakout":
            times = f"{cfg['n']} straight closes" if cfg["n"] > 1 else "a close"
            tol = f" by more than {_g(cfg['x'])}{_unit(cfg)}" if cfg["x"] > 0 else ""
            return f"{times} below breakout{tol}"
        if key == "count":
            return f"fewer than {cfg['n']} new ATH closes in {K}D"
    return key


def expr_text(rules, side):
    K = rules["window_days"]
    items = [(cfg, cond_desc(side, key, cfg, K)) for key, cfg in enabled(rules, side)]
    if not items:
        return "never (no conditions switched on)"
    groups = []
    for idx, (cfg, d) in enumerate(items):
        if idx == 0 or cfg["join"] == "or":
            groups.append([d])
        else:
            groups[-1].append(d)
    multi = len(groups) > 1
    return " OR ".join(("(" + " AND ".join(gr) + ")") if multi and len(gr) > 1 else " AND ".join(gr)
                       for gr in groups)


def cond_detail(side, key, cfg, s, st, i, K):
    px, a = s["c"][i], s["atr"][i]
    if key == "stoch":
        return f"Stoch %K {_g(s['kv'][i])}"
    if key == "breakout":
        return f"closed below breakout {_money(st['brk'][i])}"
    if key == "count":
        return (f"{st['cnt'][i]} of {cfg['n']} new ATH closes" if side == "entry"
                else f"only {st['cnt'][i]} new ATH closes in {K}D")
    if key in ("near", "drawdown"):
        word = "max" if side == "entry" else "limit"
        if cfg["unit"] == "atr":
            if a is None or a <= 0:
                return "ATR n/a"
            return f"{(st['lvl'][i] - px) / a:.1f}x ATR below ATH ({word} {_g(cfg['x'])}x)"
        return f"{st['below'][i]:.1f}% below ATH ({word} {_g(cfg['x'])}%)"
    if key == "recent":
        d = st["dsa"][i]
        return "no new ATH yet" if d is None else f"last new ATH {d}D ago (max {cfg['n']})"
    if key == "stale":
        d = st["dsa"][i]
        return "no new ATH" if d is None else f"no new ATH in {d}D (limit {cfg['n']})"
    return key


def why_text(s, st, i, rules):
    keys, side = st["why"][i], st["wk"][i]
    if not keys:
        return ""
    return "; ".join(cond_detail(side, k, rules[side][k], s, st, i, rules["window_days"]) for k in keys)


def filters_sentence(f):
    p = []
    if f["rsi_min"] is not None or f["rsi_max"] is not None:
        lo = "any" if f["rsi_min"] is None else f"{f['rsi_min']:g}"
        hi = "any" if f["rsi_max"] is None else f"{f['rsi_max']:g}"
        p.append(f"RSI {lo}-{hi}")
    if f["rvol_min"] is not None:
        p.append(f"RVOL >= {f['rvol_min']:g}")
    if f["ret_min"] is not None:
        p.append(f"day return >= {f['ret_min']:g}%")
    if f["macd"] != "any":
        p.append(f"MACD {'bullish' if f['macd'] == 'bull' else 'bearish'}")
    if f["stoch"] != "any":
        p.append(f"Stochastic {'bullish' if f['stoch'] == 'bull' else 'bearish'}")
    return "Indicator filters on new alerts: " + (", ".join(p) if p else "none")


# ============================================================
# EMAIL  (the page's emailText() builds the same text)
# ============================================================

def build_email(cal, prep, states, rules, a):
    K = rules["window_days"]
    day = pd.Timestamp(cal[a])
    day_s = f"{day:%b %d, %Y}"
    new, ended, running = [], [], []
    for t in sorted(prep):
        s, st = prep[t], states[t]
        is_new, _, is_end = events(s, st, a, rules)
        if is_new:
            new.append(t)
        elif is_end:
            ended.append(t)
        elif st["conf"][a] and s["c"][a] is not None:
            running.append(t)
    new.sort(key=lambda t: (-states[t]["cnt"][a], states[t]["below"][a] or 0, t))
    ended.sort(key=lambda t: (-states[t]["cnt"][a], t))
    running.sort(key=lambda t: (-states[t]["cnt"][a], states[t]["below"][a] or 0, t))

    def mmmdd(i):
        return f"{pd.Timestamp(cal[i]):%b %d}"

    def when(st, i):
        d = st["dsa"][i]
        if d is None:
            return "n/a"
        return "today" if d == 0 else f"{d} day{'s' if d != 1 else ''} ago ({mmmdd(i - d)})"

    L = ["=" * 60, f"  ALL-TIME HIGH SCANNER  -  {day_s} close", "=" * 60, "",
         f"Run starts when: {expr_text(rules, 'entry')}.",
         f"Run ends when: {expr_text(rules, 'exit')}.",
         filters_sentence(rules["filters"]), ""]

    L.append(f"NEW RUNS ({len(new)})")
    L.append("-" * 60)
    if not new:
        L.append("   none today")
    for n_, t in enumerate(new, 1):
        s, st = prep[t], states[t]
        px, at = s["c"][a], s["atr"][a]
        atrp = None if at is None or not px else at / px * 100
        start = "n/a" if st["start"][a] is None else mmmdd(st["start"][a])
        L.append(f"{n_}. {t} - {s['company']} ({s['sector']})")
        L.append(f"   New ATH closes ({K}D): {st['cnt'][a]}  |  last: {when(st, a)}  |  run start: {start}")
        L.append(f"   Close {_money(px)}  |  ATH {_money(st['lvl'][a])}  |  {_pct(st['below'][a], 1, False)} below  |  "
                 f"breakout {_money(st['brk'][a])} ({_pct(above_brk(s, st, a))} above)")
        L.append(f"   ATR {_money(at)} ({_pct(atrp, 1, False)} of price)  |  "
                 f"Stoch %K {_g(s['kv'][a])} / %D {_g(s['dv'][a])}")
        L.append(f"   Day {_pct(day_return(s['c'], a))}  |  RVOL {_g(s['v'][a])}x  |  RSI {_g(s['r'][a])}  |  "
                 f"MACD {'bullish' if s['m'][a] == 1 else 'bearish'}")
    L.append("")

    L.append(f"RUN ENDED - possible peak ({len(ended)})")
    L.append("-" * 60)
    if not ended:
        L.append("   none today")
    for n_, t in enumerate(ended, 1):
        s, st = prep[t], states[t]
        d = st["dsa"][a]
        L.append(f"{n_}. {t} - {s['company']} ({s['sector']})")
        L.append(f"   Peak close {_money(st['lvl'][a])}" + (f" on {mmmdd(a - d)}" if d is not None else "") +
                 f"  |  now {_money(s['c'][a])}, {_pct(st['below'][a], 1, False)} below  |  "
                 f"day {_pct(day_return(s['c'], a))}")
        L.append(f"   Why: {why_text(s, st, a, rules)}  |  new ATH closes ({K}D): {st['cnt'][a]}  |  "
                 f"Stoch %K {_g(s['kv'][a])}")
    L.append("")

    if rules["email"]["list_running"]:
        # Grouped by sector (most tickers first), each sector sorted by % gain since breakout.
        # Breakout date = the run's first new ATH close; gain = close vs the frozen breakout level.
        L.append(f"STILL RUNNING ({len(running)})")
        L.append("-" * 60)
        if not running:
            L.append("   none")
        else:
            by_sec = {}
            for t in running:
                by_sec.setdefault(prep[t]["sector"], []).append(t)
            sec_order = sorted(by_sec, key=lambda e: (-len(by_sec[e]), prep[by_sec[e][0]]["sectorName"]))
            L.append(f"   {'Ticker':<7}{'Breakout':<13}{'Gain':>8}{f'ATH hits {K}D':>14}")
            for etf in sec_order:
                ts = by_sec[etf]
                gains = {t: above_brk(prep[t], states[t], a) for t in ts}
                ts.sort(key=lambda t: (gains[t] is None, -(gains[t] or 0), t))
                L.append("")
                L.append(f"{prep[ts[0]]['sectorName']} ({etf}) - {len(ts)}")
                for t in ts:
                    si = states[t]["start"][a]
                    bd = "n/a" if si is None else f"{pd.Timestamp(cal[si]):%m-%d-%Y}"
                    L.append(f"   {t:<7}{bd:<13}{_pct(gains[t]):>8}{states[t]['cnt'][a]:>14}")
        L.append("")
    L.append("=" * 60)

    if new or ended:
        bits = []
        if new:
            bits.append(f"{len(new)} new ({', '.join(new[:8])}{', ...' if len(new) > 8 else ''})")
        if ended:
            bits.append(f"{len(ended)} ended ({', '.join(ended[:8])}{', ...' if len(ended) > 8 else ''})")
        subject = f"{SCANNER_NAME}: " + " · ".join(bits) + f" - {day_s}"
    else:
        subject = f"{SCANNER_NAME}: no new or ended runs ({len(running)} running) - {day_s}"

    when_ = rules["email"]["send_when"]
    send = when_ == "always" or (when_ == "new" and bool(new)) or (when_ == "changes" and bool(new or ended))
    return subject, "\n".join(L), send, {"new": new, "ended": ended, "running": running}


def send_email(subject, body):
    if not (EMAIL_USER and EMAIL_PASS and ALERT_TO):
        print("Email credentials not set (EMAIL_USER / EMAIL_PASS / ALERT_TO) - skipping send.")
        return
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_USER
    msg["To"] = ALERT_TO
    with smtplib.SMTP_SSL(os.environ.get("SMTP_HOST", "smtp.gmail.com"),
                          int(os.environ.get("SMTP_PORT", "465"))) as server:
        server.login(EMAIL_USER, EMAIL_PASS)
        server.sendmail(EMAIL_USER, [ALERT_TO], msg.as_string())
    print(f"Email sent to {ALERT_TO}")


def run_alerts():
    rules, _ = load_rules()
    sp500 = get_sp500()
    print(f"Loaded {len(sp500)} S&P 500 tickers. Downloading full history (period='max')...")
    data = download(sp500["Ticker"].tolist())
    end = pd.DatetimeIndex(data.index).max()
    # a long lead-in so a run that started months ago is still tracked as running today
    cal, s0, prep = prepare(data, sp500, end - pd.DateOffset(months=BACKTEST_MONTHS))
    states = {t: compute_states(s, rules) for t, s in prep.items()}
    subject, body, send, _ = build_email(cal, prep, states, rules, len(cal) - 1)
    print(subject)
    print(body)
    if send:
        send_email(subject, body)
    else:
        print(f"Nothing to send (send_when = {rules['email']['send_when']}).")


# ============================================================
# DASHBOARD
# ============================================================

def github_links(rules_exists):
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not repo:
        return None
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    branch = os.environ.get("GITHUB_REF_NAME") or "main"
    wf = (os.environ.get("GITHUB_WORKFLOW_REF") or "").split("@")[0].rsplit("/", 1)[-1]
    rules_url = (f"{server}/{repo}/edit/{branch}/{RULES_FILE_NAME}" if rules_exists
                 else f"{server}/{repo}/new/{branch}?filename={RULES_FILE_NAME}")
    return {
        "repo": repo,
        "rules": rules_url,
        "rulesExists": rules_exists,
        "run": f"{server}/{repo}/actions/workflows/{wf}" if wf else f"{server}/{repo}/actions",
    }


def write_csv(cal, s0, prep, states, rules):
    K, rows = rules["window_days"], []
    for t in sorted(prep):
        s, st = prep[t], states[t]
        for i in range(s0, len(cal)):
            if s["c"][i] is None:
                continue
            is_new, filtered, is_end = events(s, st, i, rules)
            if not (st["cnt"][i] or st["conf"][i] or is_end):
                continue
            ab = above_brk(s, st, i)
            row = {
                "Date": f"{pd.Timestamp(cal[i]):%Y-%m-%d}", "Ticker": t, "Company": s["company"],
                "Sector_ETF": s["sector"], "Close": s["c"][i], "Is_New_ATH": st["ath"][i],
                "ATH_Close": st["lvl"][i], "Pct_Below_ATH": round(st["below"][i], 2),
                f"ATH_Days_{K}D": st["cnt"][i], "Days_Since_ATH": st["dsa"][i],
                "Breakout_Level": st["brk"][i], "Pct_Above_Breakout": None if ab is None else round(ab, 2),
                "Run_Start": None if st["start"][i] is None else f"{pd.Timestamp(cal[st['start'][i]]):%Y-%m-%d}",
                "In_Run": st["conf"][i], "New_Alert": is_new, "New_Run_Filtered_Out": filtered, "Run_Ended": is_end,
                "Entry_Not_Met": why_text(s, st, i, rules) if st["wk"][i] == "entry" and not st["conf"][i] else "",
                "Exit_Triggered": why_text(s, st, i, rules) if is_end else "",
                "Day_Return%": _num(day_return(s["c"], i), 2), "RVOL": s["v"][i], "RSI": s["r"][i],
                "ATR": s["atr"][i], "Stoch_K": s["kv"][i], "Stoch_D": s["dv"][i],
                "MACD_Bull": s["m"][i] == 1, "Stoch_Bull": s["k"][i] == 1,
            }
            for n in FORWARD_DAYS:
                row[f"Fwd_{n}D%"] = _num(fwd_return(s["c"], i, n), 2)
            rows.append(row)
    pd.DataFrame(rows).to_csv(CSV_OUT, index=False)


def build_report(cal, s0, prep, rules, rules_exists):
    from plotly.offline import get_plotlyjs, get_plotlyjs_version

    OUT_DIR.mkdir(parents=True, exist_ok=True)   # site/scanners/... doesn't exist on a fresh runner

    # only tickers that made at least one new ATH close in the calendar matter here
    states = {t: compute_states(s, rules) for t, s in prep.items()}
    keep = sorted(t for t in prep if any(states[t]["ath"]))
    prep = {t: prep[t] for t in keep}
    states = {t: states[t] for t in keep}
    write_csv(cal, s0, prep, states, rules)

    dates = [f"{pd.Timestamp(d):%Y-%m-%d}" for d in cal]
    shown = pd.DatetimeIndex(cal[s0:])
    holidays = [f"{d:%Y-%m-%d}" for d in pd.bdate_range(shown.min(), shown.max()).difference(shown)]
    data = {
        "dates": dates, "s0": s0, "holidays": holidays, "tickers": keep,
        "company": {t: prep[t]["company"] for t in keep},
        "sector": {t: prep[t]["sector"] for t in keep},
        "sectorName": {prep[t]["sector"]: prep[t]["sectorName"] for t in keep},
        "s": {t: {k: prep[t][k] for k in ("a0", "c", "r", "v", "atr", "kv", "dv", "m", "k")} for t in keep},
        "rules": rules, "defaults": normalize_rules(DEFAULT_RULES), "spec": COND_SPEC,
        "rulesFile": RULES_FILE_NAME, "rulesExists": rules_exists, "gh": github_links(rules_exists),
        "maxWindow": MAX_WINDOW, "fwdDays": FORWARD_DAYS, "colorDays": COLOR_FWD_DAYS, "cap": COLOR_CAP_PCT,
        "gradient": GRADIENT, "rvolScale": RVOL_SCALE, "rvolRange": RVOL_RANGE, "pending": PENDING_COLOR,
        "heatDays": HEATMAP_DAYS, "heatRows": HEATMAP_ROWS, "scannerName": SCANNER_NAME,
    }

    if PLOTLY_JS == "inline":
        plotly_tag = f"<script>{get_plotlyjs()}</script>"
    else:
        plotly_tag = f'<script src="https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"></script>'

    html = (PAGE_TEMPLATE
            .replace("__CSV__", CSV_OUT.name)
            .replace("__SURFACE__", SURFACE).replace("__INK2__", INK_2)
            .replace("__INK__", INK).replace("__GRID__", GRID))
    html = html.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    html = html.replace("__PLOTLY__", plotly_tag)
    HTML_OUT.write_text(html, encoding="utf-8")

    a = len(cal) - 1
    subject, _, _, groups = build_email(cal, prep, states, rules, a)
    if os.environ.get("BACKTEST_OUTPUT_DIR"):
        (OUT_DIR / "scanner.json").write_text(json.dumps({
            "title": SCANNER_TITLE, "order": SCANNER_ORDER, "page": HTML_OUT.name,
            "subtitle": f"Data through {pd.Timestamp(cal[a]):%b %d, %Y} · {len(groups['running']) + len(groups['new'])} "
                        f"runs in progress · {len(groups['new'])} new · {len(groups['ended'])} ended today",
        }, indent=2), encoding="utf-8")
    return subject


PAGE_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>ATH Runs Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
__PLOTLY__
<style>
  body { background:__SURFACE__; color:__INK__; font-family:Inter,'Segoe UI',Arial,sans-serif; margin:0; padding:24px 16px; }
  .wrap { max-width:1320px; margin:0 auto; }
  h1 { font-size:22px; margin:0 0 4px; }
  .sub { color:__INK2__; font-size:13px; margin-bottom:16px; line-height:1.55; }
  .tiles { display:flex; flex-wrap:wrap; gap:12px; margin:0 0 16px; }
  .tile { border:1px solid __GRID__; border-radius:8px; padding:10px 16px; min-width:120px; background:#fff; }
  .tile .v { font-size:22px; font-weight:600; font-variant-numeric:tabular-nums; }
  .tile .l { font-size:12px; color:__INK2__; }
  .card { border:1px solid __GRID__; border-radius:10px; padding:10px 12px; margin-bottom:18px; background:#fff; }
  h2 { font-size:16px; font-weight:600; margin:4px 0 8px; }
  h3 { font-size:13px; font-weight:600; margin:12px 0 6px; color:__INK2__; text-transform:uppercase; letter-spacing:.04em; }
  .panel { border:1px solid __GRID__; border-radius:10px; padding:12px 14px; margin-bottom:16px; background:#fff;
           position:sticky; top:0; z-index:5; }
  .row { display:flex; flex-wrap:wrap; align-items:center; gap:10px 16px; }
  .ctl { display:flex; align-items:center; gap:6px; font-size:13px; color:__INK2__; }
  .ctl input, .ctl select { font:inherit; font-size:14px; padding:6px 8px; border:1px solid #c9c8c2; border-radius:6px; background:#fff; color:__INK__; }
  #tickerInput { width:220px; }
  button, a.btn { font:inherit; font-size:13px; padding:7px 12px; border:1px solid #c9c8c2; border-radius:6px; background:#fff;
                  color:__INK__; cursor:pointer; text-decoration:none; display:inline-block; }
  button:hover, a.btn:hover { background:#f0efec; }
  button.primary, a.primary { background:__INK__; color:#fff; border-color:__INK__; }
  button.primary:hover, a.primary:hover { background:#333; }
  .hint, .note { font-size:12px; color:__INK2__; line-height:1.5; }
  .note { margin:8px 0 2px; }
  .grid2 { display:grid; grid-template-columns:minmax(340px, 580px) 1fr; gap:18px; }
  @media (max-width:980px) { .grid2 { grid-template-columns:1fr; } }
  .form { display:grid; grid-template-columns:auto 1fr; gap:8px 10px; align-items:center; font-size:13px; }
  .form label { color:__INK2__; }
  input[type=number], .form select, .cond select { font:inherit; font-size:13px; padding:4px 6px; border:1px solid #c9c8c2;
         border-radius:6px; background:#fff; color:__INK__; }
  .form input[type=number] { width:80px; }
  .form span input[type=number] { width:62px; }
  .side { border:1px solid __GRID__; border-left:5px solid; border-radius:8px; padding:8px 12px 10px; margin:10px 0; }
  .side.entry { border-left-color:#0b5a24; background:#fbfdfb; }
  .side.exit { border-left-color:#8e1b1b; background:#fffbfa; }
  .side .ttl { font-size:13px; font-weight:700; letter-spacing:.04em; text-transform:uppercase; }
  .side.entry .ttl { color:#0b5a24; } .side.exit .ttl { color:#8e1b1b; }
  .side .what { font-size:12px; color:__INK2__; margin:2px 0 6px; }
  .cond { display:flex; align-items:center; gap:8px; padding:6px 0; border-top:1px dashed __GRID__; font-size:13px; }
  .cond .jw { width:62px; flex:none; }
  .cond .jw .ifl { display:none; font-size:12px; font-weight:600; color:__INK2__; padding-left:6px; }
  .cond.first .jw select { display:none; } .cond.first .jw .ifl { display:inline; }
  .cond input[type=checkbox] { width:16px; height:16px; flex:none; margin:0; }
  .cond .ctext { line-height:2; }
  .cond .ctext input[type=number] { width:58px; }
  .cond.off .ctext, .cond.off .jw { opacity:.42; }
  .expr { font-size:12px; background:#f3f2ee; border-radius:6px; padding:6px 8px; margin-top:6px; line-height:1.45; }
  .expr.bad { background:#fff6e0; color:#6b4e00; }
  .chip { display:inline-block; font-size:12px; padding:3px 9px; border-radius:999px; border:1px solid; margin-left:8px; vertical-align:2px; }
  .chip.ok { color:#0b5a24; border-color:#9fd3ad; background:#eef8f0; }
  .chip.edit { color:#6b4e00; border-color:#f0d58a; background:#fff6e0; }
  pre#emailBody { background:#f7f6f2; border:1px solid __GRID__; border-radius:8px; padding:12px; font-size:12px;
                  line-height:1.45; max-height:620px; overflow:auto; white-space:pre-wrap; margin:6px 0 0; }
  .subject { font-size:13px; font-weight:600; padding:8px 10px; border:1px solid __GRID__; border-radius:8px; background:#fff; }
  .sendflag { font-size:12px; margin:6px 0 0; }
  .btns { display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }
  table { border-collapse:collapse; width:100%; font-size:13px; }
  th, td { text-align:left; padding:6px 8px; border-bottom:1px solid __GRID__; white-space:nowrap; }
  th { color:__INK2__; font-weight:500; }
  td.why { white-space:normal; min-width:220px; max-width:360px; }
  .num { text-align:right; font-variant-numeric:tabular-nums; }
  tr.click { cursor:pointer; }
  tr.click:hover td { background:#f5f4f0; }
  tr.all td { font-weight:600; background:#f7f6f2; }
  tr.sel td { background:#eef4fb; }
  .tscroll { overflow-x:auto; }
  .sw { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; vertical-align:-1px; border:1px solid rgba(0,0,0,.08); }
  .st { font-size:12px; padding:2px 8px; border-radius:999px; }
  .st.new { background:#0b5a24; color:#fff; } .st.run { background:#dcefe1; color:#0b5a24; }
  .st.end { background:#fbe3e1; color:#8e1b1b; } .st.no { background:#efeeea; color:__INK2__; }
  .warn { background:#fff6e0; border:1px solid #f0d58a; border-radius:6px; padding:6px 10px; color:#6b4e00; font-size:13px; margin:6px 0; }
  .toast { position:fixed; bottom:20px; left:50%; transform:translateX(-50%); background:__INK__; color:#fff; padding:8px 14px;
           border-radius:8px; font-size:13px; opacity:0; transition:opacity .2s; pointer-events:none; z-index:20; }
  .toast.on { opacity:1; }
  a { color:#2a78d6; }
</style></head><body><div class="wrap">
<h1>All-time-high runs</h1>
<div class="sub" id="sub"></div>

<div class="panel">
  <div class="row">
    <label class="ctl">As of <select id="asOf"></select></label>
    <label class="ctl">Sector <select id="sector"><option value="">All sectors</option></select></label>
    <label class="ctl">Tickers <input id="tickerInput" list="tickerList" autocomplete="off" placeholder="e.g. AMD  or  AMD, NVDA"></label>
    <datalist id="tickerList"></datalist>
    <button id="applyBtn">Apply</button>
    <button id="clearBtn">Clear</button>
    <span class="hint">Applies to every view below. The email always uses all S&amp;P 500 tickers.</span>
  </div>
</div>

<div class="card" id="emailCard">
  <h2>Run rules + email alert setup <span class="chip ok" id="ruleChip">Matches the email rules</span></h2>
  <div class="grid2">
    <div>
      <div class="form">
        <label for="r_window">Window (trading days)</label><input id="r_window" type="number" min="2" step="1">
      </div>
      <div class="hint" style="margin-top:4px">Used by "new ATH closes in the window" and to find the breakout level. Each condition is
        joined to the one above it with AND or OR; AND is applied before OR. ATR = the ticker's own 14-day average true range.</div>

      <div class="side entry">
        <div class="ttl">Entry · when a run starts</div>
        <div class="what">Checked only while a ticker is NOT in a run. Make these strict.</div>
        <div id="cond_entry"></div>
        <div class="expr" id="expr_entry"></div>
      </div>

      <div class="side exit">
        <div class="ttl">Exit · when a run ends (possible peak)</div>
        <div class="what">Checked only while a ticker IS in a run. The breakout level is frozen when the run starts.</div>
        <div id="cond_exit"></div>
        <div class="expr" id="expr_exit"></div>
      </div>

      <h3>Indicator filters on new-run alerts <span class="hint" style="text-transform:none;letter-spacing:0">(blank = off; they don't change runs)</span></h3>
      <div class="form">
        <label>RSI</label><span><input id="f_rsimin" type="number" step="1" placeholder="min"> to <input id="f_rsimax" type="number" step="1" placeholder="max"></span>
        <label for="f_rvol">Min RVOL</label><input id="f_rvol" type="number" step="0.1" placeholder="any">
        <label for="f_ret">Min day return %</label><input id="f_ret" type="number" step="0.5" placeholder="any">
        <label for="f_macd">MACD</label><select id="f_macd"><option value="any">any</option><option value="bull">bullish</option><option value="bear">bearish</option></select>
        <label for="f_stoch">Stochastic</label><select id="f_stoch"><option value="any">any</option><option value="bull">bullish</option><option value="bear">bearish</option></select>
      </div>
      <h3>Email</h3>
      <div class="form">
        <label for="e_when">Send when</label><select id="e_when" style="width:auto">
          <option value="changes">new or ended runs</option><option value="new">new runs only</option><option value="always">every day</option></select>
        <label for="e_run">List still-running</label><select id="e_run"><option value="1">yes</option><option value="0">no</option></select>
      </div>
      <div class="btns">
        <button id="resetRules">Reset to email rules</button>
        <button id="restoreRules" style="display:none">Restore my last edits</button>
        <button id="defaultRules">Starting defaults</button>
      </div>
    </div>
    <div>
      <div class="subject" id="emailSubject"></div>
      <div class="sendflag" id="sendFlag"></div>
      <pre id="emailBody"></pre>
      <div class="btns">
        <button class="primary" id="copyRules">Copy rules JSON</button>
        <button id="dlRules">Download rules file</button>
        <a class="btn" id="ghRules" target="_blank" rel="noopener" style="display:none">Open rules file on GitHub</a>
        <a class="btn" id="ghRun" target="_blank" rel="noopener" style="display:none">Run workflow now</a>
        <button id="copyEmail">Copy email text</button>
      </div>
      <div class="note" id="pushHow"></div>
    </div>
  </div>
</div>

<div class="tiles" id="tiles"></div>

<div class="card">
  <h2 id="lbTitle">ATH runs leaderboard</h2>
  <div class="tscroll"><table id="lbTable"></table></div>
  <div class="note" id="lbNote"></div>
</div>

<div class="card">
  <h2 id="rtTitle">Run tracker</h2>
  <div class="row" style="margin-bottom:6px">
    <label class="ctl">Ticker <input id="rtTicker" list="tickerList" autocomplete="off" style="width:110px"></label>
    <label class="ctl">Bar color <select id="rtColor"></select></label>
    <span class="hint" id="rtHint"></span>
  </div>
  <div id="rtSummary" class="note"></div>
  <div id="rtChart"></div>
  <div class="note">Bars = new all-time closing highs in the trailing window (30 bars tall means a new ATH every day for 30 days).
    ▲ = run started (entry), ▼ = run ended (exit). Black line = close, green dotted = ATH, gray dashed = breakout level (frozen for the run),
    red dotted = the drawdown exit level while a run is on.</div>
</div>

<div class="card">
  <h2 id="hmTitle">Run heatmap</h2>
  <div class="row" style="margin-bottom:6px">
    <label class="ctl">Cell color <select id="hmColor"><option value="below">% below ATH</option><option value="rvol">Relative volume (RVOL)</option></select></label>
    <label class="ctl">Volume rings <select id="hmRing"><option value="1.5">RVOL ≥ 1.5x</option><option value="2">RVOL ≥ 2x</option><option value="0">off</option></select></label>
  </div>
  <div id="hmChart"></div>
  <div class="note" id="hmNote"></div>
</div>

<div class="card">
  <h2>Backtest - how the hits did afterwards</h2>
  <div class="row" style="margin-bottom:8px">
    <label class="ctl">Hits <select id="hitType">
      <option value="new">Email alerts (new runs, after filters)</option>
      <option value="conf">Every day a ticker is in a run</option>
      <option value="ath">Any new ATH close (no entry rules)</option>
      <option value="end">Run ended (possible peak)</option></select></label>
    <label class="ctl">Forward return <select id="fwdN"></select></label>
    <label class="ctl">Show <select id="fwdSign"><option value="all">all hits</option><option value="pos">positive only</option><option value="neg">negative only</option></select></label>
  </div>
  <div id="btWarn"></div>
  <div class="tiles" id="btTiles"></div>
  <h3>Daily hits</h3><div id="dailyChart"></div>
  <h3>Weekly hits (segment = days that ticker hit that week)</h3><div id="weeklyChart"></div>
  <h3>Forward returns by sector</h3>
  <div class="tscroll"><table id="sectorTable"></table></div>
  <h3>Most frequent tickers</h3>
  <div class="tscroll"><table id="topTable"></table></div>
  <div class="note">Close-to-close returns. Hits too recent for a forward result are hatched and left out of averages.
    Uses today's S&amp;P 500 members (survivorship bias), so treat returns as indicative.
    <a href="__CSV__" download>Download every ATH-run day (CSV)</a> - computed with the email rules.</div>
</div>
</div>
<div class="toast" id="toast"></div>

<script>
const D = __DATA__;
const SURF = '__SURFACE__', INK = '__INK__', INK2 = '__INK2__', GRIDC = '__GRID__';
const DAY = 86400000, $ = id => document.getElementById(id);
const T = D.tickers, S = D.s, NDAYS = D.dates.length, S0 = D.s0, LAST = NDAYS - 1;
const SIDES = ['entry', 'exit'];

// ---------- formatting ----------
const hexRgb = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
function interp(stops, x) {        // stops: [[pos, hex], ...] sorted by pos
  if (x <= stops[0][0]) return stops[0][1];
  for (let k = 0; k < stops.length - 1; k++) {
    const [p0, c0] = stops[k], [p1, c1] = stops[k + 1];
    if (x <= p1) { const f = (x - p0) / (p1 - p0), a = hexRgb(c0), b = hexRgb(c1);
      return '#' + a.map((v, j) => Math.round(v + (b[j] - v) * f).toString(16).padStart(2, '0')).join(''); }
  }
  return stops[stops.length - 1][1];
}
const grad = x => interp(D.gradient, Math.max(-1, Math.min(1, x)));
const colorRet = v => (v === null || v === undefined || Number.isNaN(v)) ? D.pending : grad(v / D.cap);
const colorBelow = p => p === null || p === undefined ? D.pending : grad(1 - 2 * Math.min(p, D.cap) / D.cap);
const inkOn = x => (x === null || Math.abs(x) < 0.4) ? INK : '#ffffff';
const pct = (v, d = 2) => v === null || v === undefined ? 'n/a' : (v > 0 ? '+' : '') + v.toFixed(d) + '%';
const pctU = (v, d = 1) => v === null || v === undefined ? 'n/a' : v.toFixed(d) + '%';
const money = v => v === null || v === undefined ? 'n/a' : '$' + v.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
const g = v => v === null || v === undefined ? 'n/a' : (Math.round(v * 1e6) / 1e6).toString();
const mean = a => a.length ? a.reduce((s, v) => s + v, 0) / a.length : null;
function median(a) { if (!a.length) return null; const s = [...a].sort((x, y) => x - y), m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; }
const dt = i => new Date(D.dates[i] + 'T00:00:00Z');
const fmtDate = (i, o) => dt(i).toLocaleDateString('en-US', Object.assign({timeZone: 'UTC'}, o || {month: 'short', day: '2-digit', year: 'numeric'}));
const mmmdd = i => fmtDate(i, {month: 'short', day: '2-digit'});
function toast(msg) { const t = $('toast'); t.textContent = msg; t.classList.add('on'); setTimeout(() => t.classList.remove('on'), 1800); }
function parseTickers(v) { return v.toUpperCase().split(/[\s,;]+/).map(x => x.replace('.', '-')).filter(Boolean); }

// ---------- rule editor ----------
const COND_UI = {
  entry: {
    count: 'At least {n} new ATH closes in the window',
    recent: 'Latest new ATH within {n} trading days <span class="hint">(0 = today)</span>',
    near: 'Close within {x}{unit} of the ATH',
    breakout: 'Hold the breakout: never {n} straight close(s) more than {x}{unit} below it',
    stoch: 'Stochastic %K {op} {x}'},
  exit: {
    drawdown: 'Close more than {x}{unit} below the ATH',
    stale: 'No new ATH close for more than {n} trading days',
    breakout: '{n} straight close(s) more than {x}{unit} below the breakout',
    count: 'Fewer than {n} new ATH closes in the window',
    stoch: 'Stochastic %K {op} {x}'}};
function condRowHtml(side, key) {
  const id = `${side}_${key}`, step = key === 'stoch' ? 1 : 0.5;
  const txt = COND_UI[side][key]
    .replace('{n}', `<input type="number" id="${id}_n" step="1" min="0">`)
    .replace('{x}', `<input type="number" id="${id}_x" step="${step}" min="0">`)
    .replace('{unit}', ` <select id="${id}_unit"><option value="pct">%</option><option value="atr">× ATR</option></select>`)
    .replace('{op}', `<select id="${id}_op"><option value="ge">≥</option><option value="le">≤</option>` +
      `<option value="kd_up">above %D</option><option value="kd_dn">below %D</option></select>`);
  return `<div class="cond" id="row_${id}"><span class="jw"><span class="ifl">IF</span><select id="${id}_join">` +
    `<option value="and">AND</option><option value="or">OR</option></select></span>` +
    `<input type="checkbox" id="${id}_on" title="switch this condition on/off"><span class="ctext">${txt}</span></div>`;
}
function buildEditor() { SIDES.forEach(side => { $('cond_' + side).innerHTML = Object.keys(D.spec[side]).map(k => condRowHtml(side, k)).join(''); }); }

const numOrNull = id => { const el = $(id); if (!el) return null; const v = el.value.trim(); return v === '' || Number.isNaN(+v) ? null : +v; };
const lim = (v, K) => v === 'K' ? K : v === 'K-1' ? K - 1 : v;
const numval = (v, d) => (v === null || v === undefined || v === '' || typeof v === 'boolean' || Number.isNaN(+v)) ? d : +v;
function clampRules(raw) {      // mirrors normalize_rules() in Python
  const r = JSON.parse(JSON.stringify(D.defaults));
  const K = Math.min(Math.max(Math.round(numval(raw.window_days, r.window_days)), 2), D.maxWindow);
  r.window_days = K;
  SIDES.forEach(side => {
    const src = raw[side] || {};
    Object.entries(D.spec[side]).forEach(([key, spec]) => {
      const c = r[side][key], s = src[key] || {};
      c.on = s.on === undefined ? c.on : !!s.on;
      c.join = (s.join === undefined ? c.join : s.join) === 'or' ? 'or' : 'and';
      if (spec.n) c.n = Math.min(Math.max(Math.round(numval(s.n, c.n)), lim(spec.n[0], K)), lim(spec.n[1], K));
      if (spec.x) { const v = Math.max(numval(s.x, c.x), spec.x[0]); c.x = spec.x[1] === null ? v : Math.min(v, spec.x[1]); }
      if (spec.unit) c.unit = (s.unit === undefined ? c.unit : s.unit) === 'atr' ? 'atr' : 'pct';
      if (spec.op) { const op = s.op === undefined ? c.op : s.op; c.op = ['ge', 'le', 'kd_up', 'kd_dn'].includes(op) ? op : c.op; }
    });
  });
  ['filters', 'email'].forEach(sec => { if (raw[sec]) Object.keys(r[sec]).forEach(k => { if (k in raw[sec]) r[sec][k] = raw[sec][k]; }); });
  ['rsi_min', 'rsi_max', 'rvol_min', 'ret_min'].forEach(k => { r.filters[k] = numval(r.filters[k], null); });
  ['macd', 'stoch'].forEach(k => { if (!['any', 'bull', 'bear'].includes(r.filters[k])) r.filters[k] = 'any'; });
  if (!['changes', 'new', 'always'].includes(r.email.send_when)) r.email.send_when = 'changes';
  r.email.list_running = !!r.email.list_running;
  return r;
}
function readRules() {
  const raw = {window_days: numOrNull('r_window'), entry: {}, exit: {}};
  SIDES.forEach(side => Object.entries(D.spec[side]).forEach(([key, spec]) => {
    const id = `${side}_${key}`, c = {on: $(id + '_on').checked, join: $(id + '_join').value};
    if (spec.n) c.n = numOrNull(id + '_n'); if (spec.x) c.x = numOrNull(id + '_x');
    if (spec.unit) c.unit = $(id + '_unit').value; if (spec.op) c.op = $(id + '_op').value;
    raw[side][key] = c;
  }));
  raw.filters = {rsi_min: numOrNull('f_rsimin'), rsi_max: numOrNull('f_rsimax'), rvol_min: numOrNull('f_rvol'),
    ret_min: numOrNull('f_ret'), macd: $('f_macd').value, stoch: $('f_stoch').value};
  raw.email = {send_when: $('e_when').value, list_running: $('e_run').value === '1'};
  return clampRules(raw);
}
function setRules(r) {
  const s = (id, v) => { $(id).value = v === null || v === undefined ? '' : v; };
  s('r_window', r.window_days);
  SIDES.forEach(side => Object.entries(D.spec[side]).forEach(([key, spec]) => {
    const id = `${side}_${key}`, c = r[side][key];
    $(id + '_on').checked = c.on; $(id + '_join').value = c.join;
    if (spec.n) s(id + '_n', c.n); if (spec.x) s(id + '_x', c.x);
    if (spec.unit) $(id + '_unit').value = c.unit; if (spec.op) $(id + '_op').value = c.op;
  }));
  s('f_rsimin', r.filters.rsi_min); s('f_rsimax', r.filters.rsi_max); s('f_rvol', r.filters.rvol_min);
  s('f_ret', r.filters.ret_min); $('f_macd').value = r.filters.macd; $('f_stoch').value = r.filters.stoch;
  $('e_when').value = r.email.send_when; $('e_run').value = r.email.list_running ? '1' : '0';
}
function styleEditor() {     // dim off rows, show IF on the first switched-on row, show the logic sentence
  SIDES.forEach(side => {
    let first = true;
    Object.keys(D.spec[side]).forEach(key => {
      const id = `${side}_${key}`, row = $('row_' + id), on = RULES[side][key].on;
      row.classList.toggle('off', !on); row.classList.toggle('first', on && first); if (on) first = false;
      const op = $(id + '_op'); if (op) $(id + '_x').disabled = op.value.startsWith('kd');
    });
    const e = exprText(side), el = $('expr_' + side);
    el.textContent = (side === 'entry' ? 'Run starts when: ' : 'Run ends when: ') + e;
    el.classList.toggle('bad', enabled(side).length === 0);
  });
}
const rulesKey = r => JSON.stringify(r);

// ---------- run logic (mirrors compute_states() in the Python script) ----------
const enabled = side => Object.keys(D.spec[side]).filter(k => RULES[side][k].on).map(k => [k, RULES[side][k]]);
function evalExpr(items, bits) {
  if (!items.length) return false;
  let result = false, acc = null;
  items.forEach(([, cfg], idx) => { const b = bits[idx];
    if (idx === 0) acc = b; else if (cfg.join === 'or') { result = result || acc; acc = b; } else acc = acc && b; });
  return result || acc;
}
function stochOk(op, thr, k, d) {
  if (op === 'ge') return k !== null && k >= thr;
  if (op === 'le') return k !== null && k <= thr;
  if (k === null || d === null) return false;
  return op === 'kd_up' ? k > d : k < d;
}
function belowBrk(s, j, brk, cfg) {
  const c = s.c, a = s.atr;
  if (c[j] === null || brk === null) return false;
  if (cfg.unit === 'atr') return a[j] !== null && c[j] < brk - cfg.x * a[j];
  return c[j] < brk * (1 - cfg.x / 100);
}
function entryPass(key, cfg, s, st, i) {
  const px = s.c[i], a = s.atr[i];
  if (key === 'count') return st.cnt[i] >= cfg.n;
  if (key === 'recent') return st.dsa[i] !== null && st.dsa[i] <= cfg.n;
  if (key === 'near') return cfg.unit === 'atr' ? (a !== null && (st.lvl[i] - px) <= cfg.x * a) : st.below[i] <= cfg.x;
  if (key === 'breakout') {
    const f = st.first[i]; if (f === null) return true;
    const b = st.pmb[f]; let run = 0;
    for (let j = f; j <= i; j++) { if (s.c[j] === null) continue;
      if (belowBrk(s, j, b, cfg)) { run++; if (run >= cfg.n) return false; } else run = 0; }
    return true;
  }
  if (key === 'stoch') return stochOk(cfg.op, cfg.x, s.kv[i], s.dv[i]);
  return false;
}
function exitHit(key, cfg, s, st, i, rbrk) {
  const px = s.c[i], a = s.atr[i];
  if (key === 'drawdown') return cfg.unit === 'atr' ? (a !== null && (st.lvl[i] - px) > cfg.x * a) : st.below[i] > cfg.x;
  if (key === 'stale') return st.dsa[i] === null || st.dsa[i] > cfg.n;
  if (key === 'breakout') {
    if (rbrk === null) return false;
    let need = cfg.n, j = i;
    while (j >= 0 && need > 0) { if (s.c[j] !== null) { if (!belowBrk(s, j, rbrk, cfg)) return false; need--; } j--; }
    return need === 0;
  }
  if (key === 'count') return st.cnt[i] < cfg.n;
  if (key === 'stoch') return stochOk(cfg.op, cfg.x, s.kv[i], s.dv[i]);
  return false;
}
function computeTicker(s) {
  const c = s.c, n = c.length, K = RULES.window_days, E = enabled('entry'), X = enabled('exit');
  const st = {ath: Array(n).fill(false), cnt: Array(n).fill(0), conf: Array(n).fill(false)};
  for (const k of ['lvl', 'below', 'dsa', 'brk', 'first', 'start', 'why', 'wk', 'pmb']) st[k] = Array(n).fill(null);
  let pm = s.a0, last = null, inrun = false, rstart = null, rbrk = null;
  for (let i = 0; i < n; i++) {
    const x = c[i];
    st.pmb[i] = pm;
    if (x !== null) {
      const a = pm !== null && x > pm;
      st.ath[i] = a;
      if (pm === null || x > pm) pm = x;
      st.lvl[i] = pm; st.below[i] = (1 - x / pm) * 100;
      if (a) last = i;
    }
    let k = 0, f = null;
    for (let j = Math.max(0, i - K + 1); j <= i; j++) if (st.ath[j]) { k++; if (f === null) f = j; }
    st.cnt[i] = k; st.first[i] = f;
    st.dsa[i] = last === null ? null : i - last;
    if (x === null) { if (i > 0) for (const key of ['conf', 'brk', 'start', 'why', 'wk']) st[key][i] = st[key][i - 1]; continue; }
    const wbrk = f !== null ? st.pmb[f] : null;
    if (inrun) {
      const bits = X.map(([key, cfg]) => exitHit(key, cfg, s, st, i, rbrk));
      st.brk[i] = rbrk; st.start[i] = rstart;
      if (evalExpr(X, bits)) { inrun = false; st.why[i] = X.filter((_, q) => bits[q]).map(([key]) => key); st.wk[i] = 'exit'; }
    } else {
      const bits = E.map(([key, cfg]) => entryPass(key, cfg, s, st, i));
      if (evalExpr(E, bits)) { inrun = true; rstart = f !== null ? f : i; rbrk = wbrk; st.brk[i] = rbrk; st.start[i] = rstart; }
      else { st.brk[i] = wbrk; st.start[i] = f; st.why[i] = E.filter((_, q) => !bits[q]).map(([key]) => key); st.wk[i] = 'entry'; }
    }
    st.conf[i] = inrun;
  }
  return st;
}
const dayRet = (c, i) => (i < 1 || c[i] === null || c[i - 1] === null) ? null : (c[i] / c[i - 1] - 1) * 100;
const fwdRet = (c, i, n) => (i + n >= c.length || c[i] === null || c[i + n] === null) ? null : (c[i + n] / c[i] - 1) * 100;
const aboveBrk = (s, st, i) => { const b = st.brk[i], px = s.c[i]; return (b === null || px === null || b <= 0) ? null : (px / b - 1) * 100; };
function passesFilters(s, i, f) {
  const ge = (v, x) => v !== null && v !== undefined && v >= x, le = (v, x) => v !== null && v !== undefined && v <= x;
  if (f.rsi_min !== null && !ge(s.r[i], f.rsi_min)) return false;
  if (f.rsi_max !== null && !le(s.r[i], f.rsi_max)) return false;
  if (f.rvol_min !== null && !ge(s.v[i], f.rvol_min)) return false;
  if (f.ret_min !== null && !ge(dayRet(s.c, i), f.ret_min)) return false;
  if (f.macd !== 'any' && s.m[i] !== (f.macd === 'bull' ? 1 : 0)) return false;
  if (f.stoch !== 'any' && s.k[i] !== (f.stoch === 'bull' ? 1 : 0)) return false;
  return true;
}
function events(s, st, i, rules) {
  if (i < 1 || s.c[i] === null) return [false, false, false];
  const became = st.conf[i] && !st.conf[i - 1], ended = !st.conf[i] && st.conf[i - 1];
  const ok = passesFilters(s, i, rules.filters);
  return [became && ok, became && !ok, ended];
}

let RULES = null, ST = {}, STKEY = '';
function ensureStates() {
  const key = rulesKey([RULES.window_days, RULES.entry, RULES.exit]);
  if (key === STKEY) return;
  ST = {}; T.forEach(t => { ST[t] = computeTicker(S[t]); }); STKEY = key;
}

// ---------- text (mirrors the Python TEXT section) ----------
const unitTxt = cfg => cfg.unit === 'pct' ? '%' : 'x ATR';
function stochDesc(cfg) {
  if (cfg.op === 'ge') return `Stoch %K >= ${g(cfg.x)}`;
  if (cfg.op === 'le') return `Stoch %K <= ${g(cfg.x)}`;
  return cfg.op === 'kd_up' ? 'Stoch %K above %D' : 'Stoch %K below %D';
}
function condDesc(side, key, cfg, K) {
  if (key === 'stoch') return stochDesc(cfg);
  const times = cfg.n > 1 ? `${cfg.n} straight closes` : 'a close', tol = cfg.x > 0 ? ` by more than ${g(cfg.x)}${unitTxt(cfg)}` : '';
  if (side === 'entry') {
    if (key === 'count') return `${cfg.n}+ new ATH closes in ${K}D`;
    if (key === 'recent') return cfg.n === 0 ? 'a new ATH today' : `latest ATH within ${cfg.n}D`;
    if (key === 'near') return `close within ${g(cfg.x)}${unitTxt(cfg)} of ATH`;
    if (key === 'breakout') return `no ${cfg.n > 1 ? times : 'close'} below breakout${tol}`;
  } else {
    if (key === 'drawdown') return `close more than ${g(cfg.x)}${unitTxt(cfg)} below ATH`;
    if (key === 'stale') return `no new ATH in over ${cfg.n}D`;
    if (key === 'breakout') return `${times} below breakout${tol}`;
    if (key === 'count') return `fewer than ${cfg.n} new ATH closes in ${K}D`;
  }
  return key;
}
function exprText(side) {
  const K = RULES.window_days, items = enabled(side).map(([key, cfg]) => [cfg, condDesc(side, key, cfg, K)]);
  if (!items.length) return 'never (no conditions switched on)';
  const groups = [];
  items.forEach(([cfg, d], idx) => { if (idx === 0 || cfg.join === 'or') groups.push([d]); else groups[groups.length - 1].push(d); });
  const multi = groups.length > 1;
  return groups.map(gr => multi && gr.length > 1 ? '(' + gr.join(' AND ') + ')' : gr.join(' AND ')).join(' OR ');
}
function condDetail(side, key, cfg, s, st, i, K) {
  const px = s.c[i], a = s.atr[i];
  if (key === 'stoch') return `Stoch %K ${g(s.kv[i])}`;
  if (key === 'breakout') return `closed below breakout ${money(st.brk[i])}`;
  if (key === 'count') return side === 'entry' ? `${st.cnt[i]} of ${cfg.n} new ATH closes` : `only ${st.cnt[i]} new ATH closes in ${K}D`;
  if (key === 'near' || key === 'drawdown') {
    const word = side === 'entry' ? 'max' : 'limit';
    if (cfg.unit === 'atr') { if (a === null || a <= 0) return 'ATR n/a';
      return `${((st.lvl[i] - px) / a).toFixed(1)}x ATR below ATH (${word} ${g(cfg.x)}x)`; }
    return `${st.below[i].toFixed(1)}% below ATH (${word} ${g(cfg.x)}%)`;
  }
  if (key === 'recent') { const d = st.dsa[i]; return d === null ? 'no new ATH yet' : `last new ATH ${d}D ago (max ${cfg.n})`; }
  if (key === 'stale') { const d = st.dsa[i]; return d === null ? 'no new ATH' : `no new ATH in ${d}D (limit ${cfg.n})`; }
  return key;
}
function whyText(s, st, i) {
  const keys = st.why[i], side = st.wk[i];
  if (!keys || !keys.length) return '';
  return keys.map(k => condDetail(side, k, RULES[side][k], s, st, i, RULES.window_days)).join('; ');
}
function filtersSentence(f) {
  const p = [];
  if (f.rsi_min !== null || f.rsi_max !== null) p.push(`RSI ${f.rsi_min === null ? 'any' : g(f.rsi_min)}-${f.rsi_max === null ? 'any' : g(f.rsi_max)}`);
  if (f.rvol_min !== null) p.push(`RVOL >= ${g(f.rvol_min)}`);
  if (f.ret_min !== null) p.push(`day return >= ${g(f.ret_min)}%`);
  if (f.macd !== 'any') p.push(`MACD ${f.macd === 'bull' ? 'bullish' : 'bearish'}`);
  if (f.stoch !== 'any') p.push(`Stochastic ${f.stoch === 'bull' ? 'bullish' : 'bearish'}`);
  return 'Indicator filters on new alerts: ' + (p.length ? p.join(', ') : 'none');
}
const pyPct = (v, nd = 1, sign = true) => v === null || v === undefined ? 'n/a' : (sign && v >= 0 ? '+' : '') + v.toFixed(nd) + '%';

// ---------- email preview (mirrors build_email()) ----------
function emailGroups(a) {
  const nw = [], en = [], ru = [];
  T.forEach(t => {
    const [isNew, , isEnd] = events(S[t], ST[t], a, RULES);
    if (isNew) nw.push(t); else if (isEnd) en.push(t); else if (ST[t].conf[a] && S[t].c[a] !== null) ru.push(t);
  });
  const byRun = (x, y) => ST[y].cnt[a] - ST[x].cnt[a] || (ST[x].below[a] || 0) - (ST[y].below[a] || 0) || (x < y ? -1 : 1);
  nw.sort(byRun); ru.sort(byRun); en.sort((x, y) => ST[y].cnt[a] - ST[x].cnt[a] || (x < y ? -1 : 1));
  return {nw, en, ru};
}
function emailText(a) {
  const K = RULES.window_days, day = fmtDate(a), {nw, en, ru} = emailGroups(a), L = [], bar = '='.repeat(60), dash = '-'.repeat(60);
  const when = (st, i) => { const d = st.dsa[i]; return d === null ? 'n/a' : d === 0 ? 'today' : `${d} day${d !== 1 ? 's' : ''} ago (${mmmdd(i - d)})`; };
  L.push(bar, `  ALL-TIME HIGH SCANNER  -  ${day} close`, bar, '',
    `Run starts when: ${exprText('entry')}.`, `Run ends when: ${exprText('exit')}.`, filtersSentence(RULES.filters), '');
  L.push(`NEW RUNS (${nw.length})`, dash);
  if (!nw.length) L.push('   none today');
  nw.forEach((t, n) => {
    const s = S[t], st = ST[t], px = s.c[a], at = s.atr[a];
    const atrp = at === null || !px ? null : at / px * 100, start = st.start[a] === null ? 'n/a' : mmmdd(st.start[a]);
    L.push(`${n + 1}. ${t} - ${D.company[t]} (${D.sector[t]})`);
    L.push(`   New ATH closes (${K}D): ${st.cnt[a]}  |  last: ${when(st, a)}  |  run start: ${start}`);
    L.push(`   Close ${money(px)}  |  ATH ${money(st.lvl[a])}  |  ${pyPct(st.below[a], 1, false)} below  |  breakout ${money(st.brk[a])} (${pyPct(aboveBrk(s, st, a))} above)`);
    L.push(`   ATR ${money(at)} (${pyPct(atrp, 1, false)} of price)  |  Stoch %K ${g(s.kv[a])} / %D ${g(s.dv[a])}`);
    L.push(`   Day ${pyPct(dayRet(s.c, a))}  |  RVOL ${g(s.v[a])}x  |  RSI ${g(s.r[a])}  |  MACD ${s.m[a] === 1 ? 'bullish' : 'bearish'}`);
  });
  L.push('', `RUN ENDED - possible peak (${en.length})`, dash);
  if (!en.length) L.push('   none today');
  en.forEach((t, n) => {
    const s = S[t], st = ST[t], d = st.dsa[a];
    L.push(`${n + 1}. ${t} - ${D.company[t]} (${D.sector[t]})`);
    L.push(`   Peak close ${money(st.lvl[a])}${d !== null ? ' on ' + mmmdd(a - d) : ''}  |  now ${money(s.c[a])}, ${pyPct(st.below[a], 1, false)} below  |  day ${pyPct(dayRet(s.c, a))}`);
    L.push(`   Why: ${whyText(s, st, a)}  |  new ATH closes (${K}D): ${st.cnt[a]}  |  Stoch %K ${g(s.kv[a])}`);
  });
  L.push('');
  if (RULES.email.list_running) {
    L.push(`STILL RUNNING (${ru.length})`, dash);
    if (!ru.length) L.push('   none');
    else {
      const bySec = {}, secName = e => D.sectorName[e] || e;
      ru.forEach(t => { (bySec[D.sector[t]] = bySec[D.sector[t]] || []).push(t); });
      const order = Object.keys(bySec).sort((x, y) => bySec[y].length - bySec[x].length || (secName(x) < secName(y) ? -1 : secName(x) > secName(y) ? 1 : 0));
      L.push('   ' + 'Ticker'.padEnd(7) + 'Breakout'.padEnd(13) + 'Gain'.padStart(8) + `ATH hits ${K}D`.padStart(14));
      order.forEach(etf => {
        const ts = bySec[etf], gn = {};
        ts.forEach(t => { gn[t] = aboveBrk(S[t], ST[t], a); });
        ts.sort((x, y) => (gn[x] === null) - (gn[y] === null) || (gn[y] || 0) - (gn[x] || 0) || (x < y ? -1 : 1));
        L.push('', `${secName(etf)} (${etf}) - ${ts.length}`);
        ts.forEach(t => {
          const si = ST[t].start[a], d = si === null ? 'n/a' : D.dates[si].slice(5) + '-' + D.dates[si].slice(0, 4);
          L.push('   ' + t.padEnd(7) + d.padEnd(13) + pyPct(gn[t]).padStart(8) + String(ST[t].cnt[a]).padStart(14));
        });
      });
    }
    L.push('');
  }
  L.push(bar);
  let subject;
  const list = a2 => a2.slice(0, 8).join(', ') + (a2.length > 8 ? ', ...' : '');
  if (nw.length || en.length) {
    const b = []; if (nw.length) b.push(`${nw.length} new (${list(nw)})`); if (en.length) b.push(`${en.length} ended (${list(en)})`);
    subject = `${D.scannerName}: ${b.join(' · ')} - ${day}`;
  } else subject = `${D.scannerName}: no new or ended runs (${ru.length} running) - ${day}`;
  const w = RULES.email.send_when;
  const send = w === 'always' || (w === 'new' && nw.length > 0) || (w === 'changes' && (nw.length + en.length) > 0);
  return {subject, body: L.join('\n'), send, nw, en, ru};
}

// ---------- view filters ----------
function view() {
  const sec = $('sector').value, tk = new Set(parseTickers($('tickerInput').value));
  return {sec, tk, ok: t => (!sec || D.sector[t] === sec) && (!tk.size || tk.has(t))};
}
const asOf = () => +$('asOf').value;

// ---------- email card ----------
let lastEmail = null;
function renderEmail() {
  const a = asOf(), e = emailText(a); lastEmail = e;
  $('emailSubject').textContent = 'Subject: ' + e.subject;
  $('emailBody').textContent = e.body;
  $('sendFlag').innerHTML = e.send ? `<b style="color:#0b5a24">Would send</b> for the ${fmtDate(a)} close.`
    : `<b style="color:${INK2}">Would not send</b> for the ${fmtDate(a)} close (send when: ${$('e_when').selectedOptions[0].text}).`;
  const same = rulesKey(RULES) === rulesKey(D.rules);
  const chip = $('ruleChip');
  chip.className = 'chip ' + (same ? 'ok' : 'edit');
  chip.textContent = same ? (D.rulesExists ? 'Matches the email rules' : 'Email is using the starting defaults')
    : 'Edited - not in the email until you commit the rules file';
  try { if (!same) localStorage.setItem('athRulesDraftV2', rulesKey(RULES)); } catch (e) {}
}
function setupPushLinks() {
  const gh = D.gh;
  if (gh) {
    $('ghRules').href = gh.rules; $('ghRules').style.display = '';
    $('ghRules').textContent = gh.rulesExists ? 'Open rules file on GitHub' : 'Create rules file on GitHub';
    $('ghRun').href = gh.run; $('ghRun').style.display = '';
  }
  $('pushHow').innerHTML = gh
    ? `To push edited rules to the email: <b>Copy rules JSON</b> → <b>${gh.rulesExists ? 'Open' : 'Create'} rules file on GitHub</b> → ` +
      `select all, paste, commit. Then <b>Run workflow now</b> (or wait for the schedule) to send the email and rebuild this page with the new rules.`
    : `To push edited rules to the email: download <b>${D.rulesFile}</b> and commit it to the repo root (same folder as the scripts). ` +
      `The next scheduled or manual workflow run emails with it and rebuilds this page.`;
}

// ---------- tiles + leaderboard ----------
function statusOf(t, a) {
  const s = S[t], st = ST[t], [isNew, filt, isEnd] = events(s, st, a, RULES);
  if (isNew) return ['new', 'New run', 0];
  if (filt) return ['new', 'New run (filtered out)', 1];
  if (isEnd) return ['end', 'Ended today', 2];
  if (st.conf[a]) return ['run', 'Running', 3];
  return ['no', 'Not in a run', 4];
}
function renderTiles(a, v) {
  let conf = 0, nw = 0, en = 0, ath = 0;
  T.forEach(t => { if (!v.ok(t)) return; const st = ST[t], [isNew, , isEnd] = events(S[t], st, a, RULES);
    if (st.conf[a]) conf++; if (isNew) nw++; if (isEnd) en++; if (st.ath[a]) ath++; });
  const tl = [[conf, 'runs in progress'], [nw, 'new-run alerts'], [en, 'runs ended (possible peak)'], [ath, 'new ATH closes that day']];
  $('tiles').innerHTML = tl.map(([x, l]) => `<div class="tile"><div class="v">${x}</div><div class="l">${l}</div></div>`).join('');
}
let showAllLb = false;
function renderLeaderboard(a, v) {
  const K = RULES.window_days;
  const rows = T.filter(t => v.ok(t) && S[t].c[a] !== null && (ST[t].cnt[a] > 0 || ST[t].conf[a] || events(S[t], ST[t], a, RULES)[2]))
    .map(t => ({t, s: statusOf(t, a)}))
    .sort((x, y) => x.s[2] - y.s[2] || ST[y.t].cnt[a] - ST[x.t].cnt[a] || (ST[x.t].below[a] || 0) - (ST[y.t].below[a] || 0));
  $('lbTitle').textContent = `ATH runs leaderboard - ${fmtDate(a)} close`;
  const shown = showAllLb ? rows : rows.slice(0, 40);
  const head = `<tr><th>Ticker</th><th>Company</th><th>Sector</th><th>Status</th><th class="num">New ATHs (${K}D)</th>` +
    `<th class="num">Days since ATH</th><th class="num">% below ATH</th><th class="num">Close</th><th class="num">ATH close</th>` +
    `<th class="num">Breakout</th><th class="num">% above breakout</th><th>Run started</th><th class="num">Day</th>` +
    `<th class="num">RVOL</th><th class="num">Stoch %K / %D</th><th class="num">ATR</th><th>Entry not met / exit reason</th></tr>`;
  $('lbTable').innerHTML = head + (shown.length ? shown.map(({t, s}) => {
    const sd = S[t], st = ST[t], b = st.below[a], ab = aboveBrk(sd, st, a), f = (st.conf[a] || s[0] === 'end') ? st.start[a] : null;
    const why = st.conf[a] ? '' : (st.wk[a] === 'exit' ? '<b>Exit:</b> ' : '') + whyText(sd, st, a);
    const atrp = sd.atr[a] === null || !sd.c[a] ? '' : ` <span class="hint">(${(sd.atr[a] / sd.c[a] * 100).toFixed(1)}%)</span>`;
    return `<tr class="click${t === $('rtTicker').value ? ' sel' : ''}" data-t="${t}"><td><b>${t}</b></td><td>${D.company[t]}</td><td>${D.sector[t]}</td>` +
      `<td><span class="st ${s[0]}">${s[1]}</span></td><td class="num">${st.cnt[a]}</td><td class="num">${st.dsa[a] ?? 'n/a'}</td>` +
      `<td class="num"><span class="sw" style="background:${colorBelow(b)}"></span>${pctU(b)}</td><td class="num">${money(sd.c[a])}</td>` +
      `<td class="num">${money(st.lvl[a])}</td><td class="num">${money(st.brk[a])}</td>` +
      `<td class="num"><span class="sw" style="background:${colorRet(ab)}"></span>${pct(ab, 1)}</td><td>${f === null ? '' : mmmdd(f)}</td>` +
      `<td class="num">${pct(dayRet(sd.c, a), 1)}</td><td class="num">${sd.v[a] === null ? 'n/a' : sd.v[a].toFixed(2) + 'x'}</td>` +
      `<td class="num">${sd.kv[a] ?? 'n/a'} / ${sd.dv[a] ?? 'n/a'}</td><td class="num">${money(sd.atr[a])}${atrp}</td>` +
      `<td class="hint why">${why}</td></tr>`;
  }).join('') : '<tr><td colspan="17" class="hint">No tickers with a new ATH close in the window for this selection.</td></tr>');
  $('lbNote').innerHTML = `${rows.length} tickers with at least one new ATH close in the last ${K} trading days, in a run, or ending a run. ` +
    `% above breakout = close vs the breakout level (frozen at the run's start; for tickers not in a run, the old ATH their latest move broke).` +
    (rows.length > 40 ? ` <a href="#" id="lbMore">${showAllLb ? 'Show top 40' : 'Show all ' + rows.length}</a>` : '') + ' Click a row to open it in the run tracker.';
  const m = $('lbMore'); if (m) m.onclick = e => { e.preventDefault(); showAllLb = !showAllLb; renderLeaderboard(asOf(), view()); };
  $('lbTable').querySelectorAll('tr.click').forEach(tr => tr.onclick = () => { $('rtTicker').value = tr.dataset.t; renderTracker(); renderLeaderboard(asOf(), view());
    $('rtTitle').scrollIntoView({behavior: 'smooth', block: 'start'}); });
  return rows;
}

// ---------- run tracker ----------
function colorFor(mode, t, i) {
  if (mode === 'below') { const b = ST[t].below[i]; return {col: colorBelow(b), x: b === null ? null : 1 - 2 * Math.min(b, D.cap) / D.cap}; }
  const v = fwdRet(S[t].c, i, +mode); return {col: colorRet(v), x: v === null ? null : v / D.cap, v};
}
function scaleTrace(mode, x0, xref) {
  const below = mode === 'below';
  return {type: 'scatter', x: [x0], y: [0], xaxis: xref || 'x', mode: 'markers', hoverinfo: 'skip', showlegend: false,
    marker: {size: 0.1, opacity: 0, color: [0], cmin: below ? 0 : -D.cap, cmax: D.cap, showscale: true,
      colorscale: below ? [...D.gradient].reverse().map(([p, c]) => [(1 - p) / 2, c]) : D.gradient.map(([p, c]) => [(p + 1) / 2, c]),
      colorbar: {orientation: 'h', x: 1, xanchor: 'right', y: 1.02, yanchor: 'bottom', len: 0.34, thickness: 10, outlinewidth: 0,
        tickfont: {size: 10}, tickvals: below ? [0, D.cap / 2, D.cap] : [-D.cap, 0, D.cap],
        ticktext: below ? ['at ATH', `${D.cap / 2}% below`, `≥ ${D.cap}% below`] : [`≤ -${D.cap}%`, '0%', `≥ +${D.cap}%`],
        title: {text: below ? '% below ATH' : `${mode}D forward return`, side: 'top', font: {size: 11}}}}};
}
function exitLevel(s, st, i) {
  const c = RULES.exit.drawdown;
  if (!c.on || !st.conf[i] || st.lvl[i] === null) return null;
  if (c.unit === 'atr') return s.atr[i] === null ? null : st.lvl[i] - c.x * s.atr[i];
  return st.lvl[i] * (1 - c.x / 100);
}
function renderTracker() {
  let t = $('rtTicker').value.trim().toUpperCase().replace('.', '-');
  const mode = $('rtColor').value, K = RULES.window_days, a = asOf();
  if (!S[t]) { $('rtSummary').innerHTML = t ? `<b>${t}</b> made no new all-time closing high in this period (or isn't an S&amp;P 500 ticker).` : '';
    Plotly.purge('rtChart'); return; }
  const s = S[t], st = ST[t], xs = [], ys = [], cols = [], cd = [], pat = [];
  for (let i = S0; i < NDAYS; i++) {
    if (s.c[i] === null) continue;
    const cl = colorFor(mode, t, i);
    xs.push(D.dates[i]); ys.push(st.cnt[i]); cols.push(cl.col); pat.push(cl.x === null ? '/' : '');
    const f = mode === 'below' ? null : fwdRet(s.c, i, +mode);
    cd.push([money(s.c[i]), money(st.lvl[i]), pctU(st.below[i]), st.dsa[i] ?? 'n/a',
      st.conf[i] ? 'in a run' : 'not in a run' + (whyText(s, st, i) ? ': ' + whyText(s, st, i) : ''),
      st.ath[i] ? 'new ATH close' : '', mode === 'below' ? '' : `<br>${mode}D forward return: ${f === null ? 'not yet available' : pct(f)}`,
      `RVOL ${s.v[i] ?? 'n/a'}x · Stoch %K ${s.kv[i] ?? 'n/a'} · ATR ${money(s.atr[i])}`]);
  }
  const bars = {type: 'bar', x: xs, y: ys, name: `New ATHs (${K}D)`, marker: {color: cols, line: {width: 0}, pattern: {shape: pat, fgcolor: '#a9a8a2', size: 5, solidity: .25}},
    customdata: cd, width: 0.85 * DAY, hovertemplate: `<b>%{x|%b %d, %Y}</b> %{customdata[5]}<br>New ATH closes in last ${K}D: <b>%{y}</b><br>` +
      'Close %{customdata[0]} · ATH %{customdata[1]} · %{customdata[2]} below<br>Days since ATH: %{customdata[3]}<br>%{customdata[7]}<br>%{customdata[4]}%{customdata[6]}<extra></extra>'};
  const px = [], py = [], al = [], bk = [], xl = [], up = {x: [], y: []}, dn = {x: [], y: []};
  for (let i = S0; i < NDAYS; i++) {
    if (s.c[i] === null) continue;
    px.push(D.dates[i]); py.push(s.c[i]); al.push(st.lvl[i]); bk.push(st.conf[i] ? st.brk[i] : null); xl.push(exitLevel(s, st, i));
    const [isNew, filt, isEnd] = events(s, st, i, RULES);
    if (isNew || filt) { up.x.push(D.dates[i]); up.y.push(s.c[i]); }
    if (isEnd) { dn.x.push(D.dates[i]); dn.y.push(s.c[i]); }
  }
  const traces = [bars,
    {type: 'scatter', x: px, y: py, yaxis: 'y2', mode: 'lines', name: 'Close', line: {color: INK, width: 1.6}, hovertemplate: 'Close $%{y:,.2f}<extra></extra>'},
    {type: 'scatter', x: px, y: al, yaxis: 'y2', mode: 'lines', name: 'ATH close', line: {color: '#0b5a24', width: 1.2, dash: 'dot'}, hovertemplate: 'ATH $%{y:,.2f}<extra></extra>'},
    {type: 'scatter', x: px, y: bk, yaxis: 'y2', mode: 'lines', name: 'Breakout level', line: {color: '#a9a8a2', width: 1.2, dash: 'dash'}, connectgaps: false, hovertemplate: 'Breakout $%{y:,.2f}<extra></extra>'},
    {type: 'scatter', x: px, y: xl, yaxis: 'y2', mode: 'lines', name: 'Drawdown exit level', line: {color: '#c0392b', width: 1, dash: 'dot'}, connectgaps: false, hovertemplate: 'Exit level $%{y:,.2f}<extra></extra>'},
    {type: 'scatter', x: up.x, y: up.y, yaxis: 'y2', mode: 'markers', name: 'Run started', marker: {symbol: 'triangle-up', size: 12, color: '#0b5a24', line: {color: '#fff', width: 1}}, hovertemplate: 'Run started %{x|%b %d}<extra></extra>'},
    {type: 'scatter', x: dn.x, y: dn.y, yaxis: 'y2', mode: 'markers', name: 'Run ended', marker: {symbol: 'triangle-down', size: 12, color: '#8e1b1b', line: {color: '#fff', width: 1}}, hovertemplate: 'Run ended %{x|%b %d}<extra></extra>'},
    scaleTrace(mode, xs[0])];
  const lay = {barmode: 'overlay', plot_bgcolor: SURF, paper_bgcolor: '#fff', height: 500, margin: {l: 50, r: 60, t: 60, b: 40},
    font: {family: 'Inter, Segoe UI, Arial, sans-serif', color: INK2, size: 12}, hoverlabel: {bgcolor: '#fff', font: {color: INK}},
    legend: {orientation: 'h', x: 0, y: 1.02, yanchor: 'bottom', font: {size: 11}}, hovermode: 'x unified',
    xaxis: {showgrid: false, linecolor: GRIDC, rangeslider: {visible: true, thickness: 0.06}, rangebreaks: [{bounds: ['sat', 'mon']}, {values: D.holidays}]},
    yaxis: {title: {text: `New ATH closes, last ${K}D`, font: {size: 11}}, gridcolor: GRIDC, zeroline: false, range: [0, Math.max(K, 1) * 1.02], fixedrange: true},
    yaxis2: {overlaying: 'y', side: 'right', showgrid: false, tickprefix: '$', title: {text: 'Close', font: {size: 11}}},
    shapes: a < LAST ? [{type: 'line', xref: 'x', yref: 'paper', x0: D.dates[a], x1: D.dates[a], y0: 0, y1: 1, line: {color: '#2a78d6', width: 1, dash: 'dot'}}] : []};
  Plotly.react('rtChart', traces, lay, {displaylogo: false, responsive: true});
  let pkI = null; for (let i = S0; i <= a; i++) if (st.cnt[i] > 0 && (pkI === null || st.cnt[i] > st.cnt[pkI])) pkI = i;
  const sNow = statusOf(t, a), ab = aboveBrk(s, st, a);
  $('rtTitle').textContent = `Run tracker - ${t} · ${D.company[t]}`;
  $('rtSummary').innerHTML = `<span class="st ${sNow[0]}">${sNow[1]}</span> as of ${fmtDate(a)} · <b>${st.cnt[a]}</b> new ATH closes in the last ${K} days · ` +
    `last ATH ${st.dsa[a] === null ? 'n/a' : st.dsa[a] === 0 ? 'today' : st.dsa[a] + ' days ago (' + mmmdd(a - st.dsa[a]) + ')'} at ${money(st.lvl[a])} · ` +
    `close ${money(s.c[a])} (${pctU(st.below[a])} below) · breakout ${money(st.brk[a])} (${pct(ab, 1)} above) · ` +
    `ATR ${money(s.atr[a])} · Stoch %K ${s.kv[a] ?? 'n/a'} / %D ${s.dv[a] ?? 'n/a'}` +
    (pkI !== null ? ` · run count peaked at <b>${st.cnt[pkI]}</b> on ${mmmdd(pkI)}` : '') +
    (!st.conf[a] && whyText(s, st, a) ? `<br>${st.wk[a] === 'exit' ? 'Exit' : 'Entry not met'}: ${whyText(s, st, a)}` : '');
  $('rtHint').textContent = mode === 'below' ? 'Known on the day - no look-ahead.' : 'Hindsight: what happened next. Hatched = not available yet.';
}

// ---------- heatmap ----------
function renderHeatmap(a, v, lbRows) {
  const K = RULES.window_days, i0 = Math.max(S0, a - D.heatDays + 1), mode = $('hmColor').value, ring = +$('hmRing').value;
  const pick = lbRows.filter(r => r.s[2] <= 3).slice(0, D.heatRows).map(r => r.t);
  if (pick.length < D.heatRows) lbRows.forEach(r => { if (pick.length < D.heatRows && !pick.includes(r.t)) pick.push(r.t); });
  $('hmTitle').textContent = `Run heatmap - top ${pick.length} by new ATH closes (${K}D), ${mmmdd(i0)} to ${fmtDate(a)}`;
  $('hmNote').innerHTML = (mode === 'below'
      ? 'Cell color = % below the all-time high that day (darkest green = closed at a new ATH, marked •). '
      : `Cell color = relative volume: that day's volume ÷ its prior 20-day average (lightest ≤ ${D.rvolRange[0]}x, darkest ≥ ${D.rvolRange[1]}x). • = new ATH close. `) +
    (ring ? `Black rings = days with RVOL ≥ ${ring}x; a bigger ring means heavier volume. ` : '') +
    'Rows = leaders as of the selected date. Click a row to load it in the run tracker.';
  if (!pick.length) { Plotly.purge('hmChart'); return; }
  const xs = []; for (let i = i0; i <= a; i++) xs.push(D.dates[i]);
  const ys = [...pick].reverse(), z = [], txt = [], cd = [], rx = [], ry = [], rs = [], rc = [];
  ys.forEach(t => {
    const zr = [], tr = [], cr = [];
    for (let i = i0; i <= a; i++) {
      const b = ST[t].below[i], rv = S[t].v[i];
      zr.push(mode === 'below' ? (b === null ? null : Math.min(b, D.cap)) : (rv === null ? null : Math.min(Math.max(rv, D.rvolRange[0]), D.rvolRange[1])));
      tr.push(ST[t].ath[i] ? '•' : '');
      cr.push([money(S[t].c[i]), pctU(b), ST[t].cnt[i], ST[t].ath[i] ? ' · new ATH close' : '', ST[t].conf[i] ? 'in a run' : 'not in a run',
        rv === null ? 'n/a' : rv.toFixed(2) + 'x']);
      if (ring && rv !== null && rv >= ring && S[t].c[i] !== null) {
        rx.push(D.dates[i]); ry.push(t); rs.push(5 + 3 * Math.min(rv, 4)); rc.push([rv.toFixed(2) + 'x']);
      }
    }
    z.push(zr); txt.push(tr); cd.push(cr);
  });
  const below = mode === 'below';
  const trace = {type: 'heatmap', x: xs, y: ys, z, text: txt, texttemplate: '%{text}', textfont: {color: below ? '#ffffff' : '#f2b705', size: 11}, customdata: cd,
    zmin: below ? 0 : D.rvolRange[0], zmax: below ? D.cap : D.rvolRange[1], xgap: 1, ygap: 1,
    colorscale: below ? [...D.gradient].reverse().map(([p, c]) => [(1 - p) / 2, c]) : D.rvolScale,
    colorbar: below
      ? {thickness: 10, len: 0.6, outlinewidth: 0, tickvals: [0, D.cap / 2, D.cap], ticktext: ['at ATH', `${D.cap / 2}%`, `≥ ${D.cap}%`], title: {text: '% below ATH', side: 'right', font: {size: 11}}}
      : {thickness: 10, len: 0.6, outlinewidth: 0, tickvals: [D.rvolRange[0], 1, 2, D.rvolRange[1]], ticktext: [`≤ ${D.rvolRange[0]}x`, '1x', '2x', `≥ ${D.rvolRange[1]}x`], title: {text: 'RVOL', side: 'right', font: {size: 11}}},
    hovertemplate: '<b>%{y}</b> %{x|%b %d, %Y}%{customdata[3]}<br>Close %{customdata[0]} · %{customdata[1]} below ATH<br>' +
      `RVOL %{customdata[5]} · new ATH closes (${K}D): %{customdata[2]} · %{customdata[4]}<extra></extra>`};
  const traces = [trace];
  if (rx.length) traces.push({type: 'scatter', mode: 'markers', x: rx, y: ry, customdata: rc, hoverinfo: 'skip', showlegend: false,
    marker: {symbol: 'circle-open', size: rs, color: INK, line: {width: 1.6, color: INK}}});
  Plotly.react('hmChart', traces, {height: Math.max(260, 18 * ys.length + 90), margin: {l: 60, r: 20, t: 10, b: 40},
    plot_bgcolor: SURF, paper_bgcolor: '#fff', font: {family: 'Inter, Segoe UI, Arial, sans-serif', color: INK2, size: 11},
    xaxis: {type: 'category', categoryorder: 'array', categoryarray: xs, tickvals: xs.filter((_, k) => k % 5 === 0), ticktext: xs.filter((_, k) => k % 5 === 0).map(d => d.slice(5))},
    yaxis: {type: 'category', categoryorder: 'array', categoryarray: ys, automargin: true}}, {displaylogo: false, responsive: true});
  const el = $('hmChart'); el.removeAllListeners && el.removeAllListeners('plotly_click');
  el.on('plotly_click', ev => { if (ev.points && ev.points[0]) { $('rtTicker').value = ev.points[0].y; renderTracker(); renderLeaderboard(asOf(), view());
    $('rtTitle').scrollIntoView({behavior: 'smooth', block: 'start'}); } });
}

// ---------- backtest ----------
function hits(v) {
  const type = $('hitType').value, n = +$('fwdN').value, sign = $('fwdSign').value, out = [];
  T.forEach(t => {
    if (!v.ok(t)) return;
    const s = S[t], st = ST[t];
    for (let i = S0; i < NDAYS; i++) {
      if (s.c[i] === null) continue;
      let on;
      if (type === 'ath') on = st.ath[i] && passesFilters(s, i, RULES.filters);
      else if (type === 'conf') on = st.conf[i] && passesFilters(s, i, RULES.filters);
      else { const ev = events(s, st, i, RULES); on = type === 'new' ? ev[0] : ev[2]; }
      if (!on) continue;
      const f = fwdRet(s.c, i, n);
      if (sign === 'pos' && !(f !== null && f > 0)) continue;
      if (sign === 'neg' && !(f !== null && f < 0)) continue;
      out.push({t, i, v: f});
    }
  });
  return out;
}
function stackTraces(groups, widthMs, hover) {
  let maxK = 0; groups.forEach(a2 => { maxK = Math.max(maxK, a2.length); });
  const traces = [];
  for (let k = 0; k < maxK; k++) {
    const tr = {type: 'bar', x: [], y: [], text: [], customdata: [], showlegend: false,
      marker: {color: [], line: {color: SURF, width: 0.5}, pattern: {shape: [], fgcolor: '#a9a8a2', size: 5, solidity: 0.25}},
      textposition: 'inside', insidetextanchor: 'middle', constraintext: 'inside', textfont: {size: 10, color: []}, width: widthMs, hovertemplate: hover};
    groups.forEach((a2, x) => { if (a2.length <= k) return; const sg = a2[k];
      tr.x.push(x); tr.y.push(sg.y); tr.text.push(sg.t); tr.customdata.push(sg.cd);
      tr.marker.color.push(colorRet(sg.v)); tr.marker.pattern.shape.push(sg.v === null ? '/' : '');
      tr.textfont.color.push(inkOn(sg.v === null ? null : sg.v / D.cap)); });
    traces.push(tr);
  }
  return traces;
}
function baseLayout(maxY) {
  return {barmode: 'stack', bargap: 0.18, plot_bgcolor: SURF, paper_bgcolor: '#fff', height: 460, margin: {l: 50, r: 20, t: 60, b: 40},
    showlegend: false, uirevision: 'keep', font: {family: 'Inter, Segoe UI, Arial, sans-serif', color: INK2, size: 12},
    hoverlabel: {bgcolor: '#fff', font: {color: INK}},
    xaxis: {showgrid: false, linecolor: GRIDC, rangeslider: {visible: true, thickness: 0.06}},
    yaxis: {gridcolor: GRIDC, zeroline: false, rangemode: 'tozero', tickformat: ',d', dtick: maxY <= 12 ? 1 : null}};
}
const byFwdDesc = (a2, b) => (a2.v === null) - (b.v === null) || (b.v || 0) - (a2.v || 0);
function weekMid(ds) { const d = new Date(ds + 'T00:00:00Z'); return new Date(d.getTime() + (3 - d.getUTCDay()) * DAY).toISOString().slice(0, 10); }
function weekLabel(mid) { return new Date(new Date(mid + 'T00:00:00Z').getTime() - 2 * DAY).toLocaleDateString('en-US', {month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC'}); }
function drawDaily(H, n) {
  const groups = new Map(), trading = D.dates.slice(S0);
  [...H].sort(byFwdDesc).forEach(h => { const x = D.dates[h.i], s = S[h.t], st = ST[h.t];
    if (!groups.has(x)) groups.set(x, []);
    groups.get(x).push({t: h.t, y: 1, v: h.v, cd: [h.t, D.company[h.t], D.sector[h.t], st.cnt[h.i], pctU(st.below[h.i]), s.v[h.i] ?? 'n/a', s.r[h.i] ?? 'n/a',
      h.v === null ? 'not yet available' : pct(h.v), pct(dayRet(s.c, h.i)), s.kv[h.i] ?? 'n/a']}); });
  const hover = '<b>%{customdata[0]}</b> - %{customdata[1]}<br>%{x|%b %d, %Y} · %{customdata[2]}<br>' +
    `New ATHs (${RULES.window_days}D): %{customdata[3]} · %{customdata[4]} below ATH<br>RVOL %{customdata[5]}x · RSI %{customdata[6]} · Stoch %K %{customdata[9]} · day %{customdata[8]}<br>` +
    `<b>${n}D forward return: %{customdata[7]}</b><extra></extra>`;
  const traces = stackTraces(groups, 0.8 * DAY, hover), tot = trading.map(x => (groups.get(x) || []).length);
  traces.push({type: 'scatter', x: trading, y: tot, mode: 'markers', showlegend: false, marker: {opacity: 0, size: 1}, hovertemplate: '<b>%{x|%b %d, %Y}</b><br>Hits: %{y}<extra></extra>'});
  traces.push(scaleTrace(String(n), trading[0]));
  const lay = baseLayout(Math.max(0, ...tot)); lay.xaxis.rangebreaks = [{bounds: ['sat', 'mon']}, {values: D.holidays}];
  Plotly.react('dailyChart', traces, lay, {displaylogo: false, responsive: true});
}
function drawWeekly(H, n) {
  const wk = new Map();
  H.forEach(h => { const ds = D.dates[h.i], mid = weekMid(ds);
    if (!wk.has(mid)) wk.set(mid, new Map()); const m = wk.get(mid);
    if (!m.has(h.t)) m.set(h.t, {days: 0, vals: [], dates: []}); const e = m.get(h.t);
    e.days++; e.dates.push(ds); if (h.v !== null) e.vals.push(h.v); });
  const groups = new Map(), totals = {}, uniq = {};
  [...wk.keys()].sort().forEach(mid => { const segs = [];
    wk.get(mid).forEach((e, t) => { const vv = e.vals.length ? mean(e.vals) : null;
      const dl = e.dates.sort().map(d => new Date(d + 'T00:00:00Z').toLocaleDateString('en-US', {weekday: 'short', month: '2-digit', day: '2-digit', timeZone: 'UTC'})).join(', ');
      segs.push({t, y: e.days, v: vv, cd: [t, D.company[t], D.sector[t], dl, weekLabel(mid), vv === null ? 'not yet available' : pct(vv)]}); });
    segs.sort(byFwdDesc); groups.set(mid, segs); totals[mid] = segs.reduce((s, x) => s + x.y, 0); uniq[mid] = segs.length; });
  const hover = '<b>%{customdata[0]}</b> - %{customdata[1]}<br>Week of %{customdata[4]} · %{customdata[2]}<br>Days hit this week: %{y}<br>%{customdata[3]}<br>' +
    `<b>Avg ${n}D forward return: %{customdata[5]}</b><extra></extra>`;
  const traces = stackTraces(groups, 0.8 * 5 * DAY, hover), allWeeks = [...new Set(D.dates.slice(S0).map(weekMid))].sort();
  traces.push({type: 'scatter', x: allWeeks, y: allWeeks.map(w => totals[w] || 0), mode: 'markers', showlegend: false, marker: {opacity: 0, size: 1},
    customdata: allWeeks.map(w => [uniq[w] || 0, weekLabel(w)]), hovertemplate: 'Week of %{customdata[1]}<br>Total hits: %{y}<br>Unique tickers: %{customdata[0]}<extra></extra>'});
  traces.push(scaleTrace(String(n), allWeeks[0]));
  const lay = baseLayout(Math.max(0, ...Object.values(totals))); lay.xaxis.tickformat = '%b %d';
  lay.annotations = Object.entries(totals).map(([x, vv]) => ({x, y: vv, text: String(vv), showarrow: false, yshift: 9, font: {size: 10, color: INK2}}));
  Plotly.react('weeklyChart', traces, lay, {displaylogo: false, responsive: true});
}
function statRow(label, rows, n, cls, attr) {
  const fw = m => rows.map(h => fwdRet(S[h.t].c, h.i, m)).filter(x => x !== null), fc = fw(n);
  const cells = D.fwdDays.map(m => { const mm = mean(fw(m));
    return `<td class="num">${m === n ? `<span class="sw" style="background:${colorRet(mm)}"></span>` : ''}${pct(mm)}</td>`; }).join('');
  const win = fc.length ? (fc.filter(x => x > 0).length / fc.length * 100).toFixed(0) + '%' : 'n/a';
  return {avg: mean(fc), html: `<tr class="${cls}" ${attr}><td>${label}</td><td class="num">${rows.length}</td>` +
    `<td class="num">${new Set(rows.map(h => h.t)).size}</td>${cells}<td class="num">${pct(median(fc))}</td><td class="num">${win}</td></tr>`};
}
function renderBacktest(v) {
  const n = +$('fwdN').value, H = hits(v), type = $('hitType').value;
  $('btWarn').innerHTML = ($('fwdSign').value !== 'all' ? `<div class="warn">Showing only hits whose ${n}-day forward return was <b>${$('fwdSign').value === 'pos' ? 'positive' : 'negative'}</b> - a hindsight view, not a fair backtest.</div>` : '') +
    (type === 'end' ? `<div class="warn">For run ends, <b>negative</b> forward returns mean the exit caught a real peak.</div>` : '');
  const perDay = {}; H.forEach(h => { perDay[h.i] = (perDay[h.i] || 0) + 1; });
  const counts = Object.values(perDay), fv = H.map(h => h.v).filter(x => x !== null), win = fv.length ? fv.filter(x => x > 0).length / fv.length * 100 : null;
  const nd = NDAYS - S0;
  $('btTiles').innerHTML = [[H.length.toLocaleString(), 'hits'], [new Set(H.map(h => h.t)).size, 'unique tickers'],
    [(H.length / nd).toFixed(1), 'avg hits / day'], [`${counts.length}/${nd}`, 'days with ≥1 hit'],
    [win === null ? 'n/a' : win.toFixed(0) + '%', `${n}D win rate`], [pct(mean(fv)), `avg ${n}D return`], [pct(median(fv)), `median ${n}D return`]]
    .map(([x, l]) => `<div class="tile"><div class="v">${x}</div><div class="l">${l}</div></div>`).join('');
  drawDaily(H, n); drawWeekly(H, n);
  const by = new Map(); H.forEach(h => { const s = D.sector[h.t]; if (!by.has(s)) by.set(s, []); by.get(s).push(h); });
  const head = `<tr><th>Sector</th><th class="num">Hits</th><th class="num">Tickers</th>` + D.fwdDays.map(m => `<th class="num">Avg ${m}D</th>`).join('') +
    `<th class="num">Median ${n}D</th><th class="num">${n}D % positive</th></tr>`;
  const rows = [...by.entries()].map(([s, r]) => statRow(`<b>${s}</b> <span class="hint">${D.sectorName[s] || ''}</span>`, r, n, 'click', `data-sector="${s}"`))
    .sort((x, y) => (x.avg === null) - (y.avg === null) || (y.avg || 0) - (x.avg || 0));
  $('sectorTable').innerHTML = head + (H.length ? statRow('All sectors', H, n, 'all click', 'data-sector=""').html + rows.map(r => r.html).join('')
    : `<tr><td colspan="${5 + D.fwdDays.length}" class="hint">No hits for this selection.</td></tr>`);
  $('sectorTable').querySelectorAll('tr.click').forEach(tr => tr.onclick = () => { $('sector').value = tr.dataset.sector; fillTickerList(); renderAll(); });
  const bt = new Map(); H.forEach(h => { if (!bt.has(h.t)) bt.set(h.t, []); bt.get(h.t).push(h); });
  const top = [...bt.entries()].sort((x, y) => y[1].length - x[1].length).slice(0, 15);
  $('topTable').innerHTML = `<tr><th>Ticker</th><th>Company</th><th>Sector</th><th class="num">Days hit</th><th class="num">Avg ${n}D fwd</th></tr>` +
    top.map(([t, r]) => { const m = mean(r.map(h => h.v).filter(x => x !== null));
      return `<tr class="click" data-t="${t}"><td><b>${t}</b></td><td>${D.company[t]}</td><td>${D.sector[t]}</td><td class="num">${r.length}</td>` +
        `<td class="num"><span class="sw" style="background:${colorRet(m)}"></span>${pct(m)}</td></tr>`; }).join('');
  $('topTable').querySelectorAll('tr.click').forEach(tr => tr.onclick = () => { $('rtTicker').value = tr.dataset.t; renderTracker(); renderLeaderboard(asOf(), view());
    $('rtTitle').scrollIntoView({behavior: 'smooth', block: 'start'}); });
}

// ---------- wiring ----------
let LB = [];
function renderAll() {
  RULES = readRules(); styleEditor(); ensureStates();
  const a = asOf(), v = view();
  renderEmail(); renderTiles(a, v);
  LB = renderLeaderboard(a, v);
  if (!S[$('rtTicker').value.trim().toUpperCase()] && LB.length) $('rtTicker').value = LB[0].t;
  renderTracker(); renderHeatmap(a, v, LB); renderBacktest(v); postHeight();
}
function fillTickerList() {
  const sec = $('sector').value;
  $('tickerList').innerHTML = T.filter(t => !sec || D.sector[t] === sec).map(t => `<option value="${t}">${D.company[t]}</option>`).join('');
}
function postHeight() { if (window.parent !== window) window.parent.postMessage({type: 'scanner-height', h: document.documentElement.scrollHeight}, '*'); }
function rulesFileText() { return JSON.stringify(RULES, null, 2) + '\n'; }
async function copy(text, msg) {
  try { await navigator.clipboard.writeText(text); toast(msg); }
  catch (e) { const ta = document.createElement('textarea'); ta.value = text; document.body.appendChild(ta); ta.select();
    try { document.execCommand('copy'); toast(msg); } catch (e2) { toast('Copy failed - use Download instead'); } ta.remove(); }
}

(function init() {
  $('sub').innerHTML = `S&amp;P 500 · data through the <b>${fmtDate(LAST)}</b> close · showing ${fmtDate(S0)} → ${fmtDate(LAST)} · ` +
    `${T.length} tickers made at least one new all-time closing high in this period. The email and this page read the same rules file (<code>${D.rulesFile}</code>).`;
  const opts = []; for (let i = LAST; i >= S0; i--) opts.push(`<option value="${i}">${fmtDate(i, {weekday: 'short', month: 'short', day: '2-digit', year: 'numeric'})}${i === LAST ? ' (latest)' : ''}</option>`);
  $('asOf').innerHTML = opts.join('');
  $('sector').innerHTML += Object.keys(D.sectorName).sort((x, y) => D.sectorName[x].localeCompare(D.sectorName[y]))
    .map(s => `<option value="${s}">${D.sectorName[s]} (${s})</option>`).join('');
  $('rtColor').innerHTML = `<option value="below">% below ATH (that day)</option>` + D.fwdDays.map(n => `<option value="${n}">${n}D forward return</option>`).join('');
  $('fwdN').innerHTML = D.fwdDays.map(n => `<option value="${n}"${n === D.colorDays ? ' selected' : ''}>${n} days</option>`).join('');
  buildEditor(); setRules(D.rules); fillTickerList(); setupPushLinks();
  try { const d = localStorage.getItem('athRulesDraftV2');
    if (d && d !== rulesKey(D.rules)) { $('restoreRules').style.display = '';
      $('restoreRules').onclick = () => { setRules(clampRules(JSON.parse(d))); renderAll(); }; } } catch (e) {}
  renderAll();
  document.querySelectorAll('#emailCard input, #emailCard select').forEach(el => el.addEventListener('change', renderAll));
  ['asOf', 'hitType', 'fwdN', 'fwdSign'].forEach(id => $(id).addEventListener('change', renderAll));
  ['hmColor', 'hmRing'].forEach(id => $(id).addEventListener('change', () => { renderHeatmap(asOf(), view(), LB); postHeight(); }));
  $('rtColor').addEventListener('change', renderTracker);
  $('rtTicker').addEventListener('change', () => { renderTracker(); renderLeaderboard(asOf(), view()); });
  $('rtTicker').addEventListener('keydown', e => { if (e.key === 'Enter') { renderTracker(); renderLeaderboard(asOf(), view()); } });
  $('sector').addEventListener('change', () => { fillTickerList(); renderAll(); });
  $('applyBtn').onclick = renderAll;
  $('clearBtn').onclick = () => { $('sector').value = ''; $('tickerInput').value = ''; fillTickerList(); renderAll(); };
  $('tickerInput').addEventListener('keydown', e => { if (e.key === 'Enter') renderAll(); });
  $('tickerInput').addEventListener('change', () => { const tk = parseTickers($('tickerInput').value); if (tk.length === 1 && S[tk[0]]) $('rtTicker').value = tk[0]; renderAll(); });
  $('resetRules').onclick = () => { setRules(D.rules); renderAll(); toast('Back to the rules the email uses'); };
  $('defaultRules').onclick = () => { setRules(D.defaults); renderAll(); };
  $('copyRules').onclick = () => copy(rulesFileText(), `Copied ${D.rulesFile}`);
  $('copyEmail').onclick = () => copy('Subject: ' + lastEmail.subject + '\n\n' + lastEmail.body, 'Copied email text');
  $('dlRules').onclick = () => { const b = new Blob([rulesFileText()], {type: 'application/json'}), u = URL.createObjectURL(b), l = document.createElement('a');
    l.href = u; l.download = D.rulesFile; document.body.appendChild(l); l.click(); l.remove(); setTimeout(() => URL.revokeObjectURL(u), 1000); };
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
    if "--alert" in sys.argv:
        run_alerts()
        return
    rules, rules_exists = load_rules()
    sp500 = get_sp500()
    print(f"Loaded {len(sp500)} S&P 500 tickers. Downloading full history (period='max')...")
    data = download(sp500["Ticker"].tolist())
    end = pd.DatetimeIndex(data.index).max()
    start = end - pd.DateOffset(months=BACKTEST_MONTHS)
    print(f"Building dashboard {start:%Y-%m-%d} to {end:%Y-%m-%d}...")
    cal, s0, prep = prepare(data, sp500, start)
    subject = build_report(cal, s0, prep, rules, rules_exists)
    print(f"Email preview for the latest close: {subject}")
    print(f"Report: {HTML_OUT}\nCSV:    {CSV_OUT}")


if __name__ == "__main__":
    main()
