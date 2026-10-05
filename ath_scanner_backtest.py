"""
ATH Scanner - dashboard + confirmed all-time-high runs
======================================================

One script, two jobs:

  python ath_scanner_backtest.py            -> builds the interactive dashboard page
  python ath_scanner_backtest.py --alert    -> sends the daily email (the Scanner_All-Time-Highs
                                               job calls this through scanner_All-Time-High_1.py)

Both read the SAME rules file, ath_scanner_rules.json (next to this script), so what the
dashboard previews is exactly what the email sends. Edit the rules in the dashboard, copy or
download the JSON, commit it to the repo, and the next run (scheduled or "Run workflow")
emails with the new rules.

Confirmed ATH run (the email's definition of "a new all-time high worth knowing about"):
  * at least MIN_ATH_DAYS new all-time closing highs in the last WINDOW_DAYS trading days
  * the latest of them no more than MAX_DAYS_SINCE_ATH trading days ago
  * today's close within MAX_PCT_BELOW_ATH % of the all-time high (blank = no limit)
  * (optional) no close back below the breakout level since the run's first new high
    (breakout level = the old all-time high the run broke through)
A one-day spike that reverses fails the "2+ highs" and breakout checks, so it never alerts.

Email sections:
  NEW CONFIRMED RUNS  - became confirmed today (indicator filters, if any, apply here)
  RUN ENDED           - confirmed yesterday, not today: a possible peak, with the reason
  STILL RUNNING       - everything else currently confirmed (optional)

Dashboard views:
  * Email alert setup: rule editor, live email preview for any date, copy/download rules,
    links to commit the rules file and run the workflow on GitHub
  * ATH runs leaderboard (as of any date)
  * Run tracker: per ticker, bars = new-ATH days in the trailing window, colored by % below
    the ATH (or forward return), with price, ATH line and run start / end markers
  * Heatmap of the top run leaders over the last 60 trading days
  * Backtest: daily / weekly hits for alerts, confirmed runs, any new ATH close, or run ends,
    colored by forward return, plus by-sector and most-frequent tables

Requires: pip install yfinance pandas requests plotly lxml
"""

import json
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

# Starting rules, used when ath_scanner_rules.json doesn't exist yet.
# Indicator filters are blank (off) on purpose while you find the right numbers.
DEFAULT_RULES = {
    "confirm": {
        "window_days": 30,
        "min_ath_days": 2,
        "max_days_since_ath": 5,
        "max_pct_below_ath": 5.0,
        "hold_above_breakout": True,
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
PENDING_COLOR = "#f1f0ec"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"


# ============================================================
# RULES
# ============================================================

def normalize_rules(raw):
    """Merge a (possibly partial) rules dict over the defaults and clamp values."""
    r = json.loads(json.dumps(DEFAULT_RULES))
    for sec in r:
        if isinstance(raw, dict) and isinstance(raw.get(sec), dict):
            for k in r[sec]:
                if k in raw[sec]:
                    r[sec][k] = raw[sec][k]
    c = r["confirm"]
    c["window_days"] = int(min(max(int(c["window_days"] or 30), 2), MAX_WINDOW))
    c["min_ath_days"] = int(min(max(int(c["min_ath_days"] or 1), 1), c["window_days"]))
    c["max_days_since_ath"] = int(min(max(int(c["max_days_since_ath"] or 0), 0), c["window_days"] - 1))
    c["max_pct_below_ath"] = None if c["max_pct_below_ath"] in (None, "") else float(c["max_pct_below_ath"])
    c["hold_above_breakout"] = bool(c["hold_above_breakout"])
    f = r["filters"]
    for k in ("rsi_min", "rsi_max", "rvol_min", "ret_min"):
        f[k] = None if f[k] in (None, "") else float(f[k])
    for k in ("macd", "stoch"):
        f[k] = f[k] if f[k] in ("any", "bull", "bear") else "any"
    e = r["email"]
    e["send_when"] = e["send_when"] if e["send_when"] in ("changes", "new", "always") else "changes"
    e["list_running"] = bool(e["list_running"])
    return r


def load_rules():
    if RULES_FILE.exists():
        try:
            rules = normalize_rules(json.loads(RULES_FILE.read_text(encoding="utf-8")))
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


def _num(v, nd):
    return None if v is None or pd.isna(v) else round(float(v), nd)


def prepare(data, sp500, start):
    """Per-ticker arrays on a shared trading-day calendar.

    The calendar starts MAX_WINDOW+1 trading days before `start` so trailing-window
    counts are complete from the first displayed day. Prices are rounded to cents and
    every rule is evaluated on these rounded values, in Python (email) and in the page,
    so both always agree.
    """
    idx = pd.DatetimeIndex(data.index).sort_values()
    s0 = int(idx.searchsorted(start))
    cal = idx[max(0, s0 - (MAX_WINDOW + 1)):]
    out = {}
    for _, row in sp500.iterrows():
        t = row["Ticker"]
        try:
            full = data[t].dropna()
        except KeyError:
            continue
        if len(full) < 2:
            continue
        close, high, low, vol = full["Close"], full["High"], full["Low"], full["Volume"]
        avg_vol = vol.shift(1).rolling(RVOL_LOOKBACK, min_periods=1).mean()
        ind = pd.DataFrame({
            "c": close,
            "rsi": rsi(close),
            "rvol": vol / avg_vol.where(avg_vol > 0),
        })
        ml, sl = macd(close)
        k, d = stochastic(high, low, close)
        ind["m"] = (ml > sl).astype(int)
        ind["k"] = (k > d).astype(int)
        before = close[close.index < cal[0]]
        sl_df = ind.reindex(cal)
        out[t] = {
            "a0": _num(before.max(), 2) if len(before) else None,
            "c": [_num(v, 2) for v in sl_df["c"]],
            "r": [_num(v, 1) for v in sl_df["rsi"]],
            "v": [_num(v, 2) for v in sl_df["rvol"]],
            "m": [None if pd.isna(v) else int(v) for v in sl_df["m"]],
            "k": [None if pd.isna(v) else int(v) for v in sl_df["k"]],
            "company": str(row["Company"]),
            "sector": str(row["Sector_ETF"]),
            "sectorName": str(row["Sector"]),
        }
    return cal, s0 - max(0, s0 - (MAX_WINDOW + 1)), out


# ============================================================
# CONFIRMED-RUN LOGIC  (mirrored line by line in the page's computeTicker())
# ============================================================

REASONS = {
    "few": "fewer than {min} new ATH closes in {win} days",
    "stale": "no new ATH close in {days} days",
    "below": "more than {pct}% below the ATH",
    "brk": "closed back below the breakout level",
}


def compute_states(s, cf):
    c = s["c"]
    n, K = len(c), cf["window_days"]
    st = {key: [None] * n for key in ("lvl", "below", "dsa", "brk", "first", "why", "pmb")}
    st["ath"], st["cnt"], st["conf"] = [False] * n, [0] * n, [False] * n
    pm, last = s["a0"], None
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
            st["conf"][i] = st["conf"][i - 1] if i > 0 else False
            st["why"][i] = st["why"][i - 1] if i > 0 else None
            continue
        hold = True
        if f is not None:
            st["brk"][i] = st["pmb"][f]
            for j in range(f, i + 1):
                if c[j] is not None and c[j] < st["brk"][i]:
                    hold = False
                    break
        why = None
        if cf["hold_above_breakout"] and f is not None and not hold:
            why = "brk"
        elif cf["max_pct_below_ath"] is not None and st["below"][i] > cf["max_pct_below_ath"]:
            why = "below"
        elif st["dsa"][i] is None or st["dsa"][i] > cf["max_days_since_ath"]:
            why = "stale"
        elif k < cf["min_ath_days"]:
            why = "few"
        st["conf"][i] = why is None
        st["why"][i] = why
    return st


def day_return(c, i):
    if i < 1 or c[i] is None or c[i - 1] is None:
        return None
    return (c[i] / c[i - 1] - 1) * 100


def fwd_return(c, i, n):
    if i + n >= len(c) or c[i] is None or c[i + n] is None:
        return None
    return (c[i + n] / c[i] - 1) * 100


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


def reason_text(why, cf):
    if why is None:
        return ""
    pct = cf["max_pct_below_ath"]
    return REASONS[why].format(min=cf["min_ath_days"], win=cf["window_days"],
                               days=cf["max_days_since_ath"],
                               pct=(f"{pct:g}" if pct is not None else "?"))


# ============================================================
# EMAIL  (the page's emailText() builds the same text)
# ============================================================

def _g(v):
    return "n/a" if v is None else f"{v:g}"


def _money(v):
    return "n/a" if v is None else f"${v:,.2f}"


def _pct(v, nd=1, sign=True):
    if v is None:
        return "n/a"
    return f"{v:+.{nd}f}%" if sign else f"{v:.{nd}f}%"


def rules_sentence(rules):
    c = rules["confirm"]
    parts = [f"at least {c['min_ath_days']} new all-time closing high{'s' if c['min_ath_days'] != 1 else ''} "
             f"in the last {c['window_days']} trading days",
             "the latest today" if c["max_days_since_ath"] == 0
             else f"the latest within {c['max_days_since_ath']} day{'s' if c['max_days_since_ath'] != 1 else ''}"]
    if c["max_pct_below_ath"] is not None:
        parts.append(f"close within {c['max_pct_below_ath']:g}% of the ATH")
    if c["hold_above_breakout"]:
        parts.append("no close back below the breakout level")
    return "Confirmed ATH run = " + ", ".join(parts) + "."


def filters_sentence(f):
    p = []
    if f["rsi_min"] is not None or f["rsi_max"] is not None:
        lo = "" if f["rsi_min"] is None else f"{f['rsi_min']:g}"
        hi = "" if f["rsi_max"] is None else f"{f['rsi_max']:g}"
        p.append(f"RSI {lo or 'any'}-{hi or 'any'}")
    if f["rvol_min"] is not None:
        p.append(f"RVOL >= {f['rvol_min']:g}")
    if f["ret_min"] is not None:
        p.append(f"day return >= {f['ret_min']:g}%")
    if f["macd"] != "any":
        p.append(f"MACD {'bullish' if f['macd'] == 'bull' else 'bearish'}")
    if f["stoch"] != "any":
        p.append(f"Stochastic {'bullish' if f['stoch'] == 'bull' else 'bearish'}")
    return "Indicator filters on new alerts: " + (", ".join(p) if p else "none")


def build_email(cal, prep, states, rules, a):
    cf = rules["confirm"]
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

    def last_ath_date(st, i):
        d = st["dsa"][i]
        return None if d is None else pd.Timestamp(cal[i - d])

    def when(st, i):
        d = st["dsa"][i]
        if d is None:
            return "n/a"
        return "today" if d == 0 else f"{d} day{'s' if d != 1 else ''} ago ({last_ath_date(st, i):%b %d})"

    L = ["=" * 60, f"  ALL-TIME HIGH SCANNER  -  {day_s} close", "=" * 60, "",
         rules_sentence(rules), filters_sentence(rules["filters"]), ""]

    L.append(f"NEW CONFIRMED RUNS ({len(new)})")
    L.append("-" * 60)
    if not new:
        L.append("   none today")
    for n_, t in enumerate(new, 1):
        s, st = prep[t], states[t]
        L.append(f"{n_}. {t} - {s['company']} ({s['sector']})")
        L.append(f"   New ATH closes ({cf['window_days']}D): {st['cnt'][a]}  |  last: {when(st, a)}")
        L.append(f"   Close {_money(s['c'][a])}  |  ATH {_money(st['lvl'][a])}  |  "
                 f"{_pct(st['below'][a], 1, False)} below  |  breakout {_money(st['brk'][a])}")
        L.append(f"   Day {_pct(day_return(s['c'], a))}  |  RVOL {_g(s['v'][a])}x  |  "
                 f"RSI {_g(s['r'][a])}  |  "
                 f"MACD {'bullish' if s['m'][a] == 1 else 'bearish'}  |  "
                 f"Stoch {'bullish' if s['k'][a] == 1 else 'bearish'}")
    L.append("")

    L.append(f"RUN ENDED - possible peak ({len(ended)})")
    L.append("-" * 60)
    if not ended:
        L.append("   none today")
    for n_, t in enumerate(ended, 1):
        s, st = prep[t], states[t]
        pk = last_ath_date(st, a)
        L.append(f"{n_}. {t} - {s['company']} ({s['sector']})")
        L.append(f"   Peak close {_money(st['lvl'][a])}" + (f" on {pk:%b %d}" if pk is not None else "") +
                 f"  |  now {_money(s['c'][a])}, {_pct(st['below'][a], 1, False)} below  |  "
                 f"day {_pct(day_return(s['c'], a))}")
        L.append(f"   Why: {reason_text(st['why'][a], cf)}  |  new ATH closes ({cf['window_days']}D): {st['cnt'][a]}")
    L.append("")

    if rules["email"]["list_running"]:
        L.append(f"STILL RUNNING ({len(running)})  ticker: new ATH closes in {cf['window_days']}D / % below ATH")
        L.append("-" * 60)
        if not running:
            L.append("   none")
        row = []
        for t in running:
            row.append(f"{t}: {states[t]['cnt'][a]} / {_pct(states[t]['below'][a], 1, False)}")
            if len(row) == 3:
                L.append("   " + "    ".join(row))
                row = []
        if row:
            L.append("   " + "    ".join(row))
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
    cal, s0, prep = prepare(data, sp500, end - pd.DateOffset(months=1))
    states = {t: compute_states(s, rules["confirm"]) for t, s in prep.items()}
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
    cf, rows = rules["confirm"], []
    K = cf["window_days"]
    for t in sorted(prep):
        s, st = prep[t], states[t]
        for i in range(s0, len(cal)):
            if s["c"][i] is None:
                continue
            is_new, filtered, is_end = events(s, st, i, rules)
            if not (st["cnt"][i] or st["conf"][i] or is_end):
                continue
            row = {
                "Date": f"{pd.Timestamp(cal[i]):%Y-%m-%d}", "Ticker": t, "Company": s["company"],
                "Sector_ETF": s["sector"], "Close": s["c"][i], "Is_New_ATH": st["ath"][i],
                "ATH_Close": st["lvl"][i], "Pct_Below_ATH": round(st["below"][i], 2),
                f"ATH_Days_{K}D": st["cnt"][i], "Days_Since_ATH": st["dsa"][i],
                "Breakout_Level": st["brk"][i], "Confirmed": st["conf"][i],
                "New_Alert": is_new, "New_Run_Filtered_Out": filtered, "Run_Ended": is_end,
                "Not_Confirmed_Because": reason_text(st["why"][i], cf),
                "Day_Return%": _num(day_return(s["c"], i), 2), "RVOL": s["v"][i], "RSI": s["r"][i],
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
    states = {t: compute_states(s, rules["confirm"]) for t, s in prep.items()}
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
        "s": {t: {k: prep[t][k] for k in ("a0", "c", "r", "v", "m", "k")} for t in keep},
        "rules": rules, "defaults": normalize_rules(DEFAULT_RULES), "rulesFile": RULES_FILE_NAME,
        "rulesExists": rules_exists, "gh": github_links(rules_exists), "maxWindow": MAX_WINDOW,
        "fwdDays": FORWARD_DAYS, "colorDays": COLOR_FWD_DAYS, "cap": COLOR_CAP_PCT,
        "gradient": GRADIENT, "pending": PENDING_COLOR, "heatDays": HEATMAP_DAYS,
        "heatRows": HEATMAP_ROWS, "scannerName": SCANNER_NAME,
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
                        f"confirmed runs · {len(groups['new'])} new · {len(groups['ended'])} ended today",
        }, indent=2), encoding="utf-8")
    return subject


PAGE_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>ATH Runs Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
__PLOTLY__
<style>
  body { background:__SURFACE__; color:__INK__; font-family:Inter,'Segoe UI',Arial,sans-serif; margin:0; padding:24px 16px; }
  .wrap { max-width:1280px; margin:0 auto; }
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
  .ctl input.n { width:68px; }
  #tickerInput { width:220px; }
  button, a.btn { font:inherit; font-size:13px; padding:7px 12px; border:1px solid #c9c8c2; border-radius:6px; background:#fff;
                  color:__INK__; cursor:pointer; text-decoration:none; display:inline-block; }
  button:hover, a.btn:hover { background:#f0efec; }
  button.primary, a.primary { background:__INK__; color:#fff; border-color:__INK__; }
  button.primary:hover, a.primary:hover { background:#333; }
  .hint, .note { font-size:12px; color:__INK2__; line-height:1.5; }
  .note { margin:8px 0 2px; }
  .grid2 { display:grid; grid-template-columns:minmax(300px, 420px) 1fr; gap:18px; }
  @media (max-width:900px) { .grid2 { grid-template-columns:1fr; } }
  .form { display:grid; grid-template-columns:auto 1fr; gap:8px 10px; align-items:center; font-size:13px; }
  .form label { color:__INK2__; }
  .form input[type=number], .form select { font:inherit; font-size:14px; padding:5px 8px; border:1px solid #c9c8c2;
         border-radius:6px; width:90px; background:#fff; }
  .form span input[type=number] { width:62px; }
  .chip { display:inline-block; font-size:12px; padding:3px 9px; border-radius:999px; border:1px solid; margin-left:8px; vertical-align:2px; }
  .chip.ok { color:#0b5a24; border-color:#9fd3ad; background:#eef8f0; }
  .chip.edit { color:#6b4e00; border-color:#f0d58a; background:#fff6e0; }
  pre#emailBody { background:#f7f6f2; border:1px solid __GRID__; border-radius:8px; padding:12px; font-size:12px;
                  line-height:1.45; max-height:460px; overflow:auto; white-space:pre-wrap; margin:6px 0 0; }
  .subject { font-size:13px; font-weight:600; padding:8px 10px; border:1px solid __GRID__; border-radius:8px; background:#fff; }
  .sendflag { font-size:12px; margin:6px 0 0; }
  .btns { display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }
  table { border-collapse:collapse; width:100%; font-size:13px; }
  th, td { text-align:left; padding:6px 8px; border-bottom:1px solid __GRID__; white-space:nowrap; }
  th { color:__INK2__; font-weight:500; }
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
  <h2>Email alert setup <span class="chip ok" id="ruleChip">Matches the email rules</span></h2>
  <div class="grid2">
    <div>
      <h3>Confirmed ATH run</h3>
      <div class="form">
        <label for="r_window">Window (trading days)</label><input id="r_window" type="number" min="2" step="1">
        <label for="r_min">Min new ATH closes in window</label><input id="r_min" type="number" min="1" step="1">
        <label for="r_dsa">Latest ATH within (days)</label><input id="r_dsa" type="number" min="0" step="1">
        <label for="r_pct">Max % below ATH</label><input id="r_pct" type="number" min="0" step="0.5" placeholder="no limit">
        <label for="r_hold">Must stay above breakout</label><select id="r_hold"><option value="1">yes</option><option value="0">no</option></select>
      </div>
      <h3>Indicator filters on new alerts <span class="hint" style="text-transform:none;letter-spacing:0">(blank = off)</span></h3>
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
    The count tops out and fades as the run loses steam; the color shows how far price has slipped from the ATH that same day,
    so a tall bar turning red is the peak rolling over. ▲ = run confirmed, ▼ = run ended. Black line = close, dotted = ATH, gray dashed = breakout level.</div>
</div>

<div class="card">
  <h2 id="hmTitle">Run heatmap</h2>
  <div id="hmChart"></div>
  <div class="note">Rows = tickers with the most new ATH closes in the window as of the selected date. Cell color = % below the all-time high
    that day (darkest green = closed at a new ATH, marked •). Click a row to load it in the run tracker.</div>
</div>

<div class="card">
  <h2>Backtest - how the hits did afterwards</h2>
  <div class="row" style="margin-bottom:8px">
    <label class="ctl">Hits <select id="hitType">
      <option value="new">Email alerts (new confirmed runs, after filters)</option>
      <option value="conf">Every day a ticker is in a confirmed run</option>
      <option value="ath">Any new ATH close (no confirmation)</option>
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
const REASONS = {few: 'fewer than {min} new ATH closes in {win} days', stale: 'no new ATH close in {days} days',
  below: 'more than {pct}% below the ATH', brk: 'closed back below the breakout level'};

// ---------- formatting ----------
const hexRgb = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
function grad(x) {               // x in [-1, 1] on the red-gray-green scale
  x = Math.max(-1, Math.min(1, x)); const g = D.gradient;
  for (let k = 0; k < g.length - 1; k++) {
    const [p0, c0] = g[k], [p1, c1] = g[k + 1];
    if (x <= p1) { const f = (x - p0) / (p1 - p0), a = hexRgb(c0), b = hexRgb(c1);
      return '#' + a.map((v, j) => Math.round(v + (b[j] - v) * f).toString(16).padStart(2, '0')).join(''); }
  }
  return g[g.length - 1][1];
}
const colorRet = v => (v === null || v === undefined || Number.isNaN(v)) ? D.pending : grad(v / D.cap);
const colorBelow = p => p === null || p === undefined ? D.pending : grad(1 - 2 * Math.min(p, D.cap) / D.cap);
const inkOn = x => (x === null || Math.abs(x) < 0.4) ? INK : '#ffffff';
const pct = (v, d = 2) => v === null || v === undefined ? 'n/a' : (v > 0 ? '+' : '') + v.toFixed(d) + '%';
const pctU = (v, d = 1) => v === null || v === undefined ? 'n/a' : v.toFixed(d) + '%';
const money = v => v === null || v === undefined ? 'n/a' : '$' + v.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
const g = v => (Math.round(v * 1e6) / 1e6).toString();
const mean = a => a.length ? a.reduce((s, v) => s + v, 0) / a.length : null;
function median(a) { if (!a.length) return null; const s = [...a].sort((x, y) => x - y), m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; }
const dt = i => new Date(D.dates[i] + 'T00:00:00Z');
const fmtDate = (i, o) => dt(i).toLocaleDateString('en-US', Object.assign({timeZone: 'UTC'}, o || {month: 'short', day: '2-digit', year: 'numeric'}));
const mmmdd = i => fmtDate(i, {month: 'short', day: '2-digit'});
function toast(msg) { const t = $('toast'); t.textContent = msg; t.classList.add('on'); setTimeout(() => t.classList.remove('on'), 1800); }
function parseTickers(v) { return v.toUpperCase().split(/[\s,;]+/).map(x => x.replace('.', '-')).filter(Boolean); }

// ---------- rules ----------
function clampRules(r) {
  const c = r.confirm, int = (v, d) => (v === null || v === '' || Number.isNaN(+v)) ? d : Math.round(+v);
  c.window_days = Math.min(Math.max(int(c.window_days, 30), 2), D.maxWindow);
  c.min_ath_days = Math.min(Math.max(int(c.min_ath_days, 1), 1), c.window_days);
  c.max_days_since_ath = Math.min(Math.max(int(c.max_days_since_ath, 0), 0), c.window_days - 1);
  c.max_pct_below_ath = (c.max_pct_below_ath === null || c.max_pct_below_ath === '' || Number.isNaN(+c.max_pct_below_ath)) ? null : +c.max_pct_below_ath;
  c.hold_above_breakout = !!c.hold_above_breakout;
  return r;
}
const numOrNull = id => { const v = $(id).value.trim(); return v === '' || Number.isNaN(+v) ? null : +v; };
function readRules() {
  return clampRules({
    confirm: {window_days: numOrNull('r_window'), min_ath_days: numOrNull('r_min'), max_days_since_ath: numOrNull('r_dsa'),
      max_pct_below_ath: numOrNull('r_pct'), hold_above_breakout: $('r_hold').value === '1'},
    filters: {rsi_min: numOrNull('f_rsimin'), rsi_max: numOrNull('f_rsimax'), rvol_min: numOrNull('f_rvol'),
      ret_min: numOrNull('f_ret'), macd: $('f_macd').value, stoch: $('f_stoch').value},
    email: {send_when: $('e_when').value, list_running: $('e_run').value === '1'},
  });
}
function setRules(r) {
  const s = (id, v) => { $(id).value = v === null || v === undefined ? '' : v; };
  s('r_window', r.confirm.window_days); s('r_min', r.confirm.min_ath_days); s('r_dsa', r.confirm.max_days_since_ath);
  s('r_pct', r.confirm.max_pct_below_ath); $('r_hold').value = r.confirm.hold_above_breakout ? '1' : '0';
  s('f_rsimin', r.filters.rsi_min); s('f_rsimax', r.filters.rsi_max); s('f_rvol', r.filters.rvol_min);
  s('f_ret', r.filters.ret_min); $('f_macd').value = r.filters.macd; $('f_stoch').value = r.filters.stoch;
  $('e_when').value = r.email.send_when; $('e_run').value = r.email.list_running ? '1' : '0';
}
const rulesKey = r => JSON.stringify(r);
const reasonText = (w, c) => !w ? '' : REASONS[w].replace('{min}', c.min_ath_days).replace('{win}', c.window_days)
  .replace('{days}', c.max_days_since_ath).replace('{pct}', c.max_pct_below_ath === null ? '?' : g(c.max_pct_below_ath));

// ---------- confirmed-run logic (mirrors compute_states() in the Python script) ----------
function computeTicker(s, cf) {
  const c = s.c, n = c.length, K = cf.window_days;
  const st = {ath: Array(n).fill(false), cnt: Array(n).fill(0), conf: Array(n).fill(false)};
  for (const k of ['lvl', 'below', 'dsa', 'brk', 'first', 'why', 'pmb']) st[k] = Array(n).fill(null);
  let pm = s.a0, last = null;
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
    if (x === null) { st.conf[i] = i > 0 ? st.conf[i - 1] : false; st.why[i] = i > 0 ? st.why[i - 1] : null; continue; }
    let hold = true;
    if (f !== null) {
      st.brk[i] = st.pmb[f];
      for (let j = f; j <= i; j++) if (c[j] !== null && c[j] < st.brk[i]) { hold = false; break; }
    }
    let why = null;
    if (cf.hold_above_breakout && f !== null && !hold) why = 'brk';
    else if (cf.max_pct_below_ath !== null && st.below[i] > cf.max_pct_below_ath) why = 'below';
    else if (st.dsa[i] === null || st.dsa[i] > cf.max_days_since_ath) why = 'stale';
    else if (k < cf.min_ath_days) why = 'few';
    st.conf[i] = why === null; st.why[i] = why;
  }
  return st;
}
const dayRet = (c, i) => (i < 1 || c[i] === null || c[i - 1] === null) ? null : (c[i] / c[i - 1] - 1) * 100;
const fwdRet = (c, i, n) => (i + n >= c.length || c[i] === null || c[i + n] === null) ? null : (c[i + n] / c[i] - 1) * 100;
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
  const key = rulesKey(RULES.confirm);
  if (key === STKEY) return;
  ST = {}; T.forEach(t => { ST[t] = computeTicker(S[t], RULES.confirm); }); STKEY = key;
}

// ---------- email preview (mirrors build_email()) ----------
function rulesSentence(r) {
  const c = r.confirm, p = [`at least ${c.min_ath_days} new all-time closing high${c.min_ath_days !== 1 ? 's' : ''} in the last ${c.window_days} trading days`,
    c.max_days_since_ath === 0 ? 'the latest today' : `the latest within ${c.max_days_since_ath} day${c.max_days_since_ath !== 1 ? 's' : ''}`];
  if (c.max_pct_below_ath !== null) p.push(`close within ${g(c.max_pct_below_ath)}% of the ATH`);
  if (c.hold_above_breakout) p.push('no close back below the breakout level');
  return 'Confirmed ATH run = ' + p.join(', ') + '.';
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
const pyMoney = v => v === null || v === undefined ? 'n/a' : '$' + v.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
const pyPct = (v, nd = 1, sign = true) => v === null || v === undefined ? 'n/a' : (sign && v >= 0 ? '+' : '') + v.toFixed(nd) + '%';
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
  const cf = RULES.confirm, day = fmtDate(a), {nw, en, ru} = emailGroups(a), L = [], bar = '='.repeat(60), dash = '-'.repeat(60);
  const lastAth = (st, i) => st.dsa[i] === null ? null : i - st.dsa[i];
  const when = (st, i) => { const d = st.dsa[i]; return d === null ? 'n/a' : d === 0 ? 'today' : `${d} day${d !== 1 ? 's' : ''} ago (${mmmdd(i - d)})`; };
  L.push(bar, `  ALL-TIME HIGH SCANNER  -  ${day} close`, bar, '', rulesSentence(RULES), filtersSentence(RULES.filters), '');
  L.push(`NEW CONFIRMED RUNS (${nw.length})`, dash);
  if (!nw.length) L.push('   none today');
  nw.forEach((t, n) => {
    const s = S[t], st = ST[t];
    L.push(`${n + 1}. ${t} - ${D.company[t]} (${D.sector[t]})`);
    L.push(`   New ATH closes (${cf.window_days}D): ${st.cnt[a]}  |  last: ${when(st, a)}`);
    L.push(`   Close ${pyMoney(s.c[a])}  |  ATH ${pyMoney(st.lvl[a])}  |  ${pyPct(st.below[a], 1, false)} below  |  breakout ${pyMoney(st.brk[a])}`);
    L.push(`   Day ${pyPct(dayRet(s.c, a))}  |  RVOL ${s.v[a] ?? 'n/a'}x  |  RSI ${s.r[a] ?? 'n/a'}  |  MACD ${s.m[a] === 1 ? 'bullish' : 'bearish'}  |  Stoch ${s.k[a] === 1 ? 'bullish' : 'bearish'}`);
  });
  L.push('', `RUN ENDED - possible peak (${en.length})`, dash);
  if (!en.length) L.push('   none today');
  en.forEach((t, n) => {
    const s = S[t], st = ST[t], pk = lastAth(st, a);
    L.push(`${n + 1}. ${t} - ${D.company[t]} (${D.sector[t]})`);
    L.push(`   Peak close ${pyMoney(st.lvl[a])}${pk !== null ? ' on ' + mmmdd(pk) : ''}  |  now ${pyMoney(s.c[a])}, ${pyPct(st.below[a], 1, false)} below  |  day ${pyPct(dayRet(s.c, a))}`);
    L.push(`   Why: ${reasonText(st.why[a], cf)}  |  new ATH closes (${cf.window_days}D): ${st.cnt[a]}`);
  });
  L.push('');
  if (RULES.email.list_running) {
    L.push(`STILL RUNNING (${ru.length})  ticker: new ATH closes in ${cf.window_days}D / % below ATH`, dash);
    if (!ru.length) L.push('   none');
    for (let k = 0; k < ru.length; k += 3)
      L.push('   ' + ru.slice(k, k + 3).map(t => `${t}: ${ST[t].cnt[a]} / ${pyPct(ST[t].below[a], 1, false)}`).join('    '));
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
  try { if (!same) localStorage.setItem('athRulesDraft', rulesKey(RULES)); } catch (e) {}
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
  return ['no', 'Not confirmed', 4];
}
function renderTiles(a, v) {
  let conf = 0, nw = 0, en = 0, ath = 0;
  T.forEach(t => { if (!v.ok(t)) return; const st = ST[t], [isNew, , isEnd] = events(S[t], st, a, RULES);
    if (st.conf[a]) conf++; if (isNew) nw++; if (isEnd) en++; if (st.ath[a]) ath++; });
  const tl = [[conf, 'confirmed runs'], [nw, 'new alerts'], [en, 'runs ended (possible peak)'], [ath, 'new ATH closes that day']];
  $('tiles').innerHTML = tl.map(([x, l]) => `<div class="tile"><div class="v">${x}</div><div class="l">${l}</div></div>`).join('');
}
let showAllLb = false;
function renderLeaderboard(a, v) {
  const K = RULES.confirm.window_days;
  const rows = T.filter(t => v.ok(t) && S[t].c[a] !== null && (ST[t].cnt[a] > 0 || ST[t].conf[a] || events(S[t], ST[t], a, RULES)[2]))
    .map(t => ({t, s: statusOf(t, a)}))
    .sort((x, y) => x.s[2] - y.s[2] || ST[y.t].cnt[a] - ST[x.t].cnt[a] || (ST[x.t].below[a] || 0) - (ST[y.t].below[a] || 0));
  $('lbTitle').textContent = `ATH runs leaderboard - ${fmtDate(a)} close`;
  const shown = showAllLb ? rows : rows.slice(0, 40);
  const head = `<tr><th>Ticker</th><th>Company</th><th>Sector</th><th>Status</th><th class="num">New ATHs (${K}D)</th>` +
    `<th class="num">Days since ATH</th><th class="num">% below ATH</th><th class="num">Close</th><th class="num">ATH close</th>` +
    `<th class="num">Breakout</th><th>Run started</th><th class="num">Day</th><th>Why not confirmed / ended</th></tr>`;
  $('lbTable').innerHTML = head + (shown.length ? shown.map(({t, s}) => {
    const st = ST[t], b = st.below[a], f = st.first[a];
    return `<tr class="click${t === $('rtTicker').value ? ' sel' : ''}" data-t="${t}"><td><b>${t}</b></td><td>${D.company[t]}</td><td>${D.sector[t]}</td>` +
      `<td><span class="st ${s[0]}">${s[1]}</span></td><td class="num">${st.cnt[a]}</td><td class="num">${st.dsa[a] ?? 'n/a'}</td>` +
      `<td class="num"><span class="sw" style="background:${colorBelow(b)}"></span>${pctU(b)}</td><td class="num">${money(S[t].c[a])}</td>` +
      `<td class="num">${money(st.lvl[a])}</td><td class="num">${money(st.brk[a])}</td><td>${f === null ? '' : mmmdd(f)}</td>` +
      `<td class="num">${pct(dayRet(S[t].c, a), 1)}</td><td class="hint">${st.conf[a] ? '' : reasonText(st.why[a], RULES.confirm)}</td></tr>`;
  }).join('') : '<tr><td colspan="13" class="hint">No tickers with a new ATH close in the window for this selection.</td></tr>');
  $('lbNote').innerHTML = `${rows.length} tickers with at least one new ATH close in the last ${K} trading days (or a run that just ended).` +
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
function renderTracker() {
  let t = $('rtTicker').value.trim().toUpperCase().replace('.', '-');
  const mode = $('rtColor').value, K = RULES.confirm.window_days, a = asOf();
  if (!S[t]) { $('rtSummary').innerHTML = t ? `<b>${t}</b> made no new all-time closing high in this period (or isn't an S&amp;P 500 ticker).` : '';
    Plotly.purge('rtChart'); return; }
  const s = S[t], st = ST[t], xs = [], ys = [], cols = [], cd = [], pat = [];
  for (let i = S0; i < NDAYS; i++) {
    if (s.c[i] === null) continue;
    const cl = colorFor(mode, t, i);
    xs.push(D.dates[i]); ys.push(st.cnt[i]); cols.push(cl.col); pat.push(cl.x === null ? '/' : '');
    const f = mode === 'below' ? null : fwdRet(s.c, i, +mode);
    cd.push([money(s.c[i]), money(st.lvl[i]), pctU(st.below[i]), st.dsa[i] ?? 'n/a', st.conf[i] ? 'confirmed run' : 'not confirmed: ' + reasonText(st.why[i], RULES.confirm),
      st.ath[i] ? 'new ATH close' : '', mode === 'below' ? '' : `<br>${mode}D forward return: ${f === null ? 'not yet available' : pct(f)}`]);
  }
  const bars = {type: 'bar', x: xs, y: ys, name: `New ATHs (${K}D)`, marker: {color: cols, line: {width: 0}, pattern: {shape: pat, fgcolor: '#a9a8a2', size: 5, solidity: .25}},
    customdata: cd, width: 0.85 * DAY, hovertemplate: `<b>%{x|%b %d, %Y}</b> %{customdata[5]}<br>New ATH closes in last ${K}D: <b>%{y}</b><br>` +
      'Close %{customdata[0]} · ATH %{customdata[1]} · %{customdata[2]} below<br>Days since ATH: %{customdata[3]}<br>%{customdata[4]}%{customdata[6]}<extra></extra>'};
  const px = [], py = [], al = [], bk = [], up = {x: [], y: []}, dn = {x: [], y: []};
  for (let i = S0; i < NDAYS; i++) {
    if (s.c[i] === null) continue;
    px.push(D.dates[i]); py.push(s.c[i]); al.push(st.lvl[i]); bk.push(st.conf[i] ? st.brk[i] : null);
    const [isNew, filt, isEnd] = events(s, st, i, RULES);
    if (isNew || filt) { up.x.push(D.dates[i]); up.y.push(s.c[i]); }
    if (isEnd) { dn.x.push(D.dates[i]); dn.y.push(s.c[i]); }
  }
  const traces = [bars,
    {type: 'scatter', x: px, y: py, yaxis: 'y2', mode: 'lines', name: 'Close', line: {color: INK, width: 1.6}, hovertemplate: 'Close $%{y:,.2f}<extra></extra>'},
    {type: 'scatter', x: px, y: al, yaxis: 'y2', mode: 'lines', name: 'ATH close', line: {color: '#0b5a24', width: 1.2, dash: 'dot'}, hovertemplate: 'ATH $%{y:,.2f}<extra></extra>'},
    {type: 'scatter', x: px, y: bk, yaxis: 'y2', mode: 'lines', name: 'Breakout level', line: {color: '#a9a8a2', width: 1, dash: 'dash'}, connectgaps: false, hoverinfo: 'skip'},
    {type: 'scatter', x: up.x, y: up.y, yaxis: 'y2', mode: 'markers', name: 'Run confirmed', marker: {symbol: 'triangle-up', size: 12, color: '#0b5a24', line: {color: '#fff', width: 1}}, hovertemplate: 'Run confirmed %{x|%b %d}<extra></extra>'},
    {type: 'scatter', x: dn.x, y: dn.y, yaxis: 'y2', mode: 'markers', name: 'Run ended', marker: {symbol: 'triangle-down', size: 12, color: '#8e1b1b', line: {color: '#fff', width: 1}}, hovertemplate: 'Run ended %{x|%b %d}<extra></extra>'},
    scaleTrace(mode, xs[0])];
  const lay = {barmode: 'overlay', plot_bgcolor: SURF, paper_bgcolor: '#fff', height: 480, margin: {l: 50, r: 60, t: 60, b: 40},
    font: {family: 'Inter, Segoe UI, Arial, sans-serif', color: INK2, size: 12}, hoverlabel: {bgcolor: '#fff', font: {color: INK}},
    legend: {orientation: 'h', x: 0, y: 1.02, yanchor: 'bottom', font: {size: 11}}, hovermode: 'x unified',
    xaxis: {showgrid: false, linecolor: GRIDC, rangeslider: {visible: true, thickness: 0.06}, rangebreaks: [{bounds: ['sat', 'mon']}, {values: D.holidays}]},
    yaxis: {title: {text: `New ATH closes, last ${K}D`, font: {size: 11}}, gridcolor: GRIDC, zeroline: false, range: [0, Math.max(K, 1) * 1.02], fixedrange: true},
    yaxis2: {overlaying: 'y', side: 'right', showgrid: false, tickprefix: '$', title: {text: 'Close', font: {size: 11}}},
    shapes: a < LAST ? [{type: 'line', xref: 'x', yref: 'paper', x0: D.dates[a], x1: D.dates[a], y0: 0, y1: 1, line: {color: '#2a78d6', width: 1, dash: 'dot'}}] : []};
  Plotly.react('rtChart', traces, lay, {displaylogo: false, responsive: true});
  // summary: peak of the count and where price is vs the ATH
  let pkI = null; for (let i = S0; i <= a; i++) if (st.cnt[i] > 0 && (pkI === null || st.cnt[i] > st.cnt[pkI])) pkI = i;
  const sNow = statusOf(t, a);
  $('rtTitle').textContent = `Run tracker - ${t} · ${D.company[t]}`;
  $('rtSummary').innerHTML = `<span class="st ${sNow[0]}">${sNow[1]}</span> as of ${fmtDate(a)} · <b>${st.cnt[a]}</b> new ATH closes in the last ${K} days · ` +
    `last ATH ${st.dsa[a] === null ? 'n/a' : st.dsa[a] === 0 ? 'today' : st.dsa[a] + ' days ago (' + mmmdd(a - st.dsa[a]) + ')'} at ${money(st.lvl[a])} · ` +
    `close ${money(S[t].c[a])} (${pctU(st.below[a])} below)` +
    (pkI !== null ? ` · run count peaked at <b>${st.cnt[pkI]}</b> on ${mmmdd(pkI)}` : '');
  $('rtHint').textContent = mode === 'below' ? 'Known on the day - no look-ahead.' : 'Hindsight: what happened next. Hatched = not available yet.';
}

// ---------- heatmap ----------
function renderHeatmap(a, v, lbRows) {
  const K = RULES.confirm.window_days, i0 = Math.max(S0, a - D.heatDays + 1);
  const pick = lbRows.filter(r => r.s[2] <= 3).slice(0, D.heatRows).map(r => r.t);
  if (pick.length < D.heatRows) lbRows.forEach(r => { if (pick.length < D.heatRows && !pick.includes(r.t)) pick.push(r.t); });
  $('hmTitle').textContent = `Run heatmap - top ${pick.length} by new ATH closes (${K}D), ${mmmdd(i0)} to ${fmtDate(a)}`;
  if (!pick.length) { Plotly.purge('hmChart'); return; }
  const xs = []; for (let i = i0; i <= a; i++) xs.push(D.dates[i]);
  const ys = [...pick].reverse(), z = [], txt = [], cd = [];
  ys.forEach(t => {
    const zr = [], tr = [], cr = [];
    for (let i = i0; i <= a; i++) {
      const b = ST[t].below[i];
      zr.push(b === null ? null : Math.min(b, D.cap)); tr.push(ST[t].ath[i] ? '•' : '');
      cr.push([money(S[t].c[i]), pctU(b), ST[t].cnt[i], ST[t].ath[i] ? ' · new ATH close' : '', ST[t].conf[i] ? 'in confirmed run' : 'not confirmed']);
    }
    z.push(zr); txt.push(tr); cd.push(cr);
  });
  const trace = {type: 'heatmap', x: xs, y: ys, z, text: txt, texttemplate: '%{text}', textfont: {color: '#fff', size: 10}, customdata: cd,
    zmin: 0, zmax: D.cap, colorscale: [...D.gradient].reverse().map(([p, c]) => [(1 - p) / 2, c]), xgap: 1, ygap: 1,
    colorbar: {thickness: 10, len: 0.6, outlinewidth: 0, tickvals: [0, D.cap / 2, D.cap], ticktext: ['at ATH', `${D.cap / 2}%`, `≥ ${D.cap}%`],
      title: {text: '% below ATH', side: 'right', font: {size: 11}}},
    hovertemplate: '<b>%{y}</b> %{x|%b %d, %Y}%{customdata[3]}<br>Close %{customdata[0]} · %{customdata[1]} below ATH<br>' +
      `New ATH closes (${K}D): %{customdata[2]} · %{customdata[4]}<extra></extra>`};
  Plotly.react('hmChart', [trace], {height: Math.max(260, 18 * ys.length + 90), margin: {l: 60, r: 20, t: 10, b: 40},
    plot_bgcolor: SURF, paper_bgcolor: '#fff', font: {family: 'Inter, Segoe UI, Arial, sans-serif', color: INK2, size: 11},
    xaxis: {type: 'category', tickvals: xs.filter((_, k) => k % 5 === 0), ticktext: xs.filter((_, k) => k % 5 === 0).map(d => d.slice(5))},
    yaxis: {type: 'category', automargin: true}}, {displaylogo: false, responsive: true});
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
      h.v === null ? 'not yet available' : pct(h.v), pct(dayRet(s.c, h.i))]}); });
  const hover = '<b>%{customdata[0]}</b> - %{customdata[1]}<br>%{x|%b %d, %Y} · %{customdata[2]}<br>' +
    `New ATHs (${RULES.confirm.window_days}D): %{customdata[3]} · %{customdata[4]} below ATH<br>RVOL %{customdata[5]}x · RSI %{customdata[6]} · day %{customdata[8]}<br>` +
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
    (type === 'end' ? `<div class="warn">For run ends, <b>negative</b> forward returns mean the end signal caught a real peak.</div>` : '');
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
function renderAll() {
  RULES = readRules(); ensureStates();
  const a = asOf(), v = view();
  renderEmail(); renderTiles(a, v);
  const lb = renderLeaderboard(a, v);
  if (!S[$('rtTicker').value.trim().toUpperCase()] && lb.length) $('rtTicker').value = lb[0].t;
  renderTracker(); renderHeatmap(a, v, lb); renderBacktest(v); postHeight();
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
  setRules(D.rules); fillTickerList(); setupPushLinks();
  try { const d = localStorage.getItem('athRulesDraft');
    if (d && d !== rulesKey(D.rules)) { $('restoreRules').style.display = '';
      $('restoreRules').onclick = () => { setRules(clampRules(JSON.parse(d))); renderAll(); }; } } catch (e) {}
  renderAll();
  document.querySelectorAll('#emailCard input, #emailCard select').forEach(el => el.addEventListener('change', renderAll));
  ['asOf', 'hitType', 'fwdN', 'fwdSign'].forEach(id => $(id).addEventListener('change', renderAll));
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
